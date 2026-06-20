"""
Jitted per-block AMR kernels for the 3D node-centered Llama AMR.

Fixed-shape, never-recompiling building blocks: node-centered prolongation,
injection restriction, and the refinement indicator.  Ghost-zone sync lives in
``sync.py``.

Node-centered transfer (vs the 2D cell-centered original)
---------------------------------------------------------
2:1 refinement keeps every coarse node in place and inserts a midpoint in each
coarse cell.  The 1D refinement operator is therefore:

    fine[2k]   = coarse[k]                       (coincident node — EXACT copy)
    fine[2k+1] = Σ_j W_MID[j] coarse[k+off[j]]   (midpoint, 6-pt → 6th order)

with ``off = (-2,-1,0,1,2,3)`` and ``W_MID`` the Lagrange weights at offset +0.5.
Prolongation is the tensor product over the three axes; restriction is injection
(``[::2,::2,::2]`` of the fine interior) — exact at the coincident nodes, the
simplest high-order fine→coarse map (Galerkin adjoint deferred).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .state import BS, NG, NF, REFINE_RATIO


# ── Lagrange weights ────────────────────────────────────────────────────────────

def _lagrange_weights(offsets, target):
    """Lagrange weights to evaluate a polynomial at ``target`` from values at
    integer ``offsets`` (6 points → 5th-degree → 6th-order accurate)."""
    w = []
    for j, xj in enumerate(offsets):
        p = 1.0
        for m, xm in enumerate(offsets):
            if m != j:
                p *= (target - xm) / (xj - xm)
        w.append(p)
    return np.array(w, dtype=np.float64)


_MID_OFFSETS = (-2, -1, 0, 1, 2, 3)
_W_MID = jnp.asarray(_lagrange_weights(_MID_OFFSETS, 0.5))   # midpoint of nodes 0,1


# ── 1-D node-centered prolongation (N coarse nodes → 2N fine nodes) ─────────────

def _interp_to_fine_1d(coarse: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Refine ``coarse`` along ``axis`` 2:1, node-centered: even fine nodes copy
    the coincident coarse node, odd fine nodes are the 6-point midpoint interp."""
    N = coarse.shape[axis]
    pw = [(0, 0)] * coarse.ndim
    pw[axis] = (2, 3)                 # reach of _MID_OFFSETS = (-2..+3)
    padded = jnp.pad(coarse, pw, mode='edge')
    # midpoint between coarse[k] and coarse[k+1]; coarse[k+off] is padded[k+2+off]
    mids = 0.0
    for w, off in zip(_W_MID, _MID_OFFSETS):
        mids = mids + w * jax.lax.slice_in_dim(padded, 2 + off, 2 + off + N, axis=axis)
    ev = jnp.expand_dims(coarse, axis + 1)   # fine[2k]   = coarse[k]
    od = jnp.expand_dims(mids,   axis + 1)   # fine[2k+1] = midpoint
    inter = jnp.concatenate([ev, od], axis=axis + 1)
    new_shape = list(coarse.shape)
    new_shape[axis] = 2 * N
    return inter.reshape(new_shape)


# ── Prolongation: parent → full child block (interior + halo) ──────────────────

@partial(jax.jit, static_argnames=('child_corner',))
def prolongate(parent_block: jnp.ndarray, child_corner: tuple) -> jnp.ndarray:
    """Generate a child octant's FULL block (interior + NG halo) from its parent.

    Args:
      parent_block: (NF, W, W, W) with W = BS+2NG — parent data including halo.
      child_corner: (cx, cy, cz) ∈ {0,1}³ — which octant of the parent interior.

    Refines the full parent (correctness-first; the 2D windowing optimisation that
    refines only the child footprint is a deferred perf upgrade — it matters only
    at regrid / cross-level fill, not per step) then crops the child's window.

    With the full parent refined, child full-block index ff maps to refined index
    NG + corner*BS + ff (the even/copy alignment makes the child interior start
    land on a coincident coarse node), so the crop is a fixed slice.
    """
    r = parent_block
    for ax in (1, 2, 3):
        r = _interp_to_fine_1d(r, axis=ax)        # (NF, 2W, 2W, 2W)
    w = BS + 2 * NG
    cx, cy, cz = child_corner
    s = (NG + cx * BS, NG + cy * BS, NG + cz * BS)
    return jax.lax.slice(
        r, (0, s[0], s[1], s[2]), (NF, s[0] + w, s[1] + w, s[2] + w))


def prolongate_dynamic(parent_block: jnp.ndarray, child_c: jnp.ndarray) -> jnp.ndarray:
    """Like :func:`prolongate` but with a TRACED corner (``child_c`` = (cx,cy,cz)
    int array) so it can be vmapped over slots in the cross-level halo fill."""
    r = parent_block
    for ax in (1, 2, 3):
        r = _interp_to_fine_1d(r, axis=ax)
    w = BS + 2 * NG
    c = child_c.astype(jnp.int32)
    start = (jnp.int32(0),
             jnp.int32(NG) + c[0] * jnp.int32(BS),
             jnp.int32(NG) + c[1] * jnp.int32(BS),
             jnp.int32(NG) + c[2] * jnp.int32(BS))
    return jax.lax.dynamic_slice(r, start, (NF, w, w, w))


# ── Injection restriction (fine interior → coarse, coincident nodes) ───────────

def _restrict_interior_1d(child_full: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Stride-2 injection of the fine INTERIOR (size BS) → BS//2 coarse nodes.

    Coarse node c sits at fine interior index 2c = full-block index NG+2c (the
    coincident node), so injection is an exact strided gather."""
    nc = BS // REFINE_RATIO
    return jax.lax.slice_in_dim(child_full, NG, NG + REFINE_RATIO * nc,
                                stride=REFINE_RATIO, axis=axis)


@partial(jax.jit, static_argnames=('child_corner',))
def restrict_into_parent(parent_block: jnp.ndarray, child_block: jnp.ndarray,
                         child_corner: tuple) -> jnp.ndarray:
    """Inject one child's interior into the matching octant of the parent interior.

    Returns the updated parent block (halo untouched).  child_corner picks which
    (BS//2)³ sub-region of the parent interior the child covers.
    """
    rc = child_block
    for ax in (1, 2, 3):
        rc = _restrict_interior_1d(rc, axis=ax)   # (NF, BS/2, BS/2, BS/2)
    half = BS // REFINE_RATIO
    cx, cy, cz = child_corner
    i0 = NG + cx * half
    j0 = NG + cy * half
    k0 = NG + cz * half
    return jax.lax.dynamic_update_slice(parent_block, rc, (0, i0, j0, k0))


# ── Refinement indicator ───────────────────────────────────────────────────────

@jax.jit
def compute_indicator_gradient(block: jnp.ndarray) -> jnp.ndarray:
    """Scalar refinement indicator per block: the max over interior nodes and
    fields of the undivided first-difference magnitude (a cheap |∇u|·Δx proxy,
    centering- and metric-agnostic).  Shape () scalar.

    Computed on the interior plus a one-node ring (requires NG ≥ 1) so every
    interior-adjacent difference is captured."""
    core = block[:, NG-1:NG+BS+1, NG-1:NG+BS+1, NG-1:NG+BS+1]
    g = jnp.asarray(0.0)
    for ax in (1, 2, 3):
        g = jnp.maximum(g, jnp.max(jnp.abs(jnp.diff(core, axis=ax))))
    return g
