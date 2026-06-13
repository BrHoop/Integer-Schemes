"""Increment 1 of the fp32 tiled SMEM-halo fused BSSN kernel (3.2d-GPU).

The whole-grid fused prototypes (`_bssn_rhs_fused*`) do NOT lower under Triton (array
shapes not power-of-2; H200 2026-06-13). This module de-risks the hard Triton-0.9.2
lowering on the smallest surface: the **on-chip derivative stage alone**, computed
over power-of-2 haloed tiles via the proven `pallas_ozaki` pattern, before the 288-temp
algebra is fused on top (increment 2).

Pattern (3D generalization of `mcs2d.schemes.pallas_ozaki`):
  * **Wrapper (outside the kernel — `dynamic_slice` is allowed there):** edge-pad the
    grid by the halo, `vmap(dynamic_slice)` to cut OVERLAPPING haloed tiles of pow2 size
    HP (= BS + 2·NG, padded to a power of 2), stack `(n_tiles, NF, HP, HP, HP)`.
  * **Kernel (one tile/block):** every derivative is `apply(Mx,0)∘apply(My,1)∘apply(Mz,2)`
    of three per-axis `(BS, HP)` matrices — `D1`/`D2` (the FD stencil, which also crops to
    the BS interior) or `ID` (one-hot interior select). All matmuls are **fp32** (Triton-
    safe; fp64 `dot` hangs the compiler). No slice/gather/`jnp.pad` inside the kernel.

This is **interpret-validated math only** here (CPU); the power-of-2 / SMEM / fp32-dot
lowering is the H200 gate. SMEM budget (138 derivs×BS³ + halos×HP³ vs 228 KB) is the next
tuning axis and only bites on GPU — increment 1 just proves the tiling + crop + stencil.

Run:  `python -m bssn3d.tiled_deriv`  (CPU interpret self-check vs derivative_bundle).
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from mcs_common.derivatives import SpatialDerivative
from .state import BSSNState, NAME_TO_IDX
from ._bssn_rhs_generated import FIELD_INPUTS, GRAD1_INPUTS, GRAD2_INPUTS

if not os.environ.get("JAX_PALLAS_USE_MOSAIC_GPU"):
    jax.config.update("jax_pallas_use_mosaic_gpu", False)

GRAD1_NAMES = [f"grad_{a}_{f}" for a, f in GRAD1_INPUTS]
GRAD2_NAMES = [f"grad2_{i}_{j}_{f}" for i, j, f in GRAD2_INPUTS]
_ALL_DERIV_NAMES = GRAD1_NAMES + GRAD2_NAMES

# The kernel unrolls every derivative at trace time → ~3 ops each → the full 138 is a
# large straight-line kernel with a long Triton compile (the documented ~1600 s Pallas
# wall). BSSN_TILE_NDERIV caps the count for a FAST lowering smoke-test (e.g. =6 compiles
# in ~1-2 min); default = all 138. Caps the kernel only, not correctness of what's emitted.
_NLIM = int(os.environ.get("BSSN_TILE_NDERIV", str(len(_ALL_DERIV_NAMES))))
DERIV_NAMES = _ALL_DERIV_NAMES[:_NLIM]
NDERIV = len(DERIV_NAMES)

BS = int(os.environ.get("BSSN_TILE_BS", "8"))          # interior tile side (pow2)
# Derivatives default to fp64 (cancellation-accurate; broadcast-reduce lowers fp64 under
# Triton — only fp64 `dot` hangs). BSSN_PALLAS_FP32=1 forces the fast/lossy fp32 path.
_PDTYPE = jnp.float32 if os.environ.get("BSSN_PALLAS_FP32", "0") != "0" else jnp.float64
_INTERPRET = jax.default_backend() == "cpu"


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _axis_matrices(order: int, d, BS: int, HP: int, NG: int):
    """Per-axis (BS, HP) operator matrices, as a numpy array stacked
    [D1_x, D1_y, D1_z, D2_x, D2_y, D2_z, ID].

    Interior point j (0..BS-1) sits at tile index NG+j; its stencil window is tile
    indices [j, j+2NG]. D1/D2 carry the FD coeffs / spacing; ID selects NG+j. The
    matmul `M @ tile_axis` thus differentiates AND crops HP→BS in one op.
    """
    diff = SpatialDerivative(order=order)
    c1 = np.asarray(diff.C1, dtype=np.float64)          # length 2NG+1
    c2 = np.asarray(diff.C2, dtype=np.float64)
    mats = []
    for a in range(3):                                   # D1 along each axis
        M = np.zeros((BS, HP))
        for j in range(BS):
            M[j, j:j + 2 * NG + 1] = c1 / d[a]
        mats.append(M)
    for a in range(3):                                   # D2 along each axis
        M = np.zeros((BS, HP))
        for j in range(BS):
            M[j, j:j + 2 * NG + 1] = c2 / (d[a] ** 2)
        mats.append(M)
    ID = np.zeros((BS, HP))                               # interior select
    for j in range(BS):
        ID[j, NG + j] = 1.0
    mats.append(ID)
    return np.stack(mats, axis=0)                         # (7, BS, HP)


# PRECISION: the derivative stencil is cancellation-sensitive (the C2 = [2,−27,270,−490,…]
# 2nd-deriv terms), so fp32 derivatives lose ~4 digits (measured ~1e-4 on H200). The fix is
# NOT mean-subtraction (the O(1) fields make the subtraction itself lose fp32 precision) but
# computing the derivatives in **fp64**: this kernel uses broadcast-multiply-reduce, and only
# fp64 `dot` hangs Triton — fp64 elementwise/reduce lowers fine. So the production fused
# kernel = fp64 derivatives (here) → cast to fp32 → fp32 algebra = the validated
# `fp32_contraction` precision. Default fp64; BSSN_PALLAS_FP32=1 forces the fast/lossy path.

# matrix-stack row indices (D1 per axis, D2 per axis, ID interior-select)
_D1 = (0, 1, 2)
_D2 = (3, 4, 5)
_ID = 6


def _per_field_ops():
    """For each named derivative, the field and the 3 per-axis matrix choices (rows of the
    matrix stack)."""
    ops = {}
    for a, f in GRAD1_INPUTS:
        sel = [_ID, _ID, _ID]
        sel[a] = _D1[a]
        ops[f"grad_{a}_{f}"] = (f, tuple(sel))
    for i, j, f in GRAD2_INPUTS:
        sel = [_ID, _ID, _ID]
        if i == j:
            sel[i] = _D2[i]
        else:
            sel[i] = _D1[i]
            sel[j] = _D1[j]
        ops[f"grad2_{i}_{j}_{f}"] = (f, tuple(sel))
    return ops


_OPS = _per_field_ops()


def _apply_axis(M, x, ax):
    """Contract (BS, HP) `M` with axis `ax` of `x` → BS replaces that axis:
    ``out[..,j,..] = Σ_k M[j,k]·x[..,k,..]``.

    Broadcast-multiply-reduce (NOT `dot`/transpose): Triton 0.9.2 lowers only 2D×2D `dot`
    and a middle-axis 3D contraction would need a transpose, whereas multiply +
    `reduce_sum` + broadcast are the well-lowered primitives the proven `pallas_ozaki._mid`
    crop uses. Insert a BS-slot at `ax` (HP shifts to ax+1), align M, reduce the HP axis."""
    M = M.astype(x.dtype)
    xe = jnp.expand_dims(x, ax)                 # BS-slot at ax; HP now at ax+1
    mshape = [1] * (x.ndim + 1)
    mshape[ax] = M.shape[0]
    mshape[ax + 1] = M.shape[1]
    return (xe * M.reshape(mshape)).sum(axis=ax + 1)


def _derivative(tile, mats, axis_ops):
    """One derivative of a single field `tile` (HP,HP,HP): apply the chosen per-axis matrix
    along axes 2,1,0 → (BS,BS,BS). Per-field (NOT batched over fields): batching multiplies
    the un-reduced broadcast transient by NF → ~16 MB/op, which explodes compile + spills.
    Per-field keeps each transient ~256 KB; the field selection is a ref-load (`tile_ref[0,
    fidx]`), not a computed-array slice."""
    x = tile
    for ax in (2, 1, 0):
        x = _apply_axis(mats[axis_ops[ax]], x, ax)
    return x


def _kernel(tile_ref, mat_ref, out_ref):
    # tile_ref: (1, NF, HP, HP, HP) one haloed tile; mat_ref: (7, BS, HP);
    # out_ref: (1, NDERIV, BS, BS, BS). The leading 1 is the per-block n_tiles slot.
    mats = [mat_ref[r] for r in range(7)]
    for k, name in enumerate(DERIV_NAMES):
        f, axis_ops = _OPS[name]
        out_ref[0, k] = _derivative(tile_ref[0, FIELD_INPUTS.index(f)], mats, axis_ops)


def tiled_derivative_bundle(state: BSSNState, order: int, dx, dy, dz):
    """Compute the 138 derivatives on-chip over power-of-2 haloed tiles → dict
    {grad_name: (Nx,Ny,Nz) interior array}. Validates the increment-1 kernel against
    `derivative_bundle` (interpret mode). Requires N divisible by BS on each axis."""
    NG = order // 2
    HP = _next_pow2(BS + 2 * NG)
    data = state.data                                    # (NF, Sx, Sy, Sz), Sx = N+2*ng
    NF, Sx, Sy, Sz = data.shape
    ng = NG
    N = (Sx - 2 * ng, Sy - 2 * ng, Sz - 2 * ng)
    assert all(n % BS == 0 for n in N), f"interior {N} not divisible by BS={BS}"
    nt = [n // BS for n in N]
    n_tiles = nt[0] * nt[1] * nt[2]

    mats = jnp.asarray(_axis_matrices(order, (dx, dy, dz), BS, HP, NG), dtype=_PDTYPE)
    extra = HP - (BS + 2 * NG)                            # pow2 slack beyond the halo
    # edge-pad so every tile (incl. boundary) has a full HP-wide window available
    padded = jnp.pad(data, ((0, 0), (0, extra), (0, extra), (0, extra)), mode="edge")

    starts = jnp.stack(jnp.meshgrid(*[jnp.arange(n, dtype=jnp.int32) for n in nt],
                                    indexing="ij"), axis=-1).reshape(-1, 3)

    def cut_field(uf):                                   # uf: (Sx+extra, ...)
        def one(s):
            return jax.lax.dynamic_slice(uf, (s[0] * BS, s[1] * BS, s[2] * BS),
                                         (HP, HP, HP))
        return jax.vmap(one)(starts)                     # (n_tiles, HP,HP,HP)

    tiles = jax.vmap(cut_field)(padded).transpose(1, 0, 2, 3, 4).astype(_PDTYPE)
    # tiles: (n_tiles, NF, HP, HP, HP). The kernel indexes single fields (a ref load), so
    # the NF axis need not be power-of-2 — no NF_PAD needed for the unrolled per-field kernel.

    out = pl.pallas_call(
        _kernel,
        grid=(n_tiles,),
        in_specs=[pl.BlockSpec((1, NF, HP, HP, HP), lambda i: (i, 0, 0, 0, 0)),
                  pl.BlockSpec((7, BS, HP), lambda i: (0, 0, 0))],
        out_specs=pl.BlockSpec((1, NDERIV, BS, BS, BS), lambda i: (i, 0, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((n_tiles, NDERIV, BS, BS, BS), _PDTYPE),
        interpret=_INTERPRET,
    )(tiles, mats)                                       # (n_tiles, NF, HP,HP,HP)

    # reassemble tiles → interior grid
    grid = out.reshape(nt[0], nt[1], nt[2], NDERIV, BS, BS, BS)
    grid = grid.transpose(3, 0, 4, 1, 5, 2, 6).reshape(NDERIV, N[0], N[1], N[2])
    return {name: grid[k].astype(jnp.float64) for k, name in enumerate(DERIV_NAMES)}


def _selfcheck():
    jax.config.update("jax_enable_x64", True)
    from .grid import Grid
    from . import initial_data as bid
    from .derivative_bundle import derivative_bundle
    order = 6
    g = Grid.from_domain(16, order=order)               # N=16, BS=8 → 2 tiles/axis
    s = bid.gauge_wave(g, amplitude=0.02)
    diff = SpatialDerivative(order=order)
    ref = derivative_bundle(s, diff, g.dx, g.dy, g.dz)
    got = tiled_derivative_bundle(s, order, g.dx, g.dy, g.dz)
    ng = order // 2
    worst = 0.0
    for name in DERIV_NAMES:
        r = np.asarray(ref[name])[ng:-ng, ng:-ng, ng:-ng]   # physical interior
        gg = np.asarray(got[name])
        worst = max(worst, float(np.max(np.abs(r - gg))))
    print(f">> tiled_derivative_bundle vs derivative_bundle (interior): max|Δ| = {worst:.2e}")
    print(f"   BS={BS} HP={_next_pow2(BS + 2*ng)} dtype={_PDTYPE.__name__} NDERIV={NDERIV}")


if __name__ == "__main__":
    _selfcheck()
