"""Inter-patch coupling: overlap ghost fill via precomputed Lagrange interpolation.

This is the heart of the Llama multipatch method. Every patch carries ``ng``
ghost layers on each face. A ghost node that physically lands inside another
patch's *interior* is filled by interpolating that donor patch's field at the
ghost's world location; a ghost with no donor is an outer-boundary node (handled
by the outer BC, not here).

Two phases (mirrors dendrojax ``PatchOverlapTable`` / the 2D-AMR ``prolongate``):

* **Setup (host, once)** — :func:`build_overlap_table`. For every receiver ghost
  with a donor: world coord -> donor patch (``dispatch_contains``) -> donor
  logical coord (``dispatch_inverse_map``) -> enclosing donor cell + tensor-
  product Lagrange weights (``m = order + 1`` points -> degree-``order`` interp,
  matching the FD truncation order). Stencils are clamped to the donor's interior
  so we never read a donor's own ghosts (the Pollney-Llama no-ghost-from-ghost
  rule). When several patches contain a ghost (triple-overlap corners), the donor
  in which the point is most *interior* (largest stencil margin) is chosen.

* **Per step** — :func:`apply_overlap_fill`. A batched gather of ``m^3`` donor
  blocks contracted with the precomputed weights, scattered into the receiver
  ghosts. Pure JAX, fixed shapes -> jittable, no recompiles.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import coord_maps as cm

_ZERO = jnp.zeros((3,), dtype=cm.GEO_DTYPE)
_ONE = jnp.asarray(1.0, dtype=cm.GEO_DTYPE)


def _lagrange_weights(nodes_rel, target):
    """Lagrange weights at ``target`` for nodes at integer offsets ``nodes_rel``."""
    m = len(nodes_rel)
    w = np.ones((target.shape[0], m), dtype=np.float64)
    for j in range(m):
        for k in range(m):
            if k != j:
                w[:, j] *= (target - nodes_rel[k]) / (nodes_rel[j] - nodes_rel[k])
    return w


@dataclass
class _Entry:
    """Ghost-fill data for one (receiver, donor) pair (disjoint ghost subset)."""
    recv: int
    donor: int
    tgt_idx: jnp.ndarray     # (G,) int32  flat index into receiver spatial grid
    base: jnp.ndarray        # (G, 3) int32  donor array base index of the stencil
    w0: jnp.ndarray          # (G, m) float64  weights along donor axis 0
    w1: jnp.ndarray          # (G, m)
    w2: jnp.ndarray          # (G, m)


@dataclass
class OverlapTable:
    m: int
    entries: list            # list[_Entry]
    # per-patch outer-boundary ghost flat indices (no donor; for the outer BC):
    boundary_idx: list       # list[jnp.ndarray]  (one (B_p,) array per patch


def _ghost_mask(shape, ng):
    nx, ny, nz = shape
    ix = np.arange(nx); iy = np.arange(ny); iz = np.arange(nz)
    gx = (ix < ng) | (ix >= nx - ng)
    gy = (iy < ng) | (iy >= ny - ng)
    gz = (iz < ng) | (iz >= nz - ng)
    IGX, IGY, IGZ = np.meshgrid(gx, gy, gz, indexing="ij")
    return IGX | IGY | IGZ


def build_overlap_table(grid, order: int = 6) -> OverlapTable:
    """Precompute the inter-patch ghost-fill table for a :class:`atlas.LlamaGrid`."""
    patches = grid.patches
    m = order + 1
    half = (m - 1) // 2
    npatch = len(patches)

    # donor metadata as numpy for host-side index/weight math
    donor_lo = [np.asarray(p.lo, dtype=np.float64) for p in patches]
    donor_dxi = [np.asarray(p.dxi, dtype=np.float64) for p in patches]
    donor_N = [np.asarray(p.N, dtype=np.int64) for p in patches]
    donor_ng = [p.ng for p in patches]

    entries = []
    boundary_idx = []

    for pr, P in enumerate(patches):
        gmask = _ghost_mask(P.shape, P.ng)            # (nx,ny,nz) bool
        flat = np.flatnonzero(gmask.ravel())          # (G,) flat spatial indices
        Xg = np.asarray(P.X).ravel()[flat]
        Yg = np.asarray(P.Y).ravel()[flat]
        Zg = np.asarray(P.Z).ravel()[flat]
        G = flat.shape[0]

        # per-ghost: best donor, its interior-index coords, and margin
        best_donor = np.full(G, -1, dtype=np.int64)
        best_margin = np.full(G, -np.inf, dtype=np.float64)
        best_t = np.zeros((G, 3), dtype=np.float64)

        Xj = jnp.asarray(Xg); Yj = jnp.asarray(Yg); Zj = jnp.asarray(Zg)
        for pd, D in enumerate(patches):
            if pd == pr:
                continue
            contains = jax.vmap(lambda a, b, c, _D=D: cm.dispatch_contains(
                _D.patch_type, _D.patch_params, a, b, c, _ZERO, _ONE))
            inv = jax.vmap(lambda a, b, c, _D=D: cm.dispatch_inverse_map(
                _D.patch_type, _D.patch_params, a, b, c, _ZERO, _ONE))
            inside = np.asarray(contains(Xj, Yj, Zj))
            if not inside.any():
                continue
            lx, ly, lz = inv(Xj, Yj, Zj)
            lvec = np.stack([np.asarray(lx), np.asarray(ly), np.asarray(lz)], axis=1)
            t = (lvec - donor_lo[pd]) / donor_dxi[pd]   # (G,3) interior-index coords
            N = donor_N[pd]
            # margin = how far the point sits from the interior edges (min over axes)
            margin = np.minimum(t, (N - 1) - t).min(axis=1)
            take = inside & (margin > best_margin)
            best_donor[take] = pd
            best_margin[take] = margin[take]
            best_t[take] = t[take]

        # genuine outer-boundary ghosts (no donor)
        bnd = best_donor < 0
        boundary_idx.append(jnp.asarray(flat[bnd], dtype=jnp.int32))

        # group assigned ghosts by donor -> one _Entry each
        for pd in range(npatch):
            sel = best_donor == pd
            if not sel.any():
                continue
            t = best_t[sel]                              # (g,3)
            N = donor_N[pd]
            ng_d = donor_ng[pd]
            i0 = np.floor(t).astype(np.int64) - half     # centred stencil start
            i0 = np.clip(i0, 0, (N - m))                 # keep stencil in interior
            nodes_rel = np.arange(m, dtype=np.float64)
            w = []
            for ax in range(3):
                w.append(_lagrange_weights(nodes_rel, t[:, ax] - i0[:, ax]))
            base = (i0 + ng_d).astype(np.int32)          # donor *array* base index
            entries.append(_Entry(
                recv=pr, donor=pd,
                tgt_idx=jnp.asarray(flat[sel], dtype=jnp.int32),
                base=jnp.asarray(base, dtype=jnp.int32),
                w0=jnp.asarray(w[0]), w1=jnp.asarray(w[1]), w2=jnp.asarray(w[2]),
            ))

    return OverlapTable(m=m, entries=entries, boundary_idx=boundary_idx)


def _fill_entry(F_donor, F_recv, base, w0, w1, w2, tgt_idx, m):
    # m is a static Python int (table.m); callers jit the enclosing step.
    NF = F_donor.shape[0]
    a = jnp.arange(m)
    ii = base[:, 0][:, None, None, None] + a[None, :, None, None]
    jj = base[:, 1][:, None, None, None] + a[None, None, :, None]
    kk = base[:, 2][:, None, None, None] + a[None, None, None, :]
    block = F_donor[:, ii, jj, kk]                       # (NF, g, m, m, m)
    vals = jnp.einsum("ngabc,ga,gb,gc->ng", block, w0, w1, w2)  # (NF, g)
    shp = F_recv.shape
    return F_recv.reshape(NF, -1).at[:, tgt_idx].set(vals).reshape(shp)


def apply_overlap_fill(fields, table: OverlapTable):
    """Fill every patch's donor-backed ghosts. ``fields`` is a list/tuple of
    per-patch arrays ``(NF, nx, ny, nz)``; returns the updated list.

    Outer-boundary ghosts (``table.boundary_idx``) are left untouched here — the
    caller applies the outer BC to them.
    """
    fields = list(fields)
    m = table.m
    for e in table.entries:
        fields[e.recv] = _fill_entry(
            fields[e.donor], fields[e.recv],
            e.base, e.w0, e.w1, e.w2, e.tgt_idx, m)
    return fields
