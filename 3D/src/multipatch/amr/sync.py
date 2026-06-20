"""
Ghost-zone synchronization for the 3D node-centered Llama AMR.

Three fills, mirroring the 2D module (``mcs2d/amr/kernels.py``) generalized to 3D
and to NON-periodic patch interiors:

  * :func:`sync_within_level_root` — regular nbx×nby×nbz root tiling of ONE patch:
    stitch interiors into the patch-global node grid, edge-pad, re-extract.  Fills
    every inter-block halo (faces + edges + corners) in one shot.  The patch's
    OUTER ghosts are edge-padded here and then overwritten by the inter-patch
    overlap fill / outer BC — they are NOT periodic (a patch is not a torus).

  * :func:`sync_within_level` — general 6-face copy between same-level neighbours
    (for irregular fine levels).  Faces 0..5 = (-x,+x,-y,+y,-z,+z).

  * :func:`sync_across_levels` — fill each fine block's full halo by prolongating
    its parent (covers faces/edges/corners); same-level neighbour faces are then
    overwritten with exact data by :func:`sync_within_level`.

With disjoint BS-owned blocks the ghost geometry is identical to the 2D
cell-centered layout — node-centering only changes the transfer stencils
(``kernels.py``), not the copy specs here.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .state import BS, NG, NF, NFACE
from .kernels import prolongate_dynamic

W = BS + 2 * NG


# ── Root-level within-patch sync (regular tiling, non-periodic) ────────────────

@partial(jax.jit, static_argnames=('nbx', 'nby', 'nbz'))
def sync_within_level_root(interiors_l0, nbx: int, nby: int, nbz: int):
    """Build the haloed root working buffer from interiors-only storage.

    Input ``interiors_l0`` is (caps0, NF, BS, BS, BS) (slot = (bi*nby+bj)*nbz+bk).
    Stitches the BS³ interiors into the contiguous (NF, nbx*BS, nby*BS, nbz*BS)
    patch-global node grid, edge-pads by NG, and extracts per-block (NF,W,W,W)
    HALOED tiles.  Inter-block halos (incl. edges/corners) are exact; patch-
    boundary halos are edge-replicated placeholders (overwritten later by
    overlap/BC).  Returns (caps0, NF, W, W, W).
    """
    n_root = nbx * nby * nbz
    caps0 = interiors_l0.shape[0]
    interiors = interiors_l0[:n_root]
    interiors = interiors.reshape(nbx, nby, nbz, NF, BS, BS, BS)
    # (nbx,nby,nbz,NF,bx,by,bz) → (NF, nbx,bx, nby,by, nbz,bz) → global grid
    g = jnp.transpose(interiors, (3, 0, 4, 1, 5, 2, 6))
    g = g.reshape(NF, nbx * BS, nby * BS, nbz * BS)
    g = jnp.pad(g, ((0, 0), (NG, NG), (NG, NG), (NG, NG)), mode='edge')

    bi = jnp.arange(nbx); bj = jnp.arange(nby); bk = jnp.arange(nbz)

    def extract(i, j, k):
        return jax.lax.dynamic_slice(g, (0, i*BS, j*BS, k*BS), (NF, W, W, W))

    ex = jax.vmap(jax.vmap(jax.vmap(extract, (None, None, 0)),
                           (None, 0, None)), (0, None, None))
    tiles = ex(bi, bj, bk).reshape(n_root, NF, W, W, W)
    # haloed working buffer (caps0, NF, W, W, W); inactive/non-root slots stay zero
    out = jnp.zeros((caps0, NF, W, W, W), tiles.dtype)
    return out.at[:n_root].set(tiles)


# ── Cross-level halo fill (parent prolongated into child halo) ─────────────────

@jax.jit
def sync_across_levels(child_blocks, parent_blocks, parent_slot, child_c, active):
    """Fill each active child's HALO from a prolongation of its parent; interiors
    are preserved.  Inactive slots are returned unchanged.

    child_c: (slots, 3) int32 octant index.  parent_slot: (slots,) int32.
    """
    parents = parent_blocks[parent_slot]
    prolonged = jax.vmap(prolongate_dynamic)(parents, child_c)

    halo = jnp.ones((W, W, W), bool)
    halo = halo.at[NG:NG+BS, NG:NG+BS, NG:NG+BS].set(False)
    write = active[:, None, None, None, None] & halo[None, None]
    return jnp.where(write, prolonged, child_blocks)


# ── Within-level 6-face copy between same-level neighbours ──────────────────────
# Per face: (dst_slice on each of x,y,z) ← (src_slice from neighbour interior).
# Each entry is (axis, dst_lo, dst_hi, src_lo, src_hi); unlisted axes use the
# interior extent [NG:NG+BS].  Faces 0..5 = (-x,+x,-y,+y,-z,+z).
_FACE = (
    (0, 0,      NG,        BS,     BS + NG),   # -x: my low-x halo ← nbr high-x interior
    (0, NG+BS,  2*NG+BS,   NG,     2*NG),      # +x
    (1, 0,      NG,        BS,     BS + NG),   # -y
    (1, NG+BS,  2*NG+BS,   NG,     2*NG),      # +y
    (2, 0,      NG,        BS,     BS + NG),   # -z
    (2, NG+BS,  2*NG+BS,   NG,     2*NG),      # +z
)


@jax.jit
def sync_within_level(blocks, neighbor_slot, neighbor_valid):
    """Fill each block's FACE halos from same-level neighbours' interiors (exact).

    Faces with no neighbour (patch boundary / cross-level) are left unchanged so a
    preceding :func:`sync_across_levels` still provides them.  Edge/corner halos
    are not written here (the wave system's first-derivative stencils are separable
    → read only face halos); for the mixed-derivative MCS/BSSN stencils the
    cross-level prolongation supplies approximate edge/corner ghosts.
    """
    out = blocks
    interior = slice(NG, NG + BS)
    for f, (ax, dlo, dhi, slo, shi) in enumerate(_FACE):
        nb = blocks[neighbor_slot[:, f]]
        dst = [slice(None), slice(None), interior, interior, interior]
        src = [slice(None), slice(None), interior, interior, interior]
        dst[2 + ax] = slice(dlo, dhi)
        src[2 + ax] = slice(slo, shi)
        valid = neighbor_valid[:, f][:, None, None, None, None]
        out = out.at[tuple(dst)].set(jnp.where(valid, nb[tuple(src)], out[tuple(dst)]))
    return out
