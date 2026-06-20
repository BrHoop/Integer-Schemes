"""
Per-block curvilinear geometry for the AMR hierarchy.

A fine AMR block covers a logical sub-cube of its patch at a refined spacing.
Because every patch carries a *closed-form* logical→world map, the block's
geometry (``jinv``, ``d2coef``) is built **analytically** at the block's fine
nodes — there is no metric finite-differencing and no special-casing of
curvature.  We simply construct an :class:`atlas.Patch` for the block's logical
window at the block's resolution and let its ``__post_init__`` build the geometry
(the same machinery the single-level grid uses).  This works uniformly for the
affine cube (where it reduces ``CurvilinearDerivative`` to uniform FD) and the
curvilinear shells.

Block ↔ global-node alignment: a level-``L`` block whose interior starts at
level-``L`` node index ``bbox_ijk`` sits at logical
``lo = parent.lo + bbox_ijk · h_L`` with ``h_L = base_dxi / 2**L``, and spans
``BS`` nodes at spacing ``h_L`` — i.e. exactly the global level-``L`` node grid,
so fine nodes coincide with coarse nodes (the node-centered property).
"""
import jax.numpy as jnp

from multipatch.atlas import Patch
from multipatch import coord_maps as cm

from .state import BS, NG


def level_geometry(parent_patch):
    """Compact, **recompute-don't-store** geometry for an AMR patch.

    For an AFFINE patch (the central cube) the inverse Jacobian is a single
    constant — ``jinv = I / world_scale`` — identical at every node, block, and
    refinement level (the affine map's Jacobian w.r.t. the patch-logical
    coordinate is constant; level only changes the FD spacing ``dxi``, not
    ``jinv``), and ``d2coef = 0``.  So instead of storing ``(slots,3,3,W,W,W)``
    /``(slots,3,3,3,W,W,W)`` arrays per level (the 3.6×-fields memory sink), we
    carry two tiny constants that broadcast inside ``CurvilinearDerivative``.

    Curvilinear shells (Phase B) recompute full per-node geometry from the map
    inside the kernel (transient, never stored) — that hook replaces this
    affine fast-path when ``patch_type != PATCH_AFFINE``.

    Returns ``(jinv (3,3), d2coef (3,3,3))``.
    """
    if int(parent_patch.patch_type) != cm.PATCH_AFFINE:
        raise NotImplementedError(
            "level_geometry is the affine (cube) fast-path; curvilinear "
            "per-node recompute is Phase B (shells).")
    world_scale = parent_patch.patch_params[3]               # = 2a for the cube
    jinv = jnp.eye(3, dtype=jnp.float64) / world_scale
    d2coef = jnp.zeros((3, 3, 3), dtype=jnp.float64)
    return jinv, d2coef


def level_spacing(parent_patch, level: int) -> tuple:
    """Logical node spacing of ``parent_patch`` at refinement ``level``."""
    return tuple(parent_patch.dxi[a] / (2 ** level) for a in range(3))


def block_patch(parent_patch, level: int, bbox_ijk: tuple, name: str = "blk") -> Patch:
    """An :class:`atlas.Patch` for one AMR block's logical window + resolution."""
    h = level_spacing(parent_patch, level)
    lo = tuple(parent_patch.lo[a] + bbox_ijk[a] * h[a] for a in range(3))
    hi = tuple(lo[a] + (BS - 1) * h[a] for a in range(3))
    return Patch(name=name,
                 patch_type=parent_patch.patch_type,
                 patch_params=parent_patch.patch_params,
                 N=(BS, BS, BS), ng=NG, lo=lo, hi=hi)


def block_geometry(parent_patch, level: int, bbox_ijk: tuple):
    """``(jinv, d2coef)`` arrays for one block (shapes (3,3,W,W,W) / (3,3,3,W,W,W))."""
    p = block_patch(parent_patch, level, bbox_ijk)
    return p.jinv, p.d2coef
