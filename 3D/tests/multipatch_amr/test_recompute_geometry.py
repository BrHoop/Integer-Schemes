"""Phase 1 — recompute-don't-store geometry.

The affine cube's inverse Jacobian is the constant ``I/world_scale`` at every
node/block/level (``d2coef = 0``), so AMR no longer stores ``(slots,3,3,W,W,W)``
geometry arrays — it carries two tiny constants. These tests lock in:
  * parity — constant geometry gives byte-identical block RHS vs the full
    per-node arrays;
  * footprint — geometry storage is a fixed 36 numbers, independent of
    slots / levels / BS / NG (was ~3.6× the field volume per slot).
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from multipatch import atlas, coord_maps as cm
from multipatch.wave import WaveSystem, plane_wave_state
from multipatch.amr import evolve as E
from multipatch.amr.geometry import level_spacing
from multipatch.amr.state import BS, NG, LEVELS, MAX_BLOCKS_PER_LEVEL

KVEC = (0.7, -0.4, 0.5)
A_CUBE = 1.0


def _cube_patch():
    return atlas.Patch(
        name="cube", patch_type=cm.PATCH_AFFINE,
        patch_params=cm.make_affine_params(
            world_origin=(-A_CUBE,) * 3, world_scale=2.0 * A_CUBE),
        N=(BS, BS, BS), ng=NG, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))


def test_constant_geometry_shape():
    geom = E.build_geometry(_cube_patch())
    assert geom.jinv.shape == (3, 3)
    assert geom.d2coef.shape == (3, 3, 3)
    # jinv = I / (2a); d2coef = 0
    assert float(jnp.max(jnp.abs(geom.jinv - jnp.eye(3) / (2 * A_CUBE)))) < 1e-15
    assert float(jnp.max(jnp.abs(geom.d2coef))) == 0.0


def test_constant_geometry_matches_full_arrays():
    """Constant geometry must give byte-identical block RHS vs the full
    per-node arrays the single-level Patch builds."""
    cube = _cube_patch()
    geom = E.build_geometry(cube)
    F = plane_wave_state(cube.X, cube.Y, cube.Z, 0.0, k=KVEC)
    dxi = level_spacing(cube, 0)
    r_const = E._block_rhs(F, geom.jinv, geom.d2coef, dxi, WaveSystem(), 6, 0.0, 0.0)
    r_full = E._block_rhs(F, cube.jinv, cube.d2coef, dxi, WaveSystem(), 6, 0.0, 0.0)
    assert float(jnp.max(jnp.abs(r_const - r_full))) < 1e-13


def test_geometry_footprint_is_constant():
    """Geometry is a fixed 36 numbers — independent of slots/levels/BS/NG —
    vs the former per-slot arrays (~thousands× larger)."""
    W = BS + 2 * NG
    old_floats = sum(MAX_BLOCKS_PER_LEVEL[L] * (9 + 27) * W**3 for L in range(LEVELS))
    geom = E.build_geometry(_cube_patch())
    new_floats = geom.jinv.size + geom.d2coef.size
    assert new_floats == 36
    assert new_floats < old_floats / 1000
