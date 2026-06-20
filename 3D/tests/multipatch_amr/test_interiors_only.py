"""Phase 2 — interiors-only storage.

AMR blocks store the BS³ interior only; the NG-layer halo is a transient working
buffer rebuilt each RHS substage. These lock in the storage shape and the
persistent-footprint reduction (the halo was an ``((BS+2NG)/BS)³`` inflation —
8× at BS=8, NG=4).
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from multipatch import atlas, coord_maps as cm
from multipatch.wave import plane_wave_state
from multipatch.amr.state import BS, NG, NF, LEVELS, make_root_state

A_CUBE = 1.0
W = BS + 2 * NG


def _root_state():
    cube = atlas.Patch(
        name="cube", patch_type=cm.PATCH_AFFINE,
        patch_params=cm.make_affine_params(
            world_origin=(-A_CUBE,) * 3, world_scale=2.0 * A_CUBE),
        N=(BS, BS, BS), ng=NG, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))
    F0 = plane_wave_state(cube.X, cube.Y, cube.Z, 0.0, k=(0.7, -0.4, 0.5))
    state, topo = make_root_state(F0, nb_root=(1, 1, 1))
    return cube, state


def test_blocks_are_interiors_only():
    _, state = _root_state()
    for L in range(LEVELS):
        assert state.blocks[L].shape[-3:] == (BS, BS, BS), \
            "blocks must store the BS³ interior, no persistent halo"
        assert state.blocks[L].shape[1] == NF


def test_persistent_footprint_reduction():
    """Stored field footprint dropped by the halo-inflation factor (W/BS)³."""
    _, state = _root_state()
    new = sum(int(np.prod(b.shape)) for b in state.blocks)
    old = sum(b.shape[0] * NF * W**3 for b in state.blocks)   # had been BS+2NG per axis
    factor = old / new
    assert factor > 1.5
    assert abs(factor - (W / BS) ** 3) < 1e-9
