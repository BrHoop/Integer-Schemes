"""Hand-configured 2-level cube: a refined corner octant evolves with the parent
under a single dt, tracking the oracle; and the jitted step does not retrace
when only topology *values* change (the load-bearing no-recompile invariant)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from multipatch import atlas, coord_maps as cm
from multipatch.wave import WaveSystem, plane_wave_state
from multipatch.amr.state import BS, NG, NF, LEVELS, make_root_state
from multipatch.amr import evolve as E
from multipatch.amr.geometry import block_patch

W = BS + 2 * NG
KVEC = (0.7, -0.4, 0.5)
A_CUBE = 1.0

pytestmark = pytest.mark.skipif(LEVELS < 2, reason="needs LEVELS >= 2")


def _cube_patch():
    return atlas.Patch(
        name="cube", patch_type=cm.PATCH_AFFINE,
        patch_params=cm.make_affine_params(
            world_origin=(-A_CUBE,) * 3, world_scale=2.0 * A_CUBE),
        N=(BS, BS, BS), ng=NG, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))


def _halo_mask():
    h = np.ones((W, W, W), bool)
    h[NG:NG+BS, NG:NG+BS, NG:NG+BS] = False
    return jnp.asarray(h)


def _setup_two_level():
    """Root = whole cube; one level-1 child covering the (0,0,0) octant."""
    cube = _cube_patch()
    F0 = plane_wave_state(cube.X, cube.Y, cube.Z, 0.0, k=KVEC)
    state, topo = make_root_state(F0, nb_root=(1, 1, 1))

    # add the level-1 child at octant (0,0,0)
    child = block_patch(cube, 1, (0, 0, 0))
    slot = topo.find_empty_slot(1)
    topo.add_block(1, slot, (0, 0, 0), parent=(0, 0))
    Fc = plane_wave_state(child.X, child.Y, child.Z, 0.0, k=KVEC)   # (NF,W,W,W)
    blocks = list(state.blocks)
    # interiors-only storage: keep the BS³ interior
    blocks[1] = blocks[1].at[slot].set(Fc[:, NG:NG+BS, NG:NG+BS, NG:NG+BS])
    active = list(state.active)
    active[1] = active[1].at[slot].set(True)
    from multipatch.amr.state import AMRState
    state = AMRState(blocks=tuple(blocks), active=tuple(active))

    geom = E.build_geometry(cube, topo, topo.caps)
    return cube, child, state, topo, geom


def _combined_bc(cube, child):
    halo = _halo_mask()
    Xr, Yr, Zr = cube.X, cube.Y, cube.Z
    Xc, Yc, Zc = child.X, child.Y, child.Z

    def bc(blocks, t):
        blocks = list(blocks)
        vr = plane_wave_state(Xr, Yr, Zr, t, k=KVEC)
        blocks[0] = blocks[0].at[0].set(jnp.where(halo[None], vr, blocks[0][0]))
        vc = plane_wave_state(Xc, Yc, Zc, t, k=KVEC)
        blocks[1] = blocks[1].at[0].set(jnp.where(halo[None], vc, blocks[1][0]))
        return blocks
    return bc


def test_two_level_tracks_oracle():
    cube, child, state, topo, geom = _setup_two_level()
    topo_arr = topo.to_jax_arrays()
    ev = E.AMRCubeEvolution(cube, WaveSystem(), nb_root=(1, 1, 1), order=6,
                            ko_sigma=0.0, outer_bc=_combined_bc(cube, child))
    h_fine = 2.0 * A_CUBE / (BS - 1) / 2.0
    dt = 0.1 * h_fine
    out, t = ev.evolve(state, geom, topo_arr, dt, 30)

    ref_c = plane_wave_state(child.X, child.Y, child.Z, t, k=KVEC)
    ref_int = np.asarray(ref_c)[:, NG:NG+BS, NG:NG+BS, NG:NG+BS]
    got_c = np.asarray(out.blocks[1][0])       # interiors-only (NF,BS,BS,BS)
    err = np.max(np.abs(got_c - ref_int))
    assert np.isfinite(err) and err < 5e-2


def test_no_recompile_on_topology_value_change(monkeypatch):
    """Changing topology VALUES (same shapes) must not retrace the jitted step."""
    cube, child, state, topo, geom = _setup_two_level()
    topo_arr = topo.to_jax_arrays()
    ev = E.AMRCubeEvolution(cube, WaveSystem(), nb_root=(1, 1, 1), order=6,
                            ko_sigma=0.0, outer_bc=_combined_bc(cube, child))

    traces = {"n": 0}
    orig = E._block_rhs

    def counting(*a, **k):
        traces["n"] += 1            # Python-level → runs once per (re)trace
        return orig(*a, **k)
    monkeypatch.setattr(E, "_block_rhs", counting)

    jstep = ev.make_jit_step()
    dt = 0.01
    # step takes interiors directly (haloes are rebuilt inside the step)
    jstep(state.blocks, state.active, geom, topo_arr, 0.0, dt)
    after_first = traces["n"]
    assert after_first > 0

    # perturb topology VALUES only (same shapes): flip a neighbor_valid entry
    nv = list(topo_arr.neighbor_valid)
    nv[1] = nv[1].at[0, 0].set(~nv[1][0, 0])
    topo_arr2 = topo_arr._replace(neighbor_valid=tuple(nv))
    jstep(state.blocks, state.active, geom, topo_arr2, 0.0, dt)
    assert traces["n"] == after_first, "step retraced on a topology value change"
