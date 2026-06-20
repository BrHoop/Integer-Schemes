"""Cube-first AMR validation: per-block RHS accuracy vs the analytic wave
time-derivative, and single-root evolution vs the plane-wave oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from multipatch import atlas, coord_maps as cm
from multipatch.wave import WaveSystem, plane_wave_state
from multipatch.amr.state import BS, NG, NF, LEVELS, MAX_BLOCKS_PER_LEVEL, make_root_state
from multipatch.amr import evolve as E

W = BS + 2 * NG
KVEC = (0.7, -0.4, 0.5)
A_CUBE = 1.0


def _cube_patch():
    return atlas.Patch(
        name="cube", patch_type=cm.PATCH_AFFINE,
        patch_params=cm.make_affine_params(
            world_origin=(-A_CUBE,) * 3, world_scale=2.0 * A_CUBE),
        N=(BS, BS, BS), ng=NG, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))


def _wave_dt_analytic(X, Y, Z, t, k=KVEC, amp=1.0):
    """Continuous ∂t of the plane-wave state (the RHS the scheme approximates)."""
    kx, ky, kz = k
    omega = (kx*kx + ky*ky + kz*kz) ** 0.5
    phase = kx*X + ky*Y + kz*Z - omega*t
    s, c = jnp.sin(phase), jnp.cos(phase)
    dt_phi = -amp * omega * c                    # = pi
    dt_pi = -amp * omega**2 * s                  # = ∇²phi
    dt_cx = amp * kx * omega * s                 # = ∂x pi
    dt_cy = amp * ky * omega * s
    dt_cz = amp * kz * omega * s
    return jnp.stack([dt_phi, dt_pi, dt_cx, dt_cy, dt_cz])


def test_block_rhs_matches_analytic():
    """The semi-discrete block RHS on exact data matches the continuous ∂t to FD
    truncation order (r_phi = pi is exact; the derivative channels are O(h^6))."""
    assert NF == 5, "set MP_AMR_NF=5 (wave) for this suite"
    cube = _cube_patch()
    F = plane_wave_state(cube.X, cube.Y, cube.Z, 0.0, k=KVEC)   # (5,W,W,W) incl. ghosts
    dxi = E.level_spacing(cube, 0)
    r = E._block_rhs(F, cube.jinv, cube.d2coef, dxi, WaveSystem(), 6, 0.0, 0.0)
    # _block_rhs now returns the (NF,BS,BS,BS) interior directly
    ref = _wave_dt_analytic(cube.X, cube.Y, cube.Z, 0.0)
    ref_int = np.asarray(ref)[:, NG:NG+BS, NG:NG+BS, NG:NG+BS]
    err = np.max(np.abs(np.asarray(r) - ref_int))
    # r_phi channel is exact:
    e_phi = np.max(np.abs(np.asarray(r)[0] - ref_int[0]))
    assert e_phi < 1e-12
    # derivative channels: FD truncation on a coarse cube (h ~ 0.29) — generous bound
    assert err < 5e-2


def _make_cube_bc(cube):
    halo = np.ones((W, W, W), bool)
    halo[NG:NG+BS, NG:NG+BS, NG:NG+BS] = False
    halo = jnp.asarray(halo)
    Xg, Yg, Zg = cube.X, cube.Y, cube.Z

    def bc(blocks, t):
        vals = plane_wave_state(Xg, Yg, Zg, t, k=KVEC)         # (5,W,W,W)
        F0 = jnp.where(halo[None], vals, blocks[0][0])
        blocks = list(blocks)
        blocks[0] = blocks[0].at[0].set(F0)
        return blocks
    return bc


def test_single_root_evolution_tracks_oracle():
    """One root block (whole cube), exact Dirichlet halo, RK4 — stays close to the
    plane-wave oracle and does not blow up."""
    cube = _cube_patch()
    F0 = plane_wave_state(cube.X, cube.Y, cube.Z, 0.0, k=KVEC)
    state, topo = make_root_state(F0, nb_root=(1, 1, 1))
    topo_arr = topo.to_jax_arrays()
    geom = E.build_geometry(cube, topo, topo.caps)

    ev = E.AMRCubeEvolution(cube, WaveSystem(), nb_root=(1, 1, 1), order=6,
                            ko_sigma=0.0, outer_bc=_make_cube_bc(cube))
    h_world = 2.0 * A_CUBE / (BS - 1)
    dt = 0.1 * h_world
    nsteps = 40
    out, t = ev.evolve(state, geom, topo_arr, dt, nsteps)

    ref = plane_wave_state(cube.X, cube.Y, cube.Z, t, k=KVEC)
    ref_int = np.asarray(ref)[:, NG:NG+BS, NG:NG+BS, NG:NG+BS]
    got = np.asarray(out.blocks[0][0])         # interiors-only (NF,BS,BS,BS)
    err = np.max(np.abs(got - ref_int))
    assert np.isfinite(err)
    assert err < 5e-2
