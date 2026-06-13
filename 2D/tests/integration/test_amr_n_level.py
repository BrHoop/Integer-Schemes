"""
N-level AMR evolution tests.

Covers:
  * `make_n_level_step` reduces correctly to root-only behavior when no
    fine levels are active.
  * Topology snapshot round-trip: `AMRTopology.to_jax_arrays()` produces
    arrays that drive cross-level sync to the right results.
  * Recursive refinement: regrid → make_n_level_step → state evolves
    stably with a depth-3 hierarchy.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import (
    BS, NG, NF, MAX_BLOCKS, LEVELS,
    AMRState, AMRTopology, AMRTopologyArrays,
)
from mcs2d.amr.kernels import sync_ghosts_within_level_root_periodic, prolongate
from mcs2d.amr.evolve import (
    make_root_step, make_n_level_step,
    amr_state_from_global, amr_state_to_global,
)
from mcs2d.amr.regrid import apply_flags, REFINE
from mcs2d.main import MaxwellChernSimons2D, InitialData, load_parameters


# Physical constants matching params.toml
CFL      = 0.05
LAMBDA   = 0.4       # CFJ-stable regime (m_cs=0.8 < k≈0.89); see params.toml
CS       = 1.0
K1       = 1.0
K2       = 1.0
KO_SIGMA = 0.05
EZ_IDX   = 2


def _birefringent_state(nx, ny, params_file):
    """Standard birefringent IC via main.InitialData."""
    params = load_parameters(params_file)
    params.update({
        'scheme': 'floating_point', 'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'birefringent', 'bc_type': 'periodic',
        'sponge_strength': 0.0,
        'Lambda': LAMBDA,   # force consistency: sim/oracle/AMR-step all use LAMBDA
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, params['Lambda'], params)
    state = InitialData(sim, params).generate()
    return sim, state, params


class BirefringentOracle:
    def __init__(self, params):
        Lx = params["xmax"] - params["xmin"]
        Ly = params["ymax"] - params["ymin"]
        self.kx = 2 * np.pi / Lx
        self.ky = 2 * np.pi / Ly
        k_mag = np.sqrt(self.kx**2 + self.ky**2)
        m_cs = params.get("id_m_cs", params.get("Lambda", 1.0) * 2.0)
        self.omega = np.sqrt(k_mag**2 + m_cs * k_mag)
        self.E0 = params.get("id_amp", 1.0)

    def Ez(self, X, Y, t):
        return self.E0 * np.sin(self.kx * X + self.ky * Y - self.omega * t)


# ── N-level step degeneracy: should match root-only when no fine levels are active ──

class TestNLevelReducesToRootOnly:
    """With only level 0 populated, `make_n_level_step` must produce numerically
    identical results to `make_root_step` (since the fine-level sync is a no-op
    when active[L>0] is all False)."""

    N_STEPS = 50
    RTOL = 1e-12

    def test_n_level_matches_root_only(self, params_file):
        nbx, nby = 2, 2
        nx, ny = nbx * BS, nby * BS
        sim, state_main, params = _birefringent_state(nx, ny, params_file)
        interior = np.asarray(state_main.data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny])

        dx, dy = sim.dx, sim.dy
        dt = sim.dt

        # Reference: root-only step.
        root_blocks = amr_state_from_global(jnp.asarray(interior), nbx, nby)
        step_root = make_root_step(
            dx=dx, dy=dy, cs=CS, L=LAMBDA, K1=K1, K2=K2,
            ko_sigma=KO_SIGMA, dt=dt, nbx=nbx, nby=nby,
        )
        def body_r(carry, _):
            return step_root(carry), None
        root_final = jax.jit(
            lambda s: jax.lax.scan(body_r, s, None, length=self.N_STEPS)[0]
        )(root_blocks)

        # N-level step with everything but level 0 inactive.
        active = jnp.zeros((LEVELS, MAX_BLOCKS), dtype=bool)
        active = active.at[0, :nbx*nby].set(True)
        # Build full-state blocks (LEVELS, MAX_BLOCKS, ...)
        nlevel_blocks = jnp.zeros(
            (LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64
        ).at[0].set(root_blocks)
        nlevel_state = AMRState(blocks=nlevel_blocks, active=active)
        topo_arrays = AMRTopologyArrays(
            parent_slot=jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            child_cx   =jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            child_cy   =jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            neighbor_slot =jnp.zeros((LEVELS, MAX_BLOCKS, 4), dtype=jnp.int32),
            neighbor_valid=jnp.zeros((LEVELS, MAX_BLOCKS, 4), dtype=bool),
        )

        step_n = make_n_level_step(
            dx_root=dx, dy_root=dy, dt=dt,
            cs=CS, L_coupling=LAMBDA, K1=K1, K2=K2, ko_sigma=KO_SIGMA,
            nbx_root=nbx, nby_root=nby,
        )
        def body_n(carry, _):
            return step_n(carry, topo_arrays), None
        nlevel_final = jax.jit(
            lambda s: jax.lax.scan(body_n, s, None, length=self.N_STEPS)[0]
        )(nlevel_state)

        # Compare level-0 blocks.
        root_root = np.asarray(root_final)
        nlevel_root = np.asarray(nlevel_final.blocks[0])
        for slot in range(nbx * nby):
            err = np.max(np.abs(root_root[slot] - nlevel_root[slot]))
            scale = max(np.max(np.abs(root_root[slot])), 1.0)
            assert err / scale < self.RTOL, (
                f"slot {slot}: N-level diverged from root-only "
                f"(rel err {err / scale:.2e})"
            )


# ── Topology snapshot round-trip ─────────────────────────────────────────────

class TestTopologyArraysSnapshot:
    """After REFINE on a single root block, to_jax_arrays must capture parent
    links and (cx, cy) corners correctly."""

    def test_post_refine_snapshot(self):
        # Build a single-block root state.
        rng = np.random.default_rng(31)
        blocks = np.zeros((LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG))
        blocks[0, 0] = rng.standard_normal((NF, BS + 2*NG, BS + 2*NG))
        active = np.zeros((LEVELS, MAX_BLOCKS), dtype=bool)
        active[0, 0] = True

        state = AMRState(blocks=jnp.asarray(blocks), active=jnp.asarray(active))
        topo = AMRTopology()
        topo.add_block(0, 0, (0, 0))

        # Refine the root block.
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)

        # Snapshot.
        from mcs2d.amr.state import mb
        ta = topo.to_jax_arrays()
        # Ragged per-level: parent_slot is a tuple of LEVELS arrays.
        assert len(ta.parent_slot) == LEVELS
        for L in range(LEVELS):
            assert ta.parent_slot[L].shape == (mb(L),)
        # All 4 children of root slot 0 should have parent_slot[1, *] == 0.
        active_l1 = np.asarray(state.active[1])
        for s in range(MAX_BLOCKS):
            if not active_l1[s]:
                continue
            assert int(ta.parent_slot[1][s]) == 0, \
                f"slot {s}: parent_slot[1] = {int(ta.parent_slot[1][s])}, expected 0"
            # (cx, cy) must be one of the 4 quadrants.
            cx = int(ta.child_cx[1][s]); cy = int(ta.child_cy[1][s])
            assert (cx, cy) in {(0, 0), (0, 1), (1, 0), (1, 1)}, \
                f"slot {s}: invalid corner ({cx}, {cy})"

        # All 4 corners should be present exactly once across active children.
        seen = set()
        for s in range(MAX_BLOCKS):
            if not active_l1[s]:
                continue
            seen.add((int(ta.child_cx[1][s]), int(ta.child_cy[1][s])))
        expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
        missing = expected - seen
        assert seen == expected, f"missing corners: {missing}"


# ── Recursive refinement: build depth-3 hierarchy and evolve ─────────────────

class TestRecursiveRefineEvolution:
    """Refine root → level 1 → level 2 → level 3, evolve a few steps,
    confirm: (a) no crash, (b) finite/no-NaN, (c) reasonable amplitude."""

    def test_depth_3_evolves_stably(self, params_file):
        # Skip if LEVELS < 4 (we need 4 levels: 0..3).
        if LEVELS < 4:
            pytest.skip(f"LEVELS={LEVELS} < 4; can't build depth-3 hierarchy")

        nbx, nby = 1, 1
        nx, ny = nbx * BS, nby * BS
        sim, state_main, params = _birefringent_state(nx, ny, params_file)
        interior = np.asarray(state_main.data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny])
        dx, dy = sim.dx, sim.dy

        # Build root state (1×1 = single block).
        root_blocks_2d = amr_state_from_global(jnp.asarray(interior), nbx, nby)
        blocks = jnp.zeros(
            (LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64
        ).at[0].set(root_blocks_2d)
        active = jnp.zeros((LEVELS, MAX_BLOCKS), dtype=bool).at[0, 0].set(True)
        state = AMRState(blocks=blocks, active=active)
        topo = AMRTopology()
        topo.add_block(0, 0, (0, 0))

        # Refine root → 4 children at level 1.
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        # Refine first level-1 child → 4 children at level 2.
        l1_child_slot = next(s for s in range(MAX_BLOCKS)
                             if np.asarray(state.active[1])[s])
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[1, l1_child_slot] = REFINE
        state, topo = apply_flags(state, topo, flags)
        # Refine first level-2 child → 4 children at level 3.
        l2_child_slot = next(s for s in range(MAX_BLOCKS)
                             if np.asarray(state.active[2])[s])
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[2, l2_child_slot] = REFINE
        state, topo = apply_flags(state, topo, flags)

        # Now: 1 root + 4 lvl1 + 4 lvl2 + 4 lvl3 = 13 active blocks
        assert int(np.asarray(state.active[0]).sum()) == 1
        assert int(np.asarray(state.active[1]).sum()) == 4
        assert int(np.asarray(state.active[2]).sum()) == 4
        assert int(np.asarray(state.active[3]).sum()) == 4

        # Build N-level step.  dt = CFL * dx_finest = CFL * dx_root / 8.
        dt = CFL * dx / 8.0
        step = make_n_level_step(
            dx_root=dx, dy_root=dy, dt=dt,
            cs=CS, L_coupling=LAMBDA, K1=K1, K2=K2, ko_sigma=KO_SIGMA,
            nbx_root=nbx, nby_root=nby,
        )

        topo_arrays = topo.to_jax_arrays()
        # Evolve 20 steps.
        def body(carry, _):
            return step(carry, topo_arrays), None
        state_final = jax.jit(
            lambda s: jax.lax.scan(body, s, None, length=20)[0]
        )(state)

        # Sanity: no NaN/Inf anywhere active.
        for L in range(4):
            active_arr = np.asarray(state_final.active[L])
            blocks_arr = np.asarray(state_final.blocks[L])
            for s in range(MAX_BLOCKS):
                if not active_arr[s]:
                    continue
                bs = blocks_arr[s]
                assert np.all(np.isfinite(bs)), f"NaN/Inf at level {L}, slot {s}"
                # Amplitude should be of order 1 (birefringent E0 = 0.8).
                amax = np.max(np.abs(bs))
                assert amax < 10.0, (
                    f"level {L} slot {s}: max |field| = {amax:.2e} — blew up?"
                )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
