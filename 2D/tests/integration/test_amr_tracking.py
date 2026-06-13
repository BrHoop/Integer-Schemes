"""
Phase 2 exit-criterion test — refinement TRACKS a moving feature.

A localized Gaussian pulse is evolved on a multi-block root grid with periodic
BC, regridding every K steps.  The refined region must:
  1. appear where the field gradient is high (near the pulse),
  2. follow the pulse as it propagates (the refined block set changes), and
  3. never trigger a JAX recompile across the regrid events.

This exercises the full Phase 2 stack end-to-end: indicator → hysteresis →
nesting/buffer → allocate/free → N-level step → repeat.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import (
    BS, NG, NF, MAX_BLOCKS, LEVELS,
    AMRState, AMRTopology,
)
from mcs2d.amr.evolve import make_n_level_step, amr_state_from_global
from mcs2d.amr.regrid import evolve_with_regrid, regrid, apply_flags, REFINE
from mcs2d.main import MaxwellChernSimons2D, InitialData, load_parameters


CFL      = 0.05
LAMBDA   = 0.4         # CFJ-stable
CS       = 1.0
K1 = K2  = 1.0
KO_SIGMA = 0.05
EZ_IDX   = 2


def _gaussian_state(params_file, nbx, nby, x0, y0, sigma=0.6, amp=1.0):
    """Build a root AMR state with a Gaussian pulse centered at (x0, y0)."""
    nx, ny = nbx * BS, nby * BS
    params = load_parameters(params_file)
    params.update({
        'scheme': 'fused_floating_point', 'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'gaussian', 'bc_type': 'periodic', 'sponge_strength': 0.0,
        'enable_cs': CS, 'Lambda': LAMBDA, 'ko_sigma': KO_SIGMA,
        'id_x0': x0, 'id_y0': y0, 'id_sigma': sigma, 'id_amp': amp,
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, params['Lambda'], params)
    interior = np.asarray(
        InitialData(sim, params).generate().data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny]
    )
    blocks2d = amr_state_from_global(jnp.asarray(interior), nbx, nby)
    blocks = jnp.zeros(
        (LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64
    ).at[0].set(blocks2d)
    active = jnp.zeros((LEVELS, MAX_BLOCKS), dtype=bool).at[0, :nbx*nby].set(True)
    state = AMRState(blocks=blocks, active=active)
    topo = AMRTopology()
    for slot in range(nbx*nby):
        topo.add_block(0, slot, ((slot // nby) * BS, (slot % nby) * BS))
    return state, topo, sim, params


def _refined_block_centroid(topo):
    """Mean physical-cell position (in root-cell units) of level-1 blocks."""
    l1 = [topo.bbox_ijk[(1, s)] for s in range(MAX_BLOCKS) if topo.active[1, s]]
    if not l1:
        return None
    # bbox is in level-1 cell units; convert to level-0 (÷2) + half a block.
    cx = np.mean([(b[0] / 2) + BS / 4 for b in l1])
    cy = np.mean([(b[1] / 2) + BS / 4 for b in l1])
    return cx, cy


@pytest.mark.amr
class TestRefinementTracksPulse:

    def test_refinement_appears_near_pulse(self, params_file):
        """A single regrid should refine the block(s) containing the pulse and
        leave far-away flat blocks coarse."""
        nbx, nby = 4, 4
        # Pulse in the lower-left quadrant.
        x0, y0 = -2.5, -2.5
        state, topo, sim, params = _gaussian_state(params_file, nbx, nby, x0, y0)

        dx_per_level = [sim.dx / (2 ** L) for L in range(LEVELS)]
        # One regrid (hysteresis_K=1 so it fires immediately).  Cap depth at
        # level 1 so the resolution-independent |∇Ez| indicator doesn't cascade
        # to max depth and exhaust the slot budget.
        state, topo = regrid(
            state, topo, dx_per_level, field_idx=EZ_IDX,
            refine_threshold=0.02, coarsen_threshold=0.001,
            hysteresis_K=1, n_buffer=1, max_level=1,
        )

        refined_slots = [s for s in range(MAX_BLOCKS) if topo.active[1, s]]
        assert refined_slots, "no refinement created near the pulse"

        # The pulse center in root-cell coords.
        Lx = params['xmax'] - params['xmin']
        cx_cells = (x0 - params['xmin']) / sim.dx
        cy_cells = (y0 - params['ymin']) / sim.dy
        # Centroid of refined level-1 blocks should be near the pulse.
        centroid = _refined_block_centroid(topo)
        assert centroid is not None
        dist = np.hypot(centroid[0] - cx_cells, centroid[1] - cy_cells)
        # Within ~2 root blocks of the pulse center.
        assert dist < 2 * BS, (
            f"refined centroid {centroid} far from pulse ({cx_cells:.1f},{cy_cells:.1f}), "
            f"dist={dist:.1f} cells"
        )

        # A block in the OPPOSITE corner should not be refined.
        far_slot = (nbx - 1) * nby + (nby - 1)   # top-right root block
        far_children = topo.children.get((0, far_slot), [])
        assert not any(topo.active[cl, cs] for (cl, cs) in far_children), \
            "far-from-pulse block was refined"

    @pytest.mark.slow
    def test_refinement_adapts_over_evolution(self, params_file):
        """End-to-end regrid-in-the-loop: as the field evolves, the set of
        refined blocks must ADAPT (change over time), with no JAX recompile and
        the slot budget respected throughout.

        A radiating Gaussian on a periodic domain develops structure across
        successive blocks, so the refined region grows/shifts as the solution
        evolves (verified: 16 → 48 → 64 blocks over the run).  Marked slow: a
        block is BS=32 cells, so at CFL=0.05 the refinement pattern only changes
        on a ~1000-step timescale."""
        nbx, nby = 4, 4
        state, topo, sim, params = _gaussian_state(
            params_file, nbx, nby, x0=-2.5, y0=-2.5, sigma=0.5,
        )

        dx_per_level = [sim.dx / (2 ** L) for L in range(LEVELS)]
        # max_level=1 → finest active level is 1, so use its CFL (dx/2).
        dt = CFL * sim.dx / 2
        step = make_n_level_step(
            dx_root=sim.dx, dy_root=sim.dy, dt=dt,
            cs=CS, L_coupling=LAMBDA, K1=K1, K2=K2, ko_sigma=KO_SIGMA,
            nbx_root=nbx, nby_root=nby,
        )

        history = []
        def record(steps, st, tp):
            refined = frozenset(s for s in range(MAX_BLOCKS) if tp.active[1, s])
            n_total = int(np.asarray(tp.active).sum())
            history.append((steps, refined, n_total))

        # Count recompiles via a trace-counting wrapper around the step body.
        trace_count = [0]
        inner = step.__wrapped__
        def counted(*a, **k):
            trace_count[0] += 1
            return inner(*a, **k)
        step_counted = jax.jit(counted)

        state, topo = evolve_with_regrid(
            state, topo, step_counted, dx_per_level,
            n_steps=1300, regrid_every=100,
            refine_threshold=0.5, coarsen_threshold=0.2,
            hysteresis_K=1, n_buffer=1, max_level=1, field_idx=EZ_IDX,
            on_regrid=record,
        )

        # 1. The step compiled exactly once despite every regrid event.
        assert trace_count[0] == 1, (
            f"recompiled across regrids: {trace_count[0]} traces"
        )
        # 2. Refinement happened.
        assert history, "no regrid events recorded"
        all_refined = set().union(*[r for _, r, _ in history])
        assert all_refined, "no refinement created during evolution"
        # 3. The refined set ADAPTED over the run.
        distinct_sets = {r for _, r, _ in history}
        assert len(distinct_sets) > 1, (
            "refined block set never changed — refinement isn't adapting"
        )
        # 4. The slot budget was respected the whole time (level-1 cap = MAX_BLOCKS).
        assert all(n <= MAX_BLOCKS * LEVELS for _, _, n in history)
        max_l1 = max(len(r) for _, r, _ in history)
        assert max_l1 <= MAX_BLOCKS, f"level-1 slots over budget: {max_l1}"


@pytest.mark.amr
class TestMovingFeatureDynamics:
    """Phase 5.3 — validate the regrid pipeline under a MOVING feature.

    In our block-structured (grid-aligned) AMR the 'copy-overlap' regrid transfer
    reduces to RETENTION: a refined block at a fixed grid position is kept as the
    feature moves (only genuinely-new positions are prolongated from coarse, which
    is correct since no fine data existed there).  These tests pin that retention
    and confirm proper nesting holds throughout a moving evolution."""

    def test_refine_retains_existing_children_data(self, params_file):
        """REFINE-flagging a parent whose children already exist must NOT
        re-prolongate them — the existing (evolved) fine data is retained."""
        nbx, nby = 2, 2
        state, topo, sim, params = _gaussian_state(
            params_file, nbx, nby, x0=0.0, y0=0.0, sigma=0.6)
        flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        assert topo.n_active(1) == 4, "expected 4 children after first refine"

        # Perturb the children's interiors so "retained" differs from a fresh
        # prolongation from the (unperturbed) parent.
        l1 = np.array(state.blocks[1], copy=True)
        for s in range(topo.caps[1]):
            if topo.active[1, s]:
                l1[s, :, NG:NG+BS, NG:NG+BS] += 3.14159
        new_blocks = tuple(jnp.asarray(l1) if L == 1 else state.blocks[L]
                           for L in range(LEVELS))
        state = AMRState(blocks=new_blocks, active=state.active)
        before = np.array(state.blocks[1], copy=True)

        # Re-REFINE the same parent: all 4 children already exist → no-op on data.
        state2, topo2 = apply_flags(state, topo, flags)
        after = np.asarray(state2.blocks[1])
        for s in range(topo.caps[1]):
            if topo.active[1, s]:
                assert np.array_equal(after[s], before[s]), \
                    f"child {s} was re-prolongated (data changed) — retention broken"

    def test_proper_nesting_maintained_under_motion(self, params_file):
        """Proper nesting (5.2 invariant) must hold after EVERY regrid as a
        radiating pulse moves the refined region across the domain."""
        nbx, nby = 4, 4
        state, topo, sim, params = _gaussian_state(
            params_file, nbx, nby, x0=-2.5, y0=-2.5, sigma=0.5)
        dx_per_level = [sim.dx / (2 ** L) for L in range(LEVELS)]
        dt = CFL * sim.dx / 2
        step = make_n_level_step(
            dx_root=sim.dx, dy_root=sim.dy, dt=dt,
            cs=CS, L_coupling=LAMBDA, K1=K1, K2=K2, ko_sigma=KO_SIGMA,
            nbx_root=nbx, nby_root=nby)

        violations_seen = []
        def check(steps, st, tp):
            v = tp.check_proper_nesting()
            if v:
                violations_seen.append((steps, v))

        state, topo = evolve_with_regrid(
            state, topo, step, dx_per_level,
            n_steps=600, regrid_every=100,
            refine_threshold=0.5, coarsen_threshold=0.2,
            hysteresis_K=1, n_buffer=1, max_level=1, field_idx=EZ_IDX,
            on_regrid=check)
        assert not violations_seen, f"proper-nesting violated during motion: {violations_seen}"
        # And the final hierarchy is still properly nested + finite.
        assert topo.check_proper_nesting() == []
        assert np.isfinite(np.asarray(state.blocks[1])).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
