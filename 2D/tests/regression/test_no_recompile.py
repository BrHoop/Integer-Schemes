"""
Regression guard: the AMR step functions must trace once and reuse the cached
compilation forever after.  This is the load-bearing promise of the whole AMR
architecture — if any traced function accidentally branches on a dynamic Python
value, this test catches it before it silently kills GPU performance.

Approach: wrap the un-jitted body in a counter, then re-jit and call N times.
The counter must end at 1 (one trace, all subsequent calls hit the cache).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import (
    BS, NG, NF, MAX_BLOCKS, LEVELS,
    AMRState, AMRTopology, AMRTopologyArrays,
)
from mcs2d.amr.kernels import (
    sync_ghosts_within_level_root_periodic,
    sync_ghosts_across_levels,
)
from mcs2d.amr.evolve import (
    make_root_step, make_two_level_step, make_subcycled_two_level_step,
    make_n_level_step, make_subcycled_n_level_step,
)
from mcs2d.amr.regrid import apply_flags, REFINE, COARSEN
from mcs2d.schemes.fused_rhs_fp import _make_kernel_fn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trace_counter(fn):
    """Return (counted_fn, counter_list).  counter_list[0] increments each
    time the underlying Python body actually runs (i.e., each trace)."""
    counter = [0]

    def counted(*args, **kwargs):
        counter[0] += 1
        return fn(*args, **kwargs)

    return counted, counter


def _make_inputs(nbx=2, nby=2):
    """Build a tiny AMR root state for sanity-running the kernels."""
    blocks = jnp.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG))
    # Put some noise in active slots so the kernel actually exercises arithmetic.
    rng = np.random.default_rng(0)
    blocks = blocks.at[:nbx*nby, :, NG:NG+BS, NG:NG+BS].set(
        jnp.asarray(rng.standard_normal((nbx*nby, NF, BS, BS)))
    )
    return blocks


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.regression
class TestNoRecompile:
    """All AMR-relevant jitted functions must compile exactly once across many
    calls with identically-shaped inputs."""

    N_CALLS = 10

    def test_sync_within_level_root_periodic(self):
        counted, counter = _trace_counter(
            sync_ghosts_within_level_root_periodic.__wrapped__
        )
        # nbx/nby are static_argnames in the original; for our re-jit we use
        # static_argnums so positional ints are treated as static, not traced.
        jitted = jax.jit(counted, static_argnums=(1, 2))
        blocks = _make_inputs()
        for _ in range(self.N_CALLS):
            out = jitted(blocks, 2, 2)
            out.block_until_ready()
        assert counter[0] == 1, (
            f"sync_ghosts_within_level_root_periodic re-traced {counter[0]} times "
            f"across {self.N_CALLS} calls"
        )

    def test_root_step_caches(self):
        """The big one: a full RK4 step must trace exactly once."""
        nbx, nby = 2, 2
        dx = dy = 10.0 / (nbx * BS)
        step = make_root_step(
            dx=dx, dy=dy, cs=1.0, L=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            dt=0.05 * dx, nbx=nbx, nby=nby,
        )
        # Peel back to the un-jitted body to count traces.
        # `make_root_step` returns a @jax.jit-decorated function whose
        # un-jitted body is exposed as `__wrapped__`.
        counted, counter = _trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)
        blocks = _make_inputs(nbx, nby)
        for _ in range(self.N_CALLS):
            blocks = rejitted(blocks)
            blocks.block_until_ready()
        assert counter[0] == 1, (
            f"make_root_step body re-traced {counter[0]} times across {self.N_CALLS} calls"
        )

    @pytest.mark.parametrize("restrict_at_end", [False, True])
    def test_two_level_step_caches(self, restrict_at_end):
        """Two-level step likewise must trace exactly once.  Run with and
        without restriction enabled — both branches must be cache-stable."""
        nbx, nby = 2, 2
        dx = dy = 10.0 / (nbx * BS)
        parent_slot = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        child_cx    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        child_cy    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        fine_active = jnp.zeros(MAX_BLOCKS, dtype=bool).at[0].set(True)

        step = make_two_level_step(
            dx_coarse=dx, dy_coarse=dy, dt=0.025 * dx,
            cs=1.0, L=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=nbx, nby_root=nby,
            restrict_at_end=restrict_at_end,
        )
        counted, counter = _trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)
        c = _make_inputs(nbx, nby)
        f = jnp.zeros_like(c)
        for _ in range(self.N_CALLS):
            c, f = rejitted(c, f, parent_slot, child_cx, child_cy, fine_active)
            c.block_until_ready()
        assert counter[0] == 1, (
            f"make_two_level_step(restrict_at_end={restrict_at_end}) re-traced "
            f"{counter[0]} times across {self.N_CALLS} calls"
        )

    def test_subcycled_two_level_step_caches(self):
        """The Berger-Oliger sub-cycled step must trace exactly once — despite
        its internal Python loop over substeps and RK stages (all unrolled at
        trace time)."""
        nbx, nby = 2, 2
        dx = dy = 10.0 / (nbx * BS)
        parent_slot = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        child_cx    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        child_cy    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        fine_active = jnp.zeros(MAX_BLOCKS, dtype=bool).at[0].set(True)

        step = make_subcycled_two_level_step(
            dx, dy, 0.05 * dx, cs=1.0, L=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=nbx, nby_root=nby,
        )
        counted, counter = _trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)
        c = _make_inputs(nbx, nby)
        f = jnp.zeros_like(c)
        for _ in range(self.N_CALLS):
            c, f = rejitted(c, f, parent_slot, child_cx, child_cy, fine_active)
            c.block_until_ready()
        assert counter[0] == 1, (
            f"make_subcycled_two_level_step re-traced {counter[0]} times "
            f"across {self.N_CALLS} calls"
        )

    def test_n_level_step_caches(self):
        """The full N-level step must trace exactly once across many calls."""
        nbx, nby = 2, 2
        dx = dy = 10.0 / (nbx * BS)
        step = make_n_level_step(
            dx_root=dx, dy_root=dy, dt=0.025 * dx,
            cs=1.0, L_coupling=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=nbx, nby_root=nby,
        )
        counted, counter = _trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)

        # Single root block active.
        blocks = jnp.zeros((LEVELS, MAX_BLOCKS, NF, BS+2*NG, BS+2*NG))
        active = jnp.zeros((LEVELS, MAX_BLOCKS), dtype=bool).at[0, :nbx*nby].set(True)
        state = AMRState(blocks=blocks, active=active)
        topo = AMRTopologyArrays(
            parent_slot=jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            child_cx   =jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            child_cy   =jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            neighbor_slot =jnp.zeros((LEVELS, MAX_BLOCKS, 4), dtype=jnp.int32),
            neighbor_valid=jnp.zeros((LEVELS, MAX_BLOCKS, 4), dtype=bool),
        )
        for _ in range(self.N_CALLS):
            state = rejitted(state, topo)
            state.blocks[0].block_until_ready()
        assert counter[0] == 1, (
            f"make_n_level_step re-traced {counter[0]} times across {self.N_CALLS} calls"
        )

    def test_subcycled_n_level_step_caches(self):
        """The recursive Berger-Oliger sub-cycled step must trace exactly once,
        despite the deeply-nested Python recursion over levels/substeps/stages
        (all unrolled at trace time)."""
        nbx, nby = 2, 2
        dx = dy = 10.0 / (nbx * BS)
        step = make_subcycled_n_level_step(
            dx_root=dx, dy_root=dy, dt_root=0.05 * dx,
            cs=1.0, L_coupling=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=nbx, nby_root=nby,
        )
        counted, counter = _trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)
        blocks = jnp.zeros((LEVELS, MAX_BLOCKS, NF, BS+2*NG, BS+2*NG))
        active = jnp.zeros((LEVELS, MAX_BLOCKS), dtype=bool).at[0, :nbx*nby].set(True)
        state = AMRState(blocks=blocks, active=active)
        topo = AMRTopologyArrays(
            parent_slot=jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            child_cx   =jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            child_cy   =jnp.zeros((LEVELS, MAX_BLOCKS), dtype=jnp.int32),
            neighbor_slot =jnp.zeros((LEVELS, MAX_BLOCKS, 4), dtype=jnp.int32),
            neighbor_valid=jnp.zeros((LEVELS, MAX_BLOCKS, 4), dtype=bool),
        )
        for _ in range(self.N_CALLS):
            state = rejitted(state, topo)
            state.blocks[0].block_until_ready()
        assert counter[0] == 1, (
            f"make_subcycled_n_level_step re-traced {counter[0]} times across "
            f"{self.N_CALLS} calls"
        )

    def test_n_level_step_survives_regrid(self):
        """The headline Phase 2 promise: regrid events (refine + coarsen) must
        not trigger re-tracing.  Topology arrays change between calls but
        shapes are fixed, so the cache should be hit every time."""
        nbx, nby = 2, 2
        dx = dy = 10.0 / (nbx * BS)
        step = make_n_level_step(
            dx_root=dx, dy_root=dy, dt=0.025 * dx,
            cs=1.0, L_coupling=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=nbx, nby_root=nby,
        )
        counted, counter = _trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)

        # Build initial state with random data.
        rng = np.random.default_rng(0)
        blocks_np = np.zeros((LEVELS, MAX_BLOCKS, NF, BS+2*NG, BS+2*NG))
        blocks_np[0, :nbx*nby, :, NG:NG+BS, NG:NG+BS] = rng.standard_normal(
            (nbx*nby, NF, BS, BS)
        )
        active_np = np.zeros((LEVELS, MAX_BLOCKS), dtype=bool)
        active_np[0, :nbx*nby] = True
        state = AMRState(blocks=jnp.asarray(blocks_np),
                         active=jnp.asarray(active_np))
        topo = AMRTopology()
        for slot in range(nbx*nby):
            bi = slot // nby; bj = slot % nby
            topo.add_block(0, slot, (bi * BS, bj * BS))

        # Initial step (trace 1).
        for _ in range(3):
            state = rejitted(state, topo.to_jax_arrays())
            state.blocks[0].block_until_ready()
        assert counter[0] == 1

        # Simulated regrid event 1: refine root block 0.
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        for _ in range(3):
            state = rejitted(state, topo.to_jax_arrays())
            state.blocks[0].block_until_ready()
        assert counter[0] == 1, (
            f"retraced after refine: {counter[0]} traces"
        )

        # Simulated regrid event 2: refine one of the level-1 children.
        active_l1 = np.asarray(state.active[1])
        l1_slot = int(np.flatnonzero(active_l1)[0])
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[1, l1_slot] = REFINE
        state, topo = apply_flags(state, topo, flags)
        for _ in range(3):
            state = rejitted(state, topo.to_jax_arrays())
            state.blocks[0].block_until_ready()
        assert counter[0] == 1, (
            f"retraced after depth-2 refine: {counter[0]} traces"
        )

        # Simulated regrid event 3: coarsen one of the level-2 grandchildren.
        active_l2 = np.asarray(state.active[2])
        l2_slot = int(np.flatnonzero(active_l2)[0])
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[2, l2_slot] = COARSEN
        state, topo = apply_flags(state, topo, flags)
        for _ in range(3):
            state = rejitted(state, topo.to_jax_arrays())
            state.blocks[0].block_until_ready()
        assert counter[0] == 1, (
            f"retraced after coarsen: {counter[0]} traces"
        )

    def test_two_level_step_survives_regrid(self):
        """Critical Phase 2 guarantee: changing topology arrays between calls
        must NOT trigger a recompile (since shapes never change)."""
        nbx, nby = 2, 2
        dx = dy = 10.0 / (nbx * BS)
        step = make_two_level_step(
            dx_coarse=dx, dy_coarse=dy, dt=0.025 * dx,
            cs=1.0, L=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=nbx, nby_root=nby,
            restrict_at_end=True,
        )
        counted, counter = _trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)
        c = _make_inputs(nbx, nby)
        f = jnp.zeros_like(c)

        # Simulate 5 "regrid events" — each one switches which fine slot is
        # active, mimics what a regrid driver would do.  Any retracing across
        # these calls would break Phase 2's headline promise.
        for event in range(5):
            ps = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
            ccx = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
            ccy = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
            fa  = jnp.zeros(MAX_BLOCKS, dtype=bool).at[event].set(True)
            ccx = ccx.at[event].set(event % 2)
            ccy = ccy.at[event].set((event // 2) % 2)
            for _ in range(3):
                c, f = rejitted(c, f, ps, ccx, ccy, fa)
                c.block_until_ready()

        assert counter[0] == 1, (
            f"two-level step retraced across regrid events: {counter[0]} traces "
            f"(expected 1)"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
