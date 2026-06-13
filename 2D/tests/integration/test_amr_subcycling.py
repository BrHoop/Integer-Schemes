"""
Berger-Oliger sub-cycling tests (Phase 3).

`make_subcycled_two_level_step` advances a 2-level state by one coarse step
dt_c, with the fine level taking two dt_c/2 substeps and time-interpolated
coarse-fine boundaries, then restricting fine → coarse.

Correctness is checked against the SHARED-dt 2-level step run at the fine dt
(`make_two_level_step` at dt_c/2, twice per coarse step): the two reach the
same physical time, and the sub-cycled fine block must match the shared-fine-dt
fine block to the time-interpolation error O(dt_c²).  Using the shared-dt run
as the reference avoids any analytic-oracle position-convention ambiguity.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import (
    BS, NG, NF, MAX_BLOCKS, LEVELS, AMRState, AMRTopology, make_root_state,
)
from mcs2d.amr.kernels import sync_ghosts_within_level_root_periodic, prolongate
from mcs2d.amr.evolve import (
    make_subcycled_two_level_step, make_subcycled_n_level_step,
    make_subcycled_n_level_step_unrolled,
    make_two_level_step, amr_state_from_global,
)
from mcs2d.amr.regrid import apply_flags, REFINE
from mcs2d.main import MaxwellChernSimons2D, InitialData, load_parameters


CFL    = 0.05
LAMBDA = 0.4        # CFJ-stable
CS     = 1.0
K1 = K2 = 1.0
KO_SIGMA = 0.05
EZ = 2


def _two_level_birefringent(params_file, nbx=2, nby=2):
    """Coarse root state + one fine child (slot 0, corner (0,0)) from the
    birefringent IC.  Returns (coarse_blocks, fine_blocks, topo arrays, sim, params)."""
    nx, ny = nbx * BS, nby * BS
    params = load_parameters(params_file)
    params.update({
        'scheme': 'floating_point', 'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'birefringent', 'bc_type': 'periodic', 'sponge_strength': 0.0,
        'enable_cs': CS, 'Lambda': LAMBDA, 'ko_sigma': KO_SIGMA,
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, LAMBDA, params)
    interior = np.asarray(
        InitialData(sim, params).generate().data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny]
    )
    cb = amr_state_from_global(jnp.asarray(interior), nbx, nby)
    csync = sync_ghosts_within_level_root_periodic(cb, nbx, nby)
    child0 = prolongate(csync[0], (0, 0))
    fb = jnp.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG)).at[0].set(child0)
    ps  = jnp.zeros(MAX_BLOCKS, jnp.int32)
    ccx = jnp.zeros(MAX_BLOCKS, jnp.int32)
    ccy = jnp.zeros(MAX_BLOCKS, jnp.int32)
    fa  = jnp.zeros(MAX_BLOCKS, bool).at[0].set(True)
    return cb, fb, (ps, ccx, ccy, fa), sim, params


@pytest.mark.amr
class TestSubcycledTwoLevel:

    def test_tracks_shared_fine_dt(self, params_file):
        """Sanity bound: the sub-cycled fine block stays close to a shared-dt run
        at the fine dt (2N steps → same physical time).

        This is a loose tracking check, NOT a bit-match: the sub-cycled step now
        restricts at 6th order while `make_two_level_step` restricts by 2×2
        averaging, so the residual (~1e-4) is dominated by that restriction-order
        difference fed back through the fine halo over many steps — both are
        valid schemes.  Strict sub-cycling correctness is carried by
        `TestSubcycledNLevel::test_reduces_to_two_level` (machine precision)."""
        nbx, nby = 2, 2
        cb, fb, (ps, ccx, ccy, fa), sim, params = _two_level_birefringent(params_file, nbx, nby)
        dt_c = CFL * sim.dx
        dt_f = dt_c / 2.0

        sub = make_subcycled_two_level_step(
            sim.dx, sim.dy, dt_c, CS, LAMBDA, K1, K2, KO_SIGMA, nbx, nby)
        shared = make_two_level_step(
            sim.dx, sim.dy, dt_f, cs=CS, L=LAMBDA, K1=K1, K2=K2, ko_sigma=KO_SIGMA,
            nbx_root=nbx, nby_root=nby, restrict_at_end=True)

        N = 50
        c_s, f_s = cb, fb
        for _ in range(N):
            c_s, f_s = sub(c_s, f_s, ps, ccx, ccy, fa)
        c_r, f_r = cb, fb
        for _ in range(2 * N):     # fine dt, twice as many steps → same physical time
            c_r, f_r = shared(c_r, f_r, ps, ccx, ccy, fa)

        fi_sub = np.asarray(f_s[0, :, NG:NG+BS, NG:NG+BS])
        fi_ref = np.asarray(f_r[0, :, NG:NG+BS, NG:NG+BS])
        l2 = float(np.sqrt(np.mean((fi_sub - fi_ref) ** 2)))
        # Observed ~7e-5 (restriction-order difference + time interp); far below
        # any gross-error scale (the wave amplitude is ~0.8).
        assert l2 < 5e-4, (
            f"sub-cycled fine block drifted from shared-fine-dt by L2={l2:.2e} "
            f"(> 5e-4) — unexpectedly large"
        )

    def test_stable_and_bounded(self, params_file):
        """Sub-cycling must stay finite and amplitude-bounded over many steps."""
        nbx, nby = 2, 2
        cb, fb, (ps, ccx, ccy, fa), sim, params = _two_level_birefringent(params_file, nbx, nby)
        dt_c = CFL * sim.dx
        sub = make_subcycled_two_level_step(
            sim.dx, sim.dy, dt_c, CS, LAMBDA, K1, K2, KO_SIGMA, nbx, nby)

        def body(carry, _):
            c, f = carry
            return sub(c, f, ps, ccx, ccy, fa), None
        (c_f, f_f), _ = jax.jit(
            lambda c, f: jax.lax.scan(body, (c, f), None, length=200)
        )(cb, fb)

        ez_fine = np.asarray(f_f[0, EZ, NG:NG+BS, NG:NG+BS])
        assert np.all(np.isfinite(np.asarray(f_f[0]))), "fine block went non-finite"
        amp = float(np.max(np.abs(ez_fine)))
        amp0 = params.get("id_amp", 1.0)
        assert 0.9 * amp0 < amp < 1.1 * amp0, (
            f"amplitude drift under sub-cycling: |Ez|max={amp:.4f}, initial {amp0:.4f}"
        )


def _n_level_two_active(params_file, nbx=2, nby=2):
    """Full LEVELS-deep state with levels 0 and 1 active (one fine child at
    slot 0, corner (0,0)), from the birefringent IC."""
    nx, ny = nbx * BS, nby * BS
    params = load_parameters(params_file)
    params.update({
        'scheme': 'floating_point', 'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'birefringent', 'bc_type': 'periodic', 'sponge_strength': 0.0,
        'enable_cs': CS, 'Lambda': LAMBDA, 'ko_sigma': KO_SIGMA,
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, LAMBDA, params)
    interior = np.asarray(
        InitialData(sim, params).generate().data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny]
    )
    cb2d = amr_state_from_global(jnp.asarray(interior), nbx, nby)
    csync = sync_ghosts_within_level_root_periodic(cb2d, nbx, nby)
    child0 = prolongate(csync[0], (0, 0))
    blocks = (jnp.zeros((LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG))
              .at[0].set(cb2d).at[1, 0].set(child0))
    active = (jnp.zeros((LEVELS, MAX_BLOCKS), bool)
              .at[0, :nbx*nby].set(True).at[1, 0].set(True))
    state = AMRState(blocks=blocks, active=active)
    topo = AMRTopology()
    for s in range(nbx*nby):
        topo.add_block(0, s, ((s // nby) * BS, (s % nby) * BS))
    topo.add_block(1, 0, (0, 0), parent=(0, 0))
    return state, topo, (cb2d, child0), sim, params


@pytest.mark.amr
class TestSubcycledNLevel:
    """The recursive N-level sub-cycled step must reduce exactly to the 2-level
    step when only levels 0 and 1 are active, and stay stable at depth 3."""

    def test_reduces_to_two_level(self, params_file):
        """N-level with 2 active levels == make_subcycled_two_level_step, to
        machine precision (the recursion's inactive deeper levels are masked
        out and must not perturb the active ones)."""
        nbx, nby = 2, 2
        state, topo, (cb2d, child0), sim, params = _n_level_two_active(params_file, nbx, nby)
        dt_root = CFL * sim.dx
        ta = topo.to_jax_arrays()

        stepN = make_subcycled_n_level_step(
            sim.dx, sim.dy, dt_root, CS, LAMBDA, K1, K2, KO_SIGMA, nbx, nby)
        step2 = make_subcycled_two_level_step(
            sim.dx, sim.dy, dt_root, CS, LAMBDA, K1, K2, KO_SIGMA, nbx, nby)

        ps = jnp.zeros(MAX_BLOCKS, jnp.int32)
        cx = jnp.zeros(MAX_BLOCKS, jnp.int32)
        cy = jnp.zeros(MAX_BLOCKS, jnp.int32)
        fa = jnp.zeros(MAX_BLOCKS, bool).at[0].set(True)

        sN = state
        c2 = cb2d
        f2 = jnp.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG)).at[0].set(child0)
        for _ in range(30):
            sN = stepN(sN, ta)
            c2, f2 = step2(c2, f2, ps, cx, cy, fa)

        # Fine (level 1) interiors must match to FP round-off.
        fN = np.asarray(sN.blocks[1][0, :, NG:NG+BS, NG:NG+BS])
        f2i = np.asarray(f2[0, :, NG:NG+BS, NG:NG+BS])
        err = float(np.max(np.abs(fN - f2i)))
        assert err < 1e-12, (
            f"N-level (2 active) diverged from 2-level step by {err:.2e} — the "
            f"recursion does not reduce correctly"
        )

    def test_depth_3_stable(self, params_file):
        """Depth-3 sub-cycled hierarchy: finite + amplitude-bounded."""
        if LEVELS < 3:
            pytest.skip(f"LEVELS={LEVELS} < 3")
        nbx, nby = 2, 2
        state, topo, _, sim, params = _n_level_two_active(params_file, nbx, nby)
        # Refine the level-1 child once more → a level-2 grandchild.
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[1, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        assert int(np.asarray(state.active[2]).sum()) == 4

        dt_root = CFL * sim.dx
        step = make_subcycled_n_level_step(
            sim.dx, sim.dy, dt_root, CS, LAMBDA, K1, K2, KO_SIGMA, nbx, nby)
        ta = topo.to_jax_arrays()

        def body(s, _):
            return step(s, ta), None
        s_f = jax.jit(lambda s: jax.lax.scan(body, s, None, length=40)[0])(state)

        amp0 = params.get("id_amp", 1.0)
        for L in range(3):
            aa = np.asarray(s_f.active[L]); ba = np.asarray(s_f.blocks[L])
            for sl in range(MAX_BLOCKS):
                if not aa[sl]:
                    continue
                assert np.all(np.isfinite(ba[sl])), f"NaN/Inf at level {L} slot {sl}"
                amax = float(np.max(np.abs(ba[sl])))
                assert amax < 1.5 * (amp0 + 1.0), (
                    f"level {L} slot {sl}: |field|max={amax:.3f} blew up"
                )


class TestRolledEqualsUnrolled:
    """The default (rolled, lax.scan) sub-cycled step must match the reference
    unrolled step to MACHINE PRECISION over multiple steps.  The rolled refactor
    itself is bit-identical (same ops, same order); the small (~1e-13) residual is
    M1a — the rolled default factors prolongation out of the stage loop
    (prolong(hermite(·,s)) == hermite(prolong(·),s)), which reassociates the
    prolongation FP relative to the unfactored unrolled reference.  So this test
    doubles as the M1a correctness gate (factored ≈ unfactored)."""

    def test_matches_unrolled_multistep(self):
        rng = np.random.default_rng(0)
        data = jnp.asarray(rng.standard_normal((NF, 2*BS + 2*NG, 2*BS + 2*NG)))
        state, topo = make_root_state(data, 2, 2)
        # depth-3: refine root0 → 4 L1, then one L1 → 4 L2.
        for L in range(min(LEVELS - 1, 2)):
            finder = 0 if L == 0 else int(np.flatnonzero(np.asarray(state.active[L]))[0])
            flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
            flags[L, finder] = REFINE
            state, topo = apply_flags(state, topo, flags)
        # perturb fine interiors so levels carry distinct data
        blk = [np.array(state.blocks[L], copy=True) for L in range(LEVELS)]
        for L in range(1, LEVELS):
            for s in range(topo.caps[L]):
                if topo.active[L, s]:
                    blk[L][s, :, NG:NG+BS, NG:NG+BS] += 0.3 * rng.standard_normal((NF, BS, BS))
        state = AMRState(blocks=tuple(jnp.asarray(b) for b in blk), active=state.active)
        ta = topo.to_jax_arrays()

        dx = 10.0 / (2 * BS)
        args = (dx, dx, 0.05 * dx, 1.0, 0.4, 1.0, 1.0, 0.05, 2, 2)
        rolled = make_subcycled_n_level_step(*args)              # default
        unrolled = make_subcycled_n_level_step_unrolled(*args)   # reference

        su = sr = state
        for _ in range(5):
            su = unrolled(su, ta)
            sr = rolled(sr, ta)
        for L in range(LEVELS):
            a, b = np.asarray(su.blocks[L]), np.asarray(sr.blocks[L])
            assert np.allclose(a, b, rtol=0.0, atol=1e-11), \
                f"rolled != unrolled at level {L} after 5 steps (max|Δ|={np.max(np.abs(a-b)):.2e})"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
