"""
Convergence + structural-correctness guards for the MCS 2D solver (Step 1.1).

All CPU-safe and fast (small grids, few steps): these are meant to run on every
change, not just on the H200.  They catch the bug classes that a single-resolution
oracle comparison misses:

  * wrong FD order               -> spatial-convergence slope tests
  * a missing/incorrect coupling -> xi-tracking, constraint-convergence
  * an x/y axis swap or one-axis sign error -> discrete x<->y symmetry
  * accidental nonlinearity / ghost contamination -> energy-spectrum non-aliasing
  * KO degrading smooth accuracy -> KO-no-degrade
  * the methodology BBH will need (no analytic solution) -> Gaussian self-convergence
"""

import math

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.validate import (
    run_sim, field_l2_errors, energy_spectrum_offmode,
    FullBirefringentOracle, FIELD_NAMES, EZ, XI, NF,
)
from mcs2d.main import get_physical, calc_constraints, l2norm


REF_SCHEME = "fused_floating_point"


def _orders(scheme, resolutions, n_steps, fields, **overrides):
    """Fit the spatial-convergence order (slope of log L2-error vs log N) for the
    requested fields against the 10-field oracle."""
    logN, logE = [], {f: [] for f in fields}
    for N in resolutions:
        sim, state, params = run_sim(scheme, N, N, n_steps, **overrides)
        errs = field_l2_errors(sim, state, params, n_steps * sim.dt)
        logN.append(math.log(N))
        for f in fields:
            logE[f].append(math.log(errs[f]))
    return {f: -float(np.polyfit(logN, logE[f], 1)[0]) for f in fields}


# ── 1. Spatial convergence order ──────────────────────────────────────────────

class TestSpatialConvergence:
    """Loose high-order guard: every propagating field must converge at HIGH
    order (>= 5), catching a gross drop (e.g. to 2nd order from a broken stencil).

    Note: this uses the fixed-step protocol (T ∝ dx), which inflates the apparent
    order by ~1, so the numbers here run high -- it is a guard, not the certified
    spatial order.  The rigorous, fixed-time isolated orders (stencil ~6, RK4 ~4)
    live in TestOrderSeparation."""
    RES = (16, 32, 64)
    N_STEPS = 30
    MIN_ORDER = 5.0

    def test_em_fields_sixth_order(self):
        orders = _orders(REF_SCHEME, self.RES, self.N_STEPS,
                         ["Ex", "Ey", "Ez", "Bx", "By", "Bz"])
        for f, p in orders.items():
            assert p > self.MIN_ORDER, f"{f}: spatial order {p:.2f} < {self.MIN_ORDER}"

    def test_xi_tracks_oracle(self):
        """xi(t) = -cs*Pi0*t is spatially uniform, so the only error is the
        Pi->xi coupling and time integration: it must sit near machine zero,
        independent of resolution.  A nonzero spatial structure here would mean
        the scalar coupling is wrong."""
        sim, state, params = run_sim(REF_SCHEME, 32, 32, self.N_STEPS)
        errs = field_l2_errors(sim, state, params, self.N_STEPS * sim.dt)
        assert errs["xi"] < 1e-7, f"xi-oracle L2 = {errs['xi']:.2e}"


# ── 2. KO does not degrade order at the operating CFL ─────────────────────────

class TestKODoesNotDegradeOrder:
    """At the operating CFL (dt ∝ dx, fixed step count), Kreiss-Oliger dissipation
    must not lower the observed convergence order of a smooth solution.

    Note: this is the *production* view (dt ∝ dx).  KO with this normalization
    (σ/dx · 6th-difference) is formally an O(h⁵) dissipation -- one order below
    the 6th-order stencil -- which `TestOrderSeparation` exposes at FIXED time.
    Here the dt ∝ dx protocol shrinks the evolution horizon T ∝ dx as the grid
    refines, and that extra factor of h offsets KO's O(h⁵) so the *observed*
    order stays ~6.  Both statements are true; they just hold the time axis
    differently.  See TestOrderSeparation for the fixed-time spatial certificate.
    """
    RES = (16, 32, 64)
    N_STEPS = 30

    def test_order_unchanged_by_ko(self):
        # Differencing two 3-point slope fits is noisy, so assert the property
        # directly: at the operating CFL, KO-on order stays high and is not pulled
        # meaningfully below the KO-off order.
        off = _orders(REF_SCHEME, self.RES, self.N_STEPS, ["Ez"], ko_sigma=0.0)["Ez"]
        on  = _orders(REF_SCHEME, self.RES, self.N_STEPS, ["Ez"], ko_sigma=0.05)["Ez"]
        assert on > 5.5, f"KO degraded order at operating CFL: KO-on order = {on:.2f} (< 5.5)"
        assert on > off - 1.0, f"KO lowered order: off={off:.2f}, on={on:.2f}"


# ── 2b. Explicit spatial/temporal order separation ────────────────────────────

def _run_floating_dt(N, n_steps, dt, **over):
    """Evolve the unfused `floating_point` scheme with an EXPLICIT dt.  Unfused is
    required here: `step_rk4` ignores its dt argument for fused schemes (they call
    a step kernel with a baked-in dt), so only the unfused path lets us vary dt
    and dx independently -- the whole point of order separation."""
    sim, state0, params = run_sim("floating_point", N, N, 0, **over)
    def body(c, _):
        return sim.step_rk4(c, dt), None
    state = jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_steps)[0])(state0)
    return sim, state, params


def _ez_err(sim, state, params, t):
    orc = FullBirefringentOracle(params)
    x = np.asarray(sim.x[sim.ng:-sim.ng]); y = np.asarray(sim.y[sim.ng:-sim.ng])
    X, Y = np.meshgrid(x, y, indexing="ij")
    num = np.asarray(get_physical(state.data[EZ], sim.ng))
    return float(np.sqrt(np.mean((num - orc.Ez(X, Y, t)) ** 2)))


class TestOrderSeparation:
    """Separate the spatial (FD stencil) and temporal (RK4) orders so neither
    masks the other.

    The gotcha (per the Step 1.1 spec): with dt ∝ dx the temporal error
    O(dt⁴) = O(h⁴) and the spatial error O(h⁶) are entangled, and the lower order
    wins -- you measure 4 and wrongly blame the stencil.  To separate:
      * spatial: hold dt FIXED and tiny (temporal error a negligible constant
        floor), refine h -> slope = 6.  KO is disabled here, because KO is a
        deliberate O(h⁵) dissipation that would otherwise pull the stencil's true
        6th order down toward 5 (see module note in TestKODoesNotDegradeOrder).
      * temporal: hold dx FIXED and fine (spatial error a constant floor), vary
        dt against a near-exact dt/16 reference -> slope = 4.
    """

    def test_spatial_stencil_is_sixth_order(self):
        """Isolated spatial order of the 6th-order stencil (KO off, fixed tiny dt)."""
        Ns, dt, n = (24, 48, 96), 0.001, 100      # temporal floor ~3e-14 << spatial
        errs = []
        for N in Ns:
            sim, st, params = _run_floating_dt(N, n, dt, ko_sigma=0.0)
            errs.append(_ez_err(sim, st, params, n * dt))
        order = -float(np.polyfit(np.log(Ns), np.log(errs), 1)[0])
        assert order > 5.7, (
            f"isolated spatial order = {order:.2f} (expected ~6); "
            f"errs={[f'{e:.2e}' for e in errs]}")

    def test_temporal_rk4_is_fourth_order(self):
        """Isolated temporal order of RK4 (fine fixed dx, vary dt vs dt/16 ref)."""
        N, dt0, n0 = 96, 0.01, 16
        ng = 3
        ref = np.asarray(get_physical(
            _run_floating_dt(N, n0 * 16, dt0 / 16, ko_sigma=0.0)[1].data[EZ], ng))
        e1 = np.asarray(get_physical(
            _run_floating_dt(N, n0, dt0, ko_sigma=0.0)[1].data[EZ], ng))
        e2 = np.asarray(get_physical(
            _run_floating_dt(N, n0 * 2, dt0 / 2, ko_sigma=0.0)[1].data[EZ], ng))
        L = lambda a: float(np.sqrt(np.mean((a - ref) ** 2)))
        order = math.log2(L(e1) / L(e2))
        assert 3.5 < order < 4.5, (
            f"temporal order = {order:.2f} (expected ~4); "
            f"err(dt0)={L(e1):.2e}, err(dt0/2)={L(e2):.2e}")


# ── 3. Machine-precision constraint preservation ──────────────────────────────

class TestConstraintPreservation:
    """The birefringent wave is transverse, so divE and divB are identically zero
    analytically.  The scheme must preserve that to ROUND-OFF at every resolution
    -- a stronger statement than 'converges at order p' (there is no order to
    measure: the violation sits at the float64 floor, not above it).  Checking
    multiple resolutions rules out a resolution-dependent fluke; checking after
    evolution rules out slow constraint growth."""
    RES = (16, 32, 64)
    N_STEPS = 30
    TOL = 1e-12     # float64 round-off scaled by the operator norm

    def test_constraints_at_roundoff_all_resolutions(self):
        for N in self.RES:
            sim, state, _ = run_sim(REF_SCHEME, N, N, self.N_STEPS)
            divE, divB = calc_constraints(sim, state)
            dE = float(l2norm(get_physical(divE, sim.ng)))
            dB = float(l2norm(get_physical(divB, sim.ng)))
            assert dE < self.TOL, f"N={N}: divE = {dE:.2e} above round-off"
            assert dB < self.TOL, f"N={N}: divB = {dB:.2e} above round-off"


# ── 4. Energy-spectrum non-aliasing ───────────────────────────────────────────

class TestSpectralPurity:
    """The birefringent IC is a single Fourier mode.  Because MCS is linear, ALL
    Ez power must stay in that mode -- any leakage to harmonics signals aliasing,
    ghost-cell contamination, or an accidental nonlinearity (e.g. a broken Ozaki
    pipeline)."""
    def test_no_mode_leakage(self):
        sim, state, params = run_sim(REF_SCHEME, 64, 64, 50)
        ratio = energy_spectrum_offmode(sim, state, params)
        assert ratio < 1e-12, f"off-mode/on-mode power = {ratio:.2e} (expected ~0)"


# ── 5. Discrete x<->y symmetry ────────────────────────────────────────────────

class TestAxisSymmetry:
    """The discretization must treat x and y identically.  The birefringent
    oracle uses equal kx, ky, which would MASK an axis swap or one-axis sign
    error -- so we test it directly: the diagonal reflection T (swap coordinates
    x<->y and apply the matching field transform) must commute with the
    evolution.  Then T(evolve(T(u0))) == evolve(u0).  Catches a derivative on the
    wrong axis, an asymmetric stencil, or a one-axis sign error.

    The field transform is DERIVED from the RHS (not guessed): under (x,y)->(y,x)
    the in-plane vector components swap (Ex<->Ey, Bx<->By) while the out-of-plane
    components flip sign (Ez->-Ez, Bz->-Bz) and the scalars (xi, Pi, Psi, Phi)
    are invariant.  T is an involution (T^2 = identity), so the same T un-swaps.
    """
    # new_field[i] = SIGN[i] * old_field[SRC[i]]   (applied after spatial transpose)
    SRC  = [1, 0, 2, 4, 3, 5, 6, 7, 8, 9]
    SIGN = np.array([1., 1., -1., 1., 1., -1., 1., 1., 1., 1.]).reshape(10, 1, 1)

    def _T(self, data: np.ndarray) -> np.ndarray:
        """Apply the diagonal-reflection transform to a (10, Nx, Ny) array."""
        return (self.SIGN * data[self.SRC]).transpose(0, 2, 1).copy()

    def test_diagonal_reflection(self):
        from mcs_common.wave_state import WaveState
        N, STEPS = 48, 40
        sim, s_orig, _ = run_sim(REF_SCHEME, N, N, STEPS)
        orig = np.asarray(s_orig.data)                    # full, with ghosts

        sim2, s0, _ = run_sim(REF_SCHEME, N, N, 0)        # fresh IC (with ghosts)
        ic_T = self._T(np.asarray(s0.data))               # transform the IC

        def body(c, _):
            return sim2.step_rk4(c, sim2.dt), None
        s_sw = jax.jit(lambda st: jax.lax.scan(body, st, None, length=STEPS)[0])(
            WaveState(jnp.asarray(ic_T)))

        recovered = self._T(np.asarray(s_sw.data))        # un-swap (T is involution)
        ng = sim.ng
        diff = float(np.max(np.abs(recovered[:, ng:-ng, ng:-ng]
                                   - orig[:, ng:-ng, ng:-ng])))
        assert diff < 1e-10, f"x<->y symmetry broken: max diff = {diff:.2e}"


# ── 6. Self-convergence WITHOUT an oracle (the BBH methodology) ────────────────

class TestSelfConvergenceGaussian:
    """The Gaussian IC has no closed-form solution, so correctness is proven by
    Richardson self-convergence -- exactly the technique BBH will require.  Using
    three resolutions h, h/2, h/4 reaching the SAME physical time, the
    convergence factor Q = ||u_h - u_2h|| / ||u_2h - u_4h|| -> 2^p.

    We compare on the coarse grid's points (every 2nd / 4th node of the finer
    runs) and assert Q indicates high-order convergence (Q > 16, i.e. p > 4),
    establishing the methodology on a problem with no analytic answer.
    """
    def _ez_coarsegrid(self, N, n_steps):
        sim, state, _ = run_sim(REF_SCHEME, N, N, n_steps,
                                id_type="gaussian", Lambda=0.2)
        return np.asarray(get_physical(state.data[EZ], sim.ng)), sim.dt

    def test_richardson_factor(self):
        # Same physical time T across all three: refine space and step count
        # together (dt = cfl*dx, so halving dx halves dt -> double the steps).
        n0 = 20
        ez_h,  dt_h  = self._ez_coarsegrid(32, n0)
        ez_2h, _     = self._ez_coarsegrid(64, n0 * 2)
        ez_4h, _     = self._ez_coarsegrid(128, n0 * 4)

        # Sample the finer grids at the coarse (32x32) node locations.
        a = ez_h
        b = ez_2h[::2, ::2]
        c = ez_4h[::4, ::4]
        d1 = float(np.sqrt(np.mean((a - b) ** 2)))
        d2 = float(np.sqrt(np.mean((b - c) ** 2)))
        Q = d1 / d2
        assert d2 > 0, "degenerate: finer solutions identical"
        assert Q > 16.0, (
            f"self-convergence factor Q = {Q:.1f} (< 16 => order < 4); "
            f"||h-2h||={d1:.2e}, ||2h-4h||={d2:.2e}"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
