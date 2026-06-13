"""
Convergence + structural-correctness guards for the MCS 3D solver (Phase 1).

The 3D mirror of `2D/tests/validation/test_convergence.py`.  All CPU-safe and
fast (small grids, few steps): these catch the bug classes a single-resolution
oracle comparison misses:

  * wrong FD order               -> spatial-convergence slope tests
  * a missing/incorrect coupling -> xi-tracking, constraint preservation
  * an axis swap or one-axis sign error -> cyclic axis-rotation symmetry
  * accidental nonlinearity / ghost contamination -> energy-spectrum non-aliasing
  * KO degrading smooth accuracy -> KO-no-degrade
  * the methodology BBH will need (no analytic solution) -> Gaussian self-convergence

3D grids are kept small because cost scales as (N + 2*NG)^3.
"""

import math

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs3d.validate import (
    run_sim, field_l2_errors, energy_spectrum_offmode,
    FullBirefringentOracle, FIELD_NAMES, EZ, XI, NF,
)
from mcs3d.main import get_physical, calc_constraints, l2norm

REF_SCHEME = "floating_point"


def _orders(scheme, resolutions, n_steps, fields, **overrides):
    """Fit the spatial-convergence order (slope of log L2-error vs log N)."""
    logN, logE = [], {f: [] for f in fields}
    for N in resolutions:
        sim, state, params = run_sim(scheme, N, N, N, n_steps, **overrides)
        errs = field_l2_errors(sim, state, params, n_steps * sim.dt)
        logN.append(math.log(N))
        for f in fields:
            logE[f].append(math.log(errs[f]))
    return {f: -float(np.polyfit(logN, logE[f], 1)[0]) for f in fields}


# ── 1. Spatial convergence order ──────────────────────────────────────────────

class TestSpatialConvergence:
    """Loose high-order guard: every propagating field must converge at HIGH
    order (>= 5).  Uses the fixed-step protocol (T ∝ dx), which inflates the
    apparent order by ~1 -- it is a guard, not the certified spatial order (that
    lives in TestOrderSeparation)."""
    RES = (12, 16, 24)
    N_STEPS = 20
    MIN_ORDER = 5.0

    def test_em_fields_high_order(self):
        orders = _orders(REF_SCHEME, self.RES, self.N_STEPS,
                         ["Ex", "Ey", "Ez", "Bx", "By", "Bz"])
        for f, p in orders.items():
            assert p > self.MIN_ORDER, f"{f}: spatial order {p:.2f} < {self.MIN_ORDER}"

    def test_xi_tracks_oracle(self):
        """xi(t) = -cs*Pi0*t is spatially uniform, so the only error is the
        Pi->xi coupling and time integration: it must sit near machine zero."""
        sim, state, params = run_sim(REF_SCHEME, 16, 16, 16, self.N_STEPS)
        errs = field_l2_errors(sim, state, params, self.N_STEPS * sim.dt)
        assert errs["xi"] < 1e-7, f"xi-oracle L2 = {errs['xi']:.2e}"


# ── 2. KO does not degrade order at the operating CFL ─────────────────────────

class TestKODoesNotDegradeOrder:
    """At the operating CFL (dt ∝ dx, fixed step count), Kreiss-Oliger dissipation
    must not lower the observed convergence order of a smooth solution."""
    RES = (12, 16, 24)
    N_STEPS = 20

    def test_order_unchanged_by_ko(self):
        off = _orders(REF_SCHEME, self.RES, self.N_STEPS, ["Ez"], ko_sigma=0.0)["Ez"]
        on  = _orders(REF_SCHEME, self.RES, self.N_STEPS, ["Ez"], ko_sigma=0.05)["Ez"]
        assert on > 5.0, f"KO degraded order at operating CFL: KO-on order = {on:.2f} (< 5.0)"
        assert on > off - 1.0, f"KO lowered order: off={off:.2f}, on={on:.2f}"


# ── 2b. Explicit spatial/temporal order separation ────────────────────────────

def _run_floating_dt(N, n_steps, dt, **over):
    """Evolve `floating_point` with an EXPLICIT dt so dx and dt vary independently
    -- the whole point of order separation."""
    sim, state0, params = run_sim("floating_point", N, N, N, 0, **over)
    def body(c, _):
        return sim.step_rk4(c, dt), None
    state = jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_steps)[0])(state0)
    return sim, state, params


def _ez_err(sim, state, params, t):
    orc = FullBirefringentOracle(params)
    x = np.asarray(sim.x[sim.ng:-sim.ng]); y = np.asarray(sim.y[sim.ng:-sim.ng])
    z = np.asarray(sim.z[sim.ng:-sim.ng])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    num = np.asarray(get_physical(state.data[EZ], sim.ng))
    return float(np.sqrt(np.mean((num - orc.Ez(X, Y, Z, t)) ** 2)))


class TestOrderSeparation:
    """Separate the spatial (FD stencil) and temporal (RK4) orders so neither
    masks the other.  With dt ∝ dx the O(dt⁴) temporal and O(h⁶) spatial errors
    entangle and the lower order wins; we decouple them by holding one fixed."""

    def test_spatial_stencil_is_sixth_order(self):
        """Isolated spatial order of the 6th-order stencil (KO off, fixed tiny dt)."""
        Ns, dt, n = (12, 16, 24), 0.002, 100      # temporal floor << spatial
        errs = []
        for N in Ns:
            sim, st, params = _run_floating_dt(N, n, dt, ko_sigma=0.0)
            errs.append(_ez_err(sim, st, params, n * dt))
        order = -float(np.polyfit(np.log(Ns), np.log(errs), 1)[0])
        assert order > 5.6, (
            f"isolated spatial order = {order:.2f} (expected ~6); "
            f"errs={[f'{e:.2e}' for e in errs]}")

    def test_temporal_rk4_is_fourth_order(self):
        """Isolated temporal order of RK4 (fine fixed dx, vary dt vs dt/16 ref)."""
        N, dt0, n0 = 24, 0.01, 16
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
    analytically.  The scheme must preserve that to ROUND-OFF at every resolution."""
    RES = (12, 16, 24)
    N_STEPS = 20
    TOL = 1e-12

    def test_constraints_at_roundoff_all_resolutions(self):
        for N in self.RES:
            sim, state, _ = run_sim(REF_SCHEME, N, N, N, self.N_STEPS)
            divE, divB = calc_constraints(sim, state)
            dE = float(l2norm(get_physical(divE, sim.ng)))
            dB = float(l2norm(get_physical(divB, sim.ng)))
            assert dE < self.TOL, f"N={N}: divE = {dE:.2e} above round-off"
            assert dB < self.TOL, f"N={N}: divB = {dB:.2e} above round-off"


# ── 4. Energy-spectrum non-aliasing ───────────────────────────────────────────

class TestSpectralPurity:
    """The birefringent IC is a single Fourier mode.  Because MCS is linear, ALL
    Ez power must stay in that mode -- any leakage signals aliasing, ghost-cell
    contamination, or an accidental nonlinearity."""
    def test_no_mode_leakage(self):
        sim, state, params = run_sim(REF_SCHEME, 24, 24, 24, 30)
        ratio = energy_spectrum_offmode(sim, state, params)
        assert ratio < 1e-12, f"off-mode/on-mode power = {ratio:.2e} (expected ~0)"


# ── 5. Cyclic axis-rotation symmetry ──────────────────────────────────────────

class TestAxisSymmetry:
    """The discretization must treat x, y, z identically.  We test the cyclic
    axis rotation x->y->z->x (a PROPER rotation, det +1, so both the polar E and
    axial B components simply permute -- no sign flips).  The rotation R must
    commute with the evolution: Rinv(evolve(R(u0))) == evolve(u0).  Catches a
    derivative on the wrong axis, an asymmetric stencil, or an axis sign error.

    Uses an x-offset Gaussian IC (asymmetric under the rotation), so the test has
    real content -- a symmetric IC would pass trivially.  Confirmed to machine
    precision in the Phase 1 audit.
    """
    SRC = [2, 0, 1, 5, 3, 4, 6, 7, 8, 9]      # new[i] = old[SRC[i]]
    INV = [1, 2, 0, 4, 5, 3, 6, 7, 8, 9]

    def _R(self, data):
        return np.transpose(data[self.SRC], (0, 3, 1, 2)).copy()

    def _Rinv(self, data):
        return np.transpose(data[self.INV], (0, 2, 3, 1)).copy()

    def test_cyclic_rotation(self):
        from mcs_common.wave_state import WaveState
        N, STEPS = 20, 30
        sim, s0, _ = run_sim(REF_SCHEME, N, N, N, 0, id_type="gaussian",
                             Lambda=0.2, id_x0=1.5, id_y0=0.0, id_z0=0.0)
        def body(c, _):
            return sim.step_rk4(c, sim.dt), None
        ev = jax.jit(lambda st: jax.lax.scan(body, st, None, length=STEPS)[0])

        orig = np.asarray(ev(s0).data)
        s_rot = ev(WaveState(jnp.asarray(self._R(np.asarray(s0.data)))))
        recovered = self._Rinv(np.asarray(s_rot.data))
        ng = sim.ng
        diff = float(np.max(np.abs(recovered[:, ng:-ng, ng:-ng, ng:-ng]
                                   - orig[:, ng:-ng, ng:-ng, ng:-ng])))
        assert diff < 1e-10, f"cyclic axis symmetry broken: max diff = {diff:.2e}"


# ── 6. Self-convergence WITHOUT an oracle (the BBH methodology) ────────────────

class TestSelfConvergenceGaussian:
    """The Gaussian IC has no closed-form solution, so correctness is proven by
    Richardson self-convergence -- exactly the technique BBH will require.  Three
    resolutions h, h/2, h/4 reaching the SAME physical time give a convergence
    factor Q = ||u_h - u_2h|| / ||u_2h - u_4h|| -> 2^p."""
    @pytest.mark.slow
    def test_richardson_factor(self):
        def ez_coarse(N, n_steps):
            sim, state, _ = run_sim(REF_SCHEME, N, N, N, n_steps,
                                    id_type="gaussian", Lambda=0.2)
            return np.asarray(get_physical(state.data[EZ], sim.ng))
        n0 = 15
        a = ez_coarse(16, n0)
        b = ez_coarse(32, n0 * 2)[::2, ::2, ::2]
        c = ez_coarse(64, n0 * 4)[::4, ::4, ::4]
        d1 = float(np.sqrt(np.mean((a - b) ** 2)))
        d2 = float(np.sqrt(np.mean((b - c) ** 2)))
        Q = d1 / d2
        assert d2 > 0, "degenerate: finer solutions identical"
        assert Q > 16.0, (
            f"self-convergence factor Q = {Q:.1f} (< 16 => order < 4); "
            f"||h-2h||={d1:.2e}, ||2h-4h||={d2:.2e}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
