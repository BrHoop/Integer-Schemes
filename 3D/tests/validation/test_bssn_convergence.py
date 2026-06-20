"""Tier-A BSSN evolution-validation guards (see ``docs/BSSN_VALIDATION_PLAN.md``).

These exercise the FULL BSSN evolution (RK4 + KO + algebraic enforcement + periodic
BC), unlike the Phase-2 oracle which checks a single RHS eval. They are the evolution
analogue of the MCS ``test_convergence.py`` guards.

A1 — gauge-wave evolution self-convergence (Richardson Q), reference-free.
    The methodology BBH will require: no analytic solution is used, correctness is
    proven by three resolutions h, h/2, h/4 reaching the SAME physical time giving a
    convergence factor Q = ||u_h - u_2h|| / ||u_2h - u_4h|| -> 2^p.

    MEASURED order (2026-06-15, KO off, cfl=0.25, T~0.167, RES=12/24/48): the RK4
    temporal floor does NOT bind at this cfl/short time -- the spatial order shows
    through. The tensor sector (gt*, At*) self-converges at Q ~ 70 ~ 2^6 (6th order);
    the gauge/scalar sector (alpha, chi, K) at Q ~ 33-38 ~ 2^5; the Gt/beta/B sector at
    Q ~ 42-49. Off-diagonal/shift components stay at round-off (no signal). The clean
    spatial/temporal SEPARATION (fixed tiny dt) is A2's job; this is the reference-free
    BBH-style guard. Mirrors the MCS ``TestSelfConvergenceGaussian``.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams, NUM_VARS, VAR_NAMES
from bssn3d.evolve import BSSNEvolution
from bssn3d.constraints import ConstraintSolver
from bssn3d import initial_data as bid


# CFL-locked self-convergence schedule: doubling N and nsteps together holds the
# physical time T = n0 * cfl / N_coarse fixed with EXACT integer step counts at every
# resolution (non-integer T/dt would manufacture error that swamps truncation error).
CFL = 0.25
N0_STEPS = 8          # steps at the coarsest grid (-> 16, 32 at the finer two)
RES = (12, 24, 48)    # coarse, medium, fine; medium/fine subsample onto the coarse grid


def _evolve_gauge_wave(N, nsteps, ko_sigma):
    g = Grid.from_domain(N, order=6)
    ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=ko_sigma, bc="periodic")
    s = ev.evolve(bid.gauge_wave(g, amplitude=0.01, wavelength=1.0),
                  dt=CFL * g.dx, nsteps=nsteps)
    ng = g.ng
    interior = s.data[:, ng:-ng, ng:-ng, ng:-ng]   # (NUM_VARS, N, N, N)
    return np.asarray(interior)


def _per_field_Q(ko_sigma):
    nc, nm, nf = RES
    a = _evolve_gauge_wave(nc, N0_STEPS, ko_sigma)
    b = _evolve_gauge_wave(nm, 2 * N0_STEPS, ko_sigma)[:, ::2, ::2, ::2]
    c = _evolve_gauge_wave(nf, 4 * N0_STEPS, ko_sigma)[:, ::4, ::4, ::4]
    out = {}
    for v in range(NUM_VARS):
        d1 = float(np.sqrt(np.mean((a[v] - b[v]) ** 2)))
        d2 = float(np.sqrt(np.mean((b[v] - c[v]) ** 2)))
        out[VAR_NAMES[v]] = (d1, d2, (d1 / d2) if d2 > 0.0 else float("nan"))
    return out


@pytest.mark.slow
def test_gauge_wave_self_convergence():
    """Reference-free Richardson self-convergence of a full gauge-wave evolution.

    Every field carrying real signal (d2 above a floor) must self-converge at >= 4th
    order (Q > 16 caught loosely as Q > 14 to absorb the RK4-limited constant). Fields
    that stay ~zero (shift beta^i, B^i on this zero-shift ID) carry no signal and are
    skipped. Per-field Q is printed for the results log regardless of pass/fail.
    """
    Q = _per_field_Q(ko_sigma=0.0)

    print("\n  field        ||h-2h||      ||2h-4h||       Q")
    for name, (d1, d2, q) in Q.items():
        print(f"  {name:<8s} {d1:12.4e} {d2:12.4e} {q:10.2f}")

    SIGNAL = 1e-10   # below this a field is effectively zero on the gauge-wave ID
    active = {n: q for n, (d1, d2, q) in Q.items() if d2 > SIGNAL}
    assert active, "no field carried convergence signal — degenerate run"

    worst = min(active.values())
    assert worst > 14.0, (
        f"worst self-convergence factor Q = {worst:.2f} (< 14 => order < ~3.8); "
        f"per-field Q = { {n: round(q, 2) for n, q in active.items()} }")


# ── A2 — gauge-wave convergence-to-exact (gauge-independent: constraints) ─────────
#
# The production CAHD+SSL RHS uses 1+log slicing + an SSL term that drives alpha->1 +
# CAHD damping, so it does NOT preserve the harmonic gauge wave -- an evolved state
# diverges from the analytic traveling solution by an O(amplitude) gauge term that does
# NOT converge away (that is why A1's reference-free self-convergence is the evolution
# check). What IS a clean, gauge-INDEPENDENT convergence-to-exact: the analytic gauge
# wave is an exact Einstein solution at every t, so its discrete Hamiltonian/momentum
# constraints are pure truncation error and converge at the FD order at any time. This
# validates the time-dependent ``gauge_wave_solution(t)`` helper beyond the t=0 case the
# Phase-2 ``test_bssn_constraints.py`` already covers. Single-eval (no evolution) → fast.

EXACT_RES = (24, 32, 48)
EXACT_TIMES = (0.0, 0.15, 0.35)


def _constraint_orders_at(t):
    """(order_H, order_M) of the analytic solution's discrete constraints at time t."""
    Hs, Ms = [], []
    for N in EXACT_RES:
        g = Grid.from_domain(N, order=6)
        cs = ConstraintSolver(g, order=6)
        h, m = cs.l2(bid.gauge_wave_solution(g, t=t, amplitude=0.01, wavelength=1.0))
        Hs.append(float(h))
        Ms.append(float(m))
    logN = np.log(EXACT_RES)
    oH = -float(np.polyfit(logN, np.log(Hs), 1)[0])
    oM = -float(np.polyfit(logN, np.log(Ms), 1)[0])
    return oH, oM, Hs, Ms


def test_gauge_wave_solution_matches_gauge_wave_at_t0():
    """The traveling-solution helper must reduce to the existing t=0 ID to round-off."""
    g = Grid.from_domain(16, order=6)
    a = np.asarray(bid.gauge_wave_solution(g, t=0.0, amplitude=0.01, wavelength=1.0).data)
    b = np.asarray(bid.gauge_wave(g, amplitude=0.01, wavelength=1.0).data)
    assert float(np.max(np.abs(a - b))) < 1e-14


def test_gauge_wave_solution_constraints_converge_to_exact():
    """At t = 0, 0.15, 0.35 the analytic solution's discrete H and M^i must converge to
    zero at ~FD order (>= 5), confirming ``gauge_wave_solution(t)`` is a genuine exact
    Einstein solution at all times — a clean, gauge-independent convergence-to-exact."""
    print("\n     t      order_H   order_M     H(24->48)        M(24->48)")
    worst = []
    for t in EXACT_TIMES:
        oH, oM, Hs, Ms = _constraint_orders_at(t)
        print(f"  {t:5.2f}   {oH:8.2f} {oM:8.2f}   {Hs[0]:.2e}->{Hs[-1]:.2e}  "
              f"{Ms[0]:.2e}->{Ms[-1]:.2e}")
        worst += [oH, oM]
    assert min(worst) > 5.0, f"a constraint order fell below 5.0: {[round(w,2) for w in worst]}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
