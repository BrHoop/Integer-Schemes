"""Tier-B2 polarized Gowdy validation (see ``docs/BSSN_VALIDATION_PLAN.md``).

The polarized Gowdy T^3 cosmology is an exact, CURVED, nonlinear vacuum solution -- the
first testbed that stresses the RHS's real curvature dynamics, which the flat gauge waves
(A1/A2/B1) leave thin. This file is the **make-or-break gate**: a wrong analytic reduction
silently corrupts everything downstream, so before any evolution we confirm the ID is a
genuine Einstein solution by its discrete constraints converging to zero at FD order.

Because the metric is genuinely curved, the Hamiltonian/momentum constraints are a
non-trivial cancellation of LARGE curvature terms -- a reduction bug has nowhere to hide
(on flat space some errors cancel; here they cannot). So this is a real test of the
curved-space RHS plumbing, not just a transcription check.

Single-eval (no evolution) -> fast (~10-15 s), CPU-safe. Same gauge caveat as A2/B1
applies to any *evolution* of Gowdy (1+log+Gamma-driver won't preserve it); evolve-based
self-convergence is a follow-up, this gate is convergence-of-the-exact-constraints.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams, NUM_VARS, VAR_NAMES
from bssn3d.evolve import BSSNEvolution
from bssn3d.constraints import ConstraintSolver
from bssn3d import initial_data as bid

RES = (24, 32, 48)              # coarse, medium, fine
TIMES = (1.0, 1.5, 2.0)         # t0=1 (smooth expanding slice) + two later times


def _constraint_orders_at(t):
    """(order_H, order_M, H_list, M_list) of the Gowdy solution's discrete constraints."""
    Hs, Ms = [], []
    for N in RES:
        g = Grid.from_domain(N, order=6)
        cs = ConstraintSolver(g, order=6)
        h, m = cs.l2(bid.gowdy_solution(g, t=t))
        Hs.append(float(h))
        Ms.append(float(m))
    logN = np.log(RES)
    oH = -float(np.polyfit(logN, np.log(Hs), 1)[0])
    oM = -float(np.polyfit(logN, np.log(Ms), 1)[0])
    return oH, oM, Hs, Ms


def test_gowdy_wrapper_matches_solution_at_t0():
    """``gowdy(g)`` is ``gowdy_solution(g, t=1.0)`` (the default expanding slice)."""
    g = Grid.from_domain(16, order=6)
    a = np.asarray(bid.gowdy(g).data)
    b = np.asarray(bid.gowdy_solution(g, t=1.0).data)
    assert float(np.max(np.abs(a - b))) == 0.0


def test_gowdy_constraints_converge_to_exact():
    """The Gowdy ID's discrete H and M^i must converge to zero at ~FD order (>= 5) at
    t = 1.0, 1.5, 2.0 -- confirming the analytic reduction is a genuine vacuum Einstein
    solution (curved, so this is a real test of the curvature plumbing)."""
    print("\n     t      order_H   order_M     H(24->48)          M(24->48)")
    worst = []
    for t in TIMES:
        oH, oM, Hs, Ms = _constraint_orders_at(t)
        print(f"  {t:4.1f}   {oH:8.2f} {oM:8.2f}   {Hs[0]:.3e}->{Hs[-1]:.3e}  "
              f"{Ms[0]:.3e}->{Ms[-1]:.3e}")
        worst += [oH, oM]
    assert min(worst) > 5.0, (
        f"a Gowdy constraint order fell below 5.0: {[round(w, 2) for w in worst]} "
        f"(reduction bug -> constraints do not converge)")


# ── B2 self-convergence: reference-free Richardson Q on the CURVED Gowdy evolution ────
# A1's BBH-ready methodology (no analytic solution used), but on a genuinely curved,
# nonlinear, dynamical background where Gowdy earns its keep over the flat gauge waves.
# Same gauge caveat applies (production gauge won't preserve Gowdy), but self-convergence
# needs only a SMOOTH, CONVERGENT discrete evolution -- not a match to the exact solution.
# CFL-locked schedule: nsteps doubles with N so all three reach the same physical time
# with EXACT integer step counts. KO off to read the cleanest spatial order (cf. A1).
SC_CFL = 0.25
SC_N0_STEPS = 8
SC_RES = (12, 24, 48)


def _evolve_gowdy(N, nsteps, ko_sigma):
    g = Grid.from_domain(N, order=6)
    ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=ko_sigma, bc="periodic")
    s = ev.evolve(bid.gowdy(g, t0=1.0), dt=SC_CFL * g.dx, nsteps=nsteps)
    ng = g.ng
    return np.asarray(s.data[:, ng:-ng, ng:-ng, ng:-ng])     # (NUM_VARS, N, N, N)


def _gowdy_per_field_Q(ko_sigma):
    nc, nm, nf = SC_RES
    a = _evolve_gowdy(nc, SC_N0_STEPS, ko_sigma)
    b = _evolve_gowdy(nm, 2 * SC_N0_STEPS, ko_sigma)[:, ::2, ::2, ::2]
    c = _evolve_gowdy(nf, 4 * SC_N0_STEPS, ko_sigma)[:, ::4, ::4, ::4]
    out = {}
    for v in range(NUM_VARS):
        d1 = float(np.sqrt(np.mean((a[v] - b[v]) ** 2)))
        d2 = float(np.sqrt(np.mean((b[v] - c[v]) ** 2)))
        out[VAR_NAMES[v]] = (d1, d2, (d1 / d2) if d2 > 0.0 else float("nan"))
    return out


@pytest.mark.slow
def test_gowdy_self_convergence():
    """Reference-free Richardson self-convergence of a full Gowdy evolution -- the A1
    BBH-style guard on a CURVED, nonlinear, dynamical background. Every field carrying
    real signal (d2 above a floor) must self-converge at >= 4th order (Q > 14 absorbs the
    RK4-limited constant); zero-signal fields (off-diagonals preserved by the polarized
    structure) are skipped. Per-field Q is printed regardless."""
    Q = _gowdy_per_field_Q(ko_sigma=0.0)
    print("\n  field        ||h-2h||      ||2h-4h||       Q")
    for name, (d1, d2, q) in Q.items():
        print(f"  {name:<8s} {d1:12.4e} {d2:12.4e} {q:10.2f}")

    SIGNAL = 1e-10
    active = {n: q for n, (d1, d2, q) in Q.items() if d2 > SIGNAL}
    assert active, "no field carried convergence signal — degenerate run"
    worst = min(active.values())
    assert worst > 14.0, (
        f"worst Gowdy self-convergence Q = {worst:.2f} (< 14 => order < ~3.8); "
        f"per-field Q = { {n: round(q, 2) for n, q in active.items()} }")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
