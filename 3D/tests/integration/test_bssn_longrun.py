"""Tier-A4 long-run stability in crossing times (see ``docs/BSSN_VALIDATION_PLAN.md``).

Promotes the Phase-2.3 apples tests from a fixed 40 steps to several light-crossing
times (the cubic domain is [-0.5, 0.5]^3, length L = 1, c = 1 -> one crossing time = L
= 1, i.e. 1/dt steps). The evolution is sampled in chunks so the Hamiltonian/momentum
constraint L2 history ||H||(t), ||M||(t) is recorded over the run -- the regime the
production CAHD+SSL gauge was chosen for, previously untested at length. The same
series is what A5 (constraint growth-rate) / A6 (CAHD damping-rate) consume.

CPU scope here is a SHORT long-run (2 crossing times, modest N), kept ``slow``. The
extended runs (many crossing times at production resolution) are a Marylou job: see
``3D/src/bssn3d/longrun_stability.py`` (run instructions in BSSN_VALIDATION_RESULTS.md).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams
from bssn3d.evolve import BSSNEvolution
from bssn3d.constraints import ConstraintSolver
from bssn3d import initial_data as bid

CFL = 0.25


def run_series(scheme, N, n_cross, n_samples, ko_sigma, id_fn):
    """Evolve in chunks, sampling (t, ||H||, ||M||, max|alpha|, finite, max-dev) along
    the way. Returns the grid, the sample rows, and the final state."""
    g = Grid.from_domain(N, order=6)
    ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=ko_sigma,
                       bc="periodic", scheme=scheme)
    cs = ConstraintSolver(g, order=6)
    dt = CFL * g.dx
    total = round(n_cross / dt)               # crossing time = L = 1
    chunk = max(1, round(total / n_samples))
    mink = bid.minkowski(g).data

    state = id_fn(g)
    t = 0.0
    rows = []

    def sample(s, t):
        H, M = cs.l2(s)
        rows.append(dict(
            t=float(t), ncross=float(t),               # crossing time = 1
            H=float(H), M=float(M),
            max_alpha=float(jnp.max(jnp.abs(s.alpha))),
            max_dev=float(jnp.max(jnp.abs(s.data - mink))),
            finite=bool(jnp.all(jnp.isfinite(s.data))),
        ))

    sample(state, t)
    for _ in range(n_samples):
        state = ev.evolve(state, dt, chunk, t0=t)
        t += chunk * dt
        sample(state, t)
    return g, rows, state


def _print_series(title, rows):
    print(f"\n  {title}")
    print("    t(cross)     ||H||         ||M||      max|alpha|    max-dev   finite")
    for r in rows:
        print(f"    {r['ncross']:7.3f}  {r['H']:.4e}  {r['M']:.4e}  "
              f"{r['max_alpha']:9.5f}  {r['max_dev']:.3e}  {r['finite']}")


@pytest.mark.slow
def test_gauge_wave_long_run():
    """Gauge wave stays bounded over 2 crossing times; H/M history recorded."""
    _, rows, _ = run_series("verbatim", N=12, n_cross=2.0, n_samples=8,
                            ko_sigma=0.1, id_fn=lambda g: bid.gauge_wave(g, amplitude=0.01))
    _print_series("gauge wave (N=12, 2 crossing times)", rows)
    assert all(r["finite"] for r in rows)
    assert max(r["max_alpha"] for r in rows) < 2.0      # lapse O(1), no blow-up
    assert max(r["H"] for r in rows) < 1.0              # CAHD keeps Hamiltonian bounded
    assert max(r["M"] for r in rows) < 2.0              # momentum under-protected, lenient


@pytest.mark.slow
def test_robust_stability_long_run():
    """Minkowski + 1e-8 noise stays bounded over 2 crossing times (no exponential
    blow-up); deviation history recorded. The classic robustness test, at length."""
    _, rows, _ = run_series("verbatim", N=12, n_cross=2.0, n_samples=8,
                            ko_sigma=0.1,
                            id_fn=lambda g: bid.robust_stability(g, amp=1e-8, seed=1))
    _print_series("robust stability (N=12, 2 crossing times, amp=1e-8)", rows)
    assert all(r["finite"] for r in rows)
    assert max(r["max_dev"] for r in rows) < 1e-3       # noise stays bounded, no blow-up
    assert max(r["max_alpha"] for r in rows) < 2.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "slow"]))
