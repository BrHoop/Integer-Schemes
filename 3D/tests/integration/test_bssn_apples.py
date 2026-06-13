"""Phase 2.3 apples-with-apples evolution tests.

These drive the full evolution (RK4 + KO + algebraic enforcement + periodic BC).
Each compiles a graph ~4x the RHS, so they are kept SMALL and the multi-step ones
are marked ``slow`` (heavy local JAX compiles can swamp a laptop — run with
``-m slow`` deliberately, or on the H200).

Covered:
  * conformal algebra preserved under evolution (det ~g = 1, tr ~A = 0, chi > 0)
  * robust stability: Minkowski + noise stays bounded (no exponential blow-up)
  * gauge-wave dynamical stability + constraints stay bounded over the run
"""

import jax
import jax.numpy as jnp
import pytest

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams, GT0, AT0
from bssn3d.evolve import BSSNEvolution
from bssn3d.constraints import ConstraintSolver
from bssn3d import initial_data as bid

jax.config.update("jax_enable_x64", True)


def _det_trace(state):
    d = state.data
    g = [d[GT0 + i] for i in range(6)]
    det = (g[0] * (g[3] * g[5] - g[4] ** 2) - g[1] * (g[1] * g[5] - g[4] * g[2])
           + g[2] * (g[1] * g[4] - g[3] * g[2]))
    iuxx, iuyy, iuzz = g[3] * g[5] - g[4] ** 2, g[0] * g[5] - g[2] ** 2, g[0] * g[3] - g[1] ** 2
    iuxy, iuxz, iuyz = g[2] * g[4] - g[1] * g[5], g[1] * g[4] - g[2] * g[3], g[1] * g[2] - g[0] * g[4]
    a = [d[AT0 + i] for i in range(6)]
    trA = (iuxx * a[0] + iuyy * a[3] + iuzz * a[5]
           + 2 * (iuxy * a[1] + iuxz * a[2] + iuyz * a[4])) / det
    return det, trA


def test_conformal_algebra_preserved():
    g = Grid.from_domain(12, order=6)
    ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=0.1, bc="periodic")
    s = ev.evolve(bid.gauge_wave(g, amplitude=0.01), dt=0.25 * g.dx, nsteps=8)
    det, trA = _det_trace(s)
    assert bool(jnp.all(jnp.isfinite(s.data)))
    assert float(jnp.max(jnp.abs(det - 1.0))) < 1e-12   # enforcement holds det ~g = 1
    assert float(jnp.max(jnp.abs(trA))) < 1e-10         # ... and tr ~A = 0
    assert float(jnp.min(s.chi)) > 0.0                  # chi floor


@pytest.mark.slow
def test_robust_stability_bounded():
    g = Grid.from_domain(12, order=6)
    ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=0.1, bc="periodic")
    amp = 1e-8
    s = ev.evolve(bid.robust_stability(g, amp=amp, seed=1), dt=0.25 * g.dx, nsteps=40)
    mink = bid.minkowski(g).data
    dev = float(jnp.max(jnp.abs(s.data - mink)))
    assert bool(jnp.all(jnp.isfinite(s.data)))
    assert dev < 1e-4, f"noise grew to {dev:.1e} (>{amp:.0e} start) — possible instability"


@pytest.mark.slow
def test_gauge_wave_dynamically_stable():
    """Low-amplitude gauge wave under the production 1+log/Gamma-driver gauge stays
    bounded (no blow-up). NOTE: this CAHD+SSL variant damps the HAMILTONIAN
    constraint (CAHD on chi) + locks the slice (SSL on the lapse), but the MOMENTUM
    constraint stays under-protected (only Gt + KO), so it can still drift secularly
    at coarse resolution — expected, not a bug. This asserts boundedness only; the
    *rigorous* constraint check is the 6th-order convergence of the discrete
    constraints on the analytic ID (test_bssn_constraints.py)."""
    g = Grid.from_domain(16, order=6)
    ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=0.1, bc="periodic")
    cs = ConstraintSolver(g, order=6)
    gw = bid.gauge_wave(g, amplitude=0.01)
    s = ev.evolve(gw, dt=0.25 * g.dx, nsteps=40)
    H, M = cs.l2(s)
    assert bool(jnp.all(jnp.isfinite(s.data)))
    assert float(jnp.max(jnp.abs(s.alpha))) < 2.0      # lapse stays O(1), no blow-up
    assert H < 0.5 and M < 0.5                          # bounded (not exponential growth)
