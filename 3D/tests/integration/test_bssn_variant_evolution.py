"""Tier-A3 cross-variant EVOLUTION equivalence (see ``docs/BSSN_VALIDATION_PLAN.md``).

The Phase-2/3 variant gates (``test_bssn_{staged,pallas,fused}.py``) compare a SINGLE
RHS evaluation (the oracle inputs + one small-grid A/B). A3 guards the literal thesis
claim — *swap the RHS kernel, keep the physics* — over a FULL multi-step evolution
(RK4 + KO + algebraic enforcement + periodic BC), where floating-point reassociation
between variants accumulates step over step rather than appearing once.

CPU scope (this file):
  * verbatim vs staged  — both pure JAX fp64; staged adds only ``optimization_barrier``
    cut points (a fusion constraint, numerically a no-op) → must stay at round-off over
    the whole evolution.
  * verbatim vs pallas  — pallas runs here in Pallas ``interpret`` mode (CPU math
    simulator, NOT real compilation); the recompute schedule reorders fp summation →
    round-off (~1e-10). Slower (interpret), kept short + ``slow``-marked.

Deferred to Marylou (GPU), NOT run here: the ``fused``/``fused_tiled`` evolution
equivalence and the fp32-algebra divergence rate — interpret-mode full evolutions are
too slow on CPU, and the fp32 accuracy budget is only meaningful on the real kernel.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams, VAR_NAMES
from bssn3d.evolve import BSSNEvolution
from bssn3d import initial_data as bid


def _evolve(scheme, N, nsteps, ko_sigma=0.1):
    g = Grid.from_domain(N, order=6)
    ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=ko_sigma,
                       bc="periodic", scheme=scheme)
    s = ev.evolve(bid.gauge_wave(g, amplitude=0.01, wavelength=1.0),
                  dt=0.25 * g.dx, nsteps=nsteps)
    return np.asarray(s.data)


def _max_field_diff(a, b):
    """Max abs diff overall + the field where it occurs (for the log)."""
    per_field = np.max(np.abs(a - b), axis=(1, 2, 3))
    v = int(np.argmax(per_field))
    return float(per_field[v]), VAR_NAMES[v]


@pytest.mark.slow
def test_verbatim_vs_staged_evolution():
    """staged == verbatim to round-off over a full gauge-wave evolution (16 steps)."""
    N, NSTEPS = 16, 16
    a = _evolve("verbatim", N, NSTEPS)
    b = _evolve("staged", N, NSTEPS)
    diff, where = _max_field_diff(a, b)
    print(f"\n  verbatim vs staged: max|Δ| = {diff:.3e} (field {where}) "
          f"after {NSTEPS} steps at N={N}")
    assert diff < 1e-12, f"staged drifted from verbatim by {diff:.3e} (field {where})"


@pytest.mark.slow
def test_verbatim_vs_pallas_evolution():
    """pallas (interpret mode) == verbatim to round-off over a short evolution.

    Interpret mode is the CPU math simulator (not compilation), so this checks the
    scheduled-kernel algebra survives an evolution, not its register/SMEM behavior
    (that is a Marylou measurement). Kept short — interpret-mode RK4 is slow."""
    N, NSTEPS = 12, 6
    a = _evolve("verbatim", N, NSTEPS)
    b = _evolve("pallas", N, NSTEPS)
    diff, where = _max_field_diff(a, b)
    print(f"\n  verbatim vs pallas: max|Δ| = {diff:.3e} (field {where}) "
          f"after {NSTEPS} steps at N={N}")
    assert diff < 1e-9, f"pallas drifted from verbatim by {diff:.3e} (field {where})"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "slow"]))
