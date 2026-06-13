"""Phase 3.2c CPU gate: the register-resident Pallas BSSN kernel == verbatim.

Pallas runs here in ``interpret=True`` mode (CPU math simulator) — this validates the
**algebra** of the scheduled kernel (M-temps stored, R-temps recomputed inline), NOT
its compilation or register behavior. The spill/regime win is a Marylou run
(`spill_probe`/`profile_regime --scheme pallas`); this test is the correctness anchor
that must hold before that GPU work.

Agreement is round-off (~1e-10): the recompute reorders fp summation vs the verbatim
CSE. Single-point (oracle inputs) + a small 3D gauge-wave grid through the full
``BSSNSolver(scheme="pallas")``.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bssn3d import oracle
from bssn3d._bssn_rhs_generated import bssn_rhs_algebra as rhs_verbatim
from bssn3d._bssn_rhs_pallas import bssn_rhs_algebra as rhs_pallas
from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams
from bssn3d.rhs import BSSNSolver
from bssn3d import initial_data as bid

TOL = 1e-10


def _call(algebra, values):
    F = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["fields"].items()}
    D = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["derivs"].items()}
    out = algebra(
        F, D, values["eta"], values["lmbda"], values["lambda_f"],
        values["BSSN_CAHD_C"], values["dt"], values["dx_i"],
        values["h_ssl"], values["sig_ssl"], values["t"],
    )
    return {k: float(np.asarray(v)[0]) for k, v in out.items()}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pallas_equals_verbatim_pointwise(seed):
    values = oracle.random_inputs(seed)
    v = _call(rhs_verbatim, values)
    p = _call(rhs_pallas, values)
    worst, where = 0.0, None
    for k in v:
        rel = abs(v[k] - p[k]) / max(abs(v[k]), abs(p[k]), 1e-12)
        if rel > worst:
            worst, where = rel, k
    assert worst < TOL, f"seed {seed}: pallas vs verbatim {worst:.2e} on {where}"


def test_pallas_equals_verbatim_gridded():
    """Full BSSNSolver A/B on a 3D gauge wave (exercises padding + the wrapper).

    Mixed abs/rel bar: the gauge wave (amp 0.01) has many RHS components near zero,
    where the 1.43x recompute's fp-summation reorder shows as a large *relative* diff
    on a ~1e-12 *absolute* one (cancellation). |Δ| <= atol + rtol·|verbatim| is the
    honest round-off gate.
    """
    grid = Grid.from_domain(16, order=6)
    state = bid.gauge_wave(grid, amplitude=0.01)
    params = PhysicsParams()
    out_v = BSSNSolver(grid, params, order=6, scheme="verbatim").rhs(state)
    out_p = BSSNSolver(grid, params, order=6, scheme="pallas").rhs(state)
    dv, dp = np.asarray(out_v.data), np.asarray(out_p.data)
    atol, rtol = 1e-11, 1e-9
    bad = np.abs(dv - dp) - (atol + rtol * np.abs(dv))
    assert np.all(bad <= 0), (
        f"gridded pallas vs verbatim exceeds atol={atol:.0e}+rtol={rtol:.0e}: "
        f"max excess {float(bad.max()):.2e}, max |Δ| {float(np.abs(dv-dp).max()):.2e}")
