"""Phase 3.1 correctness gate: the STAGED RHS must match the VERBATIM RHS.

The staged variant (``_bssn_rhs_staged.py``) is the same Dendro-GR algebra with
``optimization_barrier`` cut points pinned on the high-fan-out tensor-hierarchy
temps — a *fusion* constraint only, numerically a no-op. So it must reproduce the
bit-validated verbatim RHS to round-off (the plan's ~1e-12 bar; with barriers
alone — no remat/reorder — it is in fact bit-identical, which we also record).

Two scales: a single point (reuse the oracle's non-degenerate inputs) and a small
3D gauge-wave grid through the full ``BSSNSolver`` A/B (``scheme="verbatim"`` vs
``"staged"``), so the FD bundle + barriers are exercised end-to-end. CPU-only, no
GPU/2FA — this gate is about *equivalence*; the spill/kernel-count win is the H200
``spill_probe`` measurement.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bssn3d import oracle
from bssn3d._bssn_rhs_generated import bssn_rhs_algebra as rhs_verbatim
from bssn3d._bssn_rhs_staged import bssn_rhs_algebra as rhs_staged
from bssn3d._bssn_rhs_staged import CUT_SET
from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams
from bssn3d.rhs import BSSNSolver
from bssn3d import initial_data as bid

TOL = 1e-12   # the plan's round-off bar (barriers alone are bit-identical)


def _call(algebra, values):
    F = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["fields"].items()}
    D = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["derivs"].items()}
    out = algebra(
        F, D, values["eta"], values["lmbda"], values["lambda_f"],
        values["BSSN_CAHD_C"], values["dt"], values["dx_i"],
        values["h_ssl"], values["sig_ssl"], values["t"],
    )
    return {k: float(np.asarray(v)[0]) for k, v in out.items()}


def test_staged_module_has_barriers():
    """Guard: the staged module actually pins a non-trivial cut-set."""
    assert len(CUT_SET) >= 32, f"staged cut-set unexpectedly small: {len(CUT_SET)}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_staged_equals_verbatim_pointwise(seed):
    values = oracle.random_inputs(seed)
    v = _call(rhs_verbatim, values)
    s = _call(rhs_staged, values)
    worst, worst_field = 0.0, None
    for k in v:
        rel = abs(v[k] - s[k]) / max(abs(v[k]), abs(s[k]), 1e-12)
        if rel > worst:
            worst, worst_field = rel, k
    assert worst < TOL, (
        f"seed {seed}: staged vs verbatim max rel diff {worst:.3e} on {worst_field} "
        f"exceeds {TOL:.0e}"
    )


def test_staged_equals_verbatim_gridded():
    """Full-pipeline A/B on a small 3D gauge wave (FD bundle + barriers + algebra)."""
    grid = Grid.from_domain(16, order=6)
    state = bid.gauge_wave(grid, amplitude=0.01)
    params = PhysicsParams()

    out_v = BSSNSolver(grid, params, order=6, scheme="verbatim").rhs(state)
    out_s = BSSNSolver(grid, params, order=6, scheme="staged").rhs(state)

    dv = np.asarray(out_v.data)
    ds = np.asarray(out_s.data)
    denom = np.maximum(np.maximum(np.abs(dv), np.abs(ds)), 1e-12)
    worst = float(np.max(np.abs(dv - ds) / denom))
    assert worst < TOL, f"gridded staged vs verbatim max rel diff {worst:.3e} > {TOL:.0e}"
