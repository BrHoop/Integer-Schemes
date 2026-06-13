"""Phase 3.2 / 1.2 CPU gate: the derivative-fused BSSN kernel == verbatim.

The fused kernel computes the 138 derivatives ON-CHIP (never materialized to HBM) and
feeds them straight into the scheduled algebra, all in one ``pallas_call``. Pallas runs
here in ``interpret=True`` (CPU math simulator), so these validate the **math** — NOT
the SMEM/occupancy/Triton-lowering payoff, which is the H200 build (3.2d):

  1. the on-chip derivative section reproduces ``derivative_bundle`` bit-for-bit (it is
     the same shared FD operator, just inlined into the kernel);
  2. the full fused RHS matches the verbatim Dendro-GR RHS to round-off (the 1.43x
     recompute schedule reorders fp summation).
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from mcs_common.derivatives import SpatialDerivative

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams
from bssn3d.rhs import BSSNSolver
from bssn3d.derivative_bundle import derivative_bundle, field_dict
from bssn3d.fused_backend import _deriv_lines
from bssn3d._bssn_rhs_fused import _d1, _d2
from bssn3d import initial_data as bid


def test_fused_onchip_derivs_match_bundle():
    """The emitted on-chip derivative section == derivative_bundle, bit-for-bit."""
    grid = Grid.from_domain(12, order=6)
    state = bid.gauge_wave(grid, amplitude=0.02)
    diff = SpatialDerivative(order=6)
    D_ref = derivative_bundle(state, diff, grid.dx, grid.dy, grid.dz)

    ns = {"_d1": _d1, "_d2": _d2, "dx": grid.dx, "dy": grid.dy, "dz": grid.dz}
    ns.update(field_dict(state))
    for ln in _deriv_lines():
        exec(ln, ns)

    assert set(ns) >= set(D_ref)                 # every referenced derivative is emitted
    for name, ref in D_ref.items():
        diff_max = float(np.max(np.abs(np.asarray(ns[name]) - np.asarray(ref))))
        assert diff_max == 0.0, f"{name}: on-chip deriv differs by {diff_max:.2e}"


def test_fused_rhs_matches_verbatim_gridded():
    """Full BSSNSolver A/B on a 3D gauge wave: fused (on-chip FD) == verbatim.

    Same mixed abs/rel round-off bar as the algebra-only Pallas gate — the gauge wave
    has many near-zero RHS components where the recompute's fp-summation reorder shows
    as a large relative diff on a ~1e-12 absolute one.
    """
    grid = Grid.from_domain(16, order=6)
    state = bid.gauge_wave(grid, amplitude=0.01)
    params = PhysicsParams()
    out_v = BSSNSolver(grid, params, order=6, scheme="verbatim").rhs(state)
    out_f = BSSNSolver(grid, params, order=6, scheme="fused").rhs(state)
    dv, df = np.asarray(out_v.data), np.asarray(out_f.data)
    atol, rtol = 1e-11, 1e-9
    bad = np.abs(dv - df) - (atol + rtol * np.abs(dv))
    assert np.all(bad <= 0), (
        f"gridded fused vs verbatim exceeds atol={atol:.0e}+rtol={rtol:.0e}: "
        f"max excess {float(bad.max()):.2e}, max |Δ| {float(np.abs(dv-df).max()):.2e}")


def test_fused_fp64_matches_verbatim_order8():
    """The fp64 + SMEM-trunk fused kernel (8th-order, output-fanout trunk schedule)
    == verbatim. The committed _bssn_rhs_fused_fp64.py is generated at FD order 8 (the
    production plan: 8th FD + 8th KO -> 7th-order), so the solver runs at order 8. Same
    mixed abs/rel round-off bar as the fp32 fused gate (the ~7x trunk recompute reorders
    fp summation more than the fp32 schedule's 1.4x, but stays under the bar)."""
    grid = Grid.from_domain(16, order=8)
    state = bid.gauge_wave(grid, amplitude=0.01)
    params = PhysicsParams()
    out_v = BSSNSolver(grid, params, order=8, scheme="verbatim").rhs(state)
    out_f = BSSNSolver(grid, params, order=8, scheme="fused_fp64").rhs(state)
    dv, df = np.asarray(out_v.data), np.asarray(out_f.data)
    atol, rtol = 1e-11, 1e-9
    bad = np.abs(dv - df) - (atol + rtol * np.abs(dv))
    assert np.all(bad <= 0), (
        f"gridded fused_fp64 vs verbatim exceeds atol={atol:.0e}+rtol={rtol:.0e}: "
        f"max excess {float(bad.max()):.2e}, max |Δ| {float(np.abs(dv-df).max()):.2e}")
