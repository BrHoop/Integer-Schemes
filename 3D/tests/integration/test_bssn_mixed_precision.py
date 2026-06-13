"""Phase 3.2c accuracy/constraint gate for the mixed-precision RHS.

The register-fit lever (fp32 for the wide Ricci-contraction working set so it fits
the 255-register file) is only worth a GPU spill measurement if it doesn't wreck the
physics. ``precision="fp32_contraction"`` keeps the FD bundle in fp64 (the
cancellation-sensitive second derivatives) and the RK4 accumulation in fp64, dropping
to fp32 only for the pointwise algebra. These tests measure the cost:

  * single-eval RHS error vs the fp64 verbatim algebra (~fp32 epsilon, ~1e-6);
  * constraint violation under a gauge-wave evolution matches fp64 (no secular growth);
  * robust stability (Minkowski + noise) stays bounded, no fp32-noise-driven blow-up.

Measured 2026-06-12: single-eval ~1e-6, gauge-wave H/M ratios 1.000 (state dev ~5e-8
over 40 steps), robust stability 4.16e-7 vs fp64 4.11e-7. The fp32 contraction is benign.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from bssn3d import oracle
from bssn3d._bssn_rhs_generated import bssn_rhs_algebra as V
from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams
from bssn3d.evolve import BSSNEvolution
from bssn3d.constraints import ConstraintSolver
from bssn3d import initial_data as bid


def _call(vals, dt):
    F = {k: jnp.asarray([x], dtype=dt) for k, x in vals["fields"].items()}
    D = {k: jnp.asarray([x], dtype=dt) for k, x in vals["derivs"].items()}
    o = V(F, D, vals["eta"], vals["lmbda"], vals["lambda_f"], vals["BSSN_CAHD_C"],
          vals["dt"], vals["dx_i"], vals["h_ssl"], vals["sig_ssl"], vals["t"])
    return {k: float(np.asarray(v)[0]) for k, v in o.items()}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fp32_contraction_rhs_accuracy(seed):
    """One RHS eval in fp32 vs fp64: ~fp32 epsilon, comfortably < 1e-4."""
    vals = oracle.random_inputs(seed)
    f64, f32 = _call(vals, jnp.float64), _call(vals, jnp.float32)
    worst = max(abs(f64[k] - f32[k]) / max(abs(f64[k]), 1e-30) for k in f64)
    assert worst < 1e-4, f"seed {seed}: fp32 RHS error {worst:.2e}"


# Ricci-heavy outputs — where the parallel chat feared fp32 catastrophic cancellation
# in the large-term conformal-Ricci differences (`docs/algebra.md` §2).
_RICCI_HEAVY = {"At0", "At1", "At2", "At3", "At4", "At5", "K"}


@pytest.mark.parametrize("dmult", [10.0, 40.0, 100.0])   # × default deriv magnitude (0.05)
def test_fp32_strongfield_no_ricci_cancellation(dmult):
    """Strong-curvature stress: scale derivative inputs up to 100× and metric off-diagonals
    5×, and confirm the fp32 algebra error stays BOUNDED — i.e. NO catastrophic-cancellation
    blow-up on the Ricci-heavy outputs (would run to ~1e-2+ if Ricci lost many digits).

    Measured 2026-06-13: worst-all ~9e-5 at 100×, Ricci-heavy not dramatically worse than
    the first-order outputs → the fp32 decline (docs/algebra.md §2) looks over-cautious. This
    guard catches a future catastrophic regression. CAVEAT: single-point, INDEPENDENT random
    derivatives — does not manufacture the correlated cancellation a real puncture might, nor
    test secular growth over long runs (both still open, gated on puncture ID + a long run).
    """
    worst_all = worst_ricci = 0.0
    for seed in range(8):
        vals = oracle.random_inputs(seed)
        vals["derivs"] = {k: x * dmult for k, x in vals["derivs"].items()}
        for k in ("gt1", "gt2", "gt4"):
            vals["fields"][k] *= 5.0
        f64, f32 = _call(vals, jnp.float64), _call(vals, jnp.float32)
        for fld in f64:
            rel = abs(f64[fld] - f32[fld]) / max(abs(f64[fld]), 1e-30)
            worst_all = max(worst_all, rel)
            if fld in _RICCI_HEAVY:
                worst_ricci = max(worst_ricci, rel)
    # 1e-3 = ~10× headroom over the measured ~1e-4; a true cancellation blow-up is ~1e-2+.
    assert worst_all < 1e-3, f"dmult={dmult}: fp32 strong-field error {worst_all:.2e}"
    assert worst_ricci < 1e-3, f"dmult={dmult}: fp32 Ricci-heavy error {worst_ricci:.2e}"


@pytest.mark.slow
def test_fp32_contraction_preserves_constraints():
    """Gauge-wave evolution: fp32-contraction constraints match fp64 (no growth)."""
    g = Grid.from_domain(16, order=6)
    cs = ConstraintSolver(g, order=6)
    dt = 0.25 * g.dx

    def run(prec):
        ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=0.1,
                           bc="periodic", precision=prec)
        s = ev.evolve(bid.gauge_wave(g, amplitude=0.01), dt=dt, nsteps=40)
        H, M = cs.l2(s)
        return s, float(H), float(M)

    s64, H64, M64 = run("fp64")
    s32, H32, M32 = run("fp32_contraction")
    assert bool(jnp.all(jnp.isfinite(s32.data)))
    # constraints track fp64 to a few percent (here: identical to 3 digits)
    assert abs(H32 / H64 - 1.0) < 0.05 and abs(M32 / M64 - 1.0) < 0.05
    # the fp32 perturbation of the evolved state stays tiny (FD + accum are fp64)
    assert float(jnp.max(jnp.abs(s64.data - s32.data))) < 1e-5


@pytest.mark.slow
def test_fp32_contraction_robust_stability():
    """Minkowski + 1e-8 noise stays bounded under fp32-contraction (no fp32 blow-up)."""
    g = Grid.from_domain(12, order=6)
    dt = 0.25 * g.dx
    mink = bid.minkowski(g).data

    def dev(prec):
        ev = BSSNEvolution(g, PhysicsParams(), order=6, ko_sigma=0.1,
                           bc="periodic", precision=prec)
        s = ev.evolve(bid.robust_stability(g, amp=1e-8, seed=1), dt=dt, nsteps=40)
        return float(jnp.max(jnp.abs(s.data - mink))), bool(jnp.all(jnp.isfinite(s.data)))

    d32, fin = dev("fp32_contraction")
    assert fin and d32 < 1e-4, f"fp32 robust stability deviation {d32:.2e}"
