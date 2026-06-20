"""Item-4 (Step 3.4) — CPU liveness analyzer + ptxas calibration: the e-graph cost function.

The fusion-level experiment showed the only lever on the 3%-of-peak wall is shrinking the
register PEAK WIDTH of the algebra (no split helps — the live curve is a broad plateau). To drive
an e-graph extraction toward that, we need to *score a candidate DAG on CPU* — predict its ptxas
register count (hence spill, occupancy, speed) without a Marylou compile. This module is that
scorer, plus the calibration that ties the CPU estimate to real `ptxas -v`.

**Calibration (2 GPU anchors we already have):**
  * standalone 1c algebra: derivs are HBM leaves (not held) -> measured **255 regs**.
  * fused M4: the 138 derivs are produced on-chip and held -> measured **276 regs**.

Computing the naive peak-live on the real DAG (min-liveness schedule, the best a scheduler can do,
== ptxas's own reordering proxy) gives **alg peak 422 values, deriv peak 63 values**. Two facts
fall out:
  1. ptxas holds only ~30% of the naive peak-live in registers (255 / (422 fp64 values x 2 regs));
     it reschedules + reloads the rest -- the handoff's "ptxas is a good allocator".
  2. derivatives are reload-DISCOUNTED even harder: fused is only +21 regs over standalone despite
     +63 naive deriv-values, because in the fused kernel derivs are recomputed/reloaded from
     L2-cached field reads, not register-held.

-> a 2-parameter linear map `ptxas_regs ~= f_alg*alg_peak + f_drv*deriv_peak` is exactly determined
by the two anchors. **Caveat (stated loudly): 2 anchors / 2 params = a PERFECT fit by construction
-- it does not TEST linearity.** Validating the model (and trusting absolute predictions) needs a
GPU calibration SWEEP of >=3 variants with known peak-live; `calibration_sweep_spec()` emits that
plan. For now the scorer is trustworthy for RELATIVE ranking of e-graph candidates (the assumption
being that ptxas's discount fraction is stable across algebraically-equivalent rewrites).

Run:  python -m bssn3d.liveness
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ._codegen import parse, DENDRO_CSE
from .staging import build_dag, min_liveness_order, DAG
from .fusion_level_model import liveness_curves, EffModel, REGS_PER_FP64, REG_CAP, E_M4

# --- GPU anchors (ptxas -v) -----------------------------------------------------------------
ANCHOR_STANDALONE_REGS = 255   # 1c algebra alone; derivs = HBM leaves
ANCHOR_FUSED_REGS = 276        # M4 fused; derivs held on-chip


def peak_live(dag: DAG, order=None) -> Tuple[int, int]:
    """(alg_peak_values, deriv_peak_values) under `order` (default = min-liveness schedule).

    Peak count of simultaneously-live algebra temps / distinct derivatives. The min-liveness
    schedule is our proxy for what ptxas itself achieves (it reorders to minimise pressure), so
    this is the honest 'even after the best schedule' width an e-graph rewrite must beat.
    """
    if order is None:
        order = min_liveness_order(dag)
    statements, _, _ = parse(DENDRO_CSE)
    # liveness_curves wants the SAME statements the dag was built from; callers pass a dag built
    # from those statements. For candidate DAGs, pass a matching `order` + rebuilt statements.
    _, alg_live, deriv_live, *_ = liveness_curves(statements, dag, order)
    return int(alg_live.max()), int(deriv_live.max())


@dataclass
class PtxasCalibration:
    """Linear naive-peak-live -> ptxas-register map, fit to the two GPU anchors."""
    f_alg: float        # ptxas regs per naive peak-live ALGEBRA value
    f_drv: float        # ptxas regs per naive peak-live DERIV value (reload-discounted)
    alg_anchor: int     # the naive alg-peak the fit was computed at (for reporting)
    drv_anchor: int

    def predict_regs(self, alg_peak: int, deriv_peak: int, *, held_derivs: bool = True) -> int:
        """Predicted ptxas register count. `held_derivs=False` = derivs are HBM leaves
        (standalone-algebra deployment); True = fused (the production kernel)."""
        d = self.f_drv * deriv_peak if held_derivs else 0.0
        return int(round(self.f_alg * alg_peak + d))

    def discount(self) -> Tuple[float, float]:
        """ptxas-held FRACTION of naive peak-live, alg vs deriv (1.0 == every value held)."""
        return self.f_alg / REGS_PER_FP64, self.f_drv / REGS_PER_FP64


def calibrate(dag: DAG | None = None) -> PtxasCalibration:
    if dag is None:
        dag = build_dag()
    alg, drv = peak_live(dag)
    # standalone (derivs = leaves): f_alg * alg = 255
    f_alg = ANCHOR_STANDALONE_REGS / alg
    # fused: f_alg*alg + f_drv*drv = 276  ->  f_drv = (276 - 255) / drv
    f_drv = (ANCHOR_FUSED_REGS - ANCHOR_STANDALONE_REGS) / drv
    return PtxasCalibration(f_alg=f_alg, f_drv=f_drv, alg_anchor=alg, drv_anchor=drv)


def _eff_model(calib: PtxasCalibration, w0: float = 16.0, spill_alpha: float = 0.05) -> EffModel:
    """Throughput model that consumes CALIBRATED ptxas regs directly (ptxas_factor=1), anchored
    so the fused M4 (276 regs) maps to 3.3% of FP64 peak."""
    m = EffModel(w0=w0, spill_alpha=spill_alpha, ptxas_factor=1.0, norm=1.0)
    m.norm = E_M4 / m.eff(ANCHOR_FUSED_REGS)
    return m


@dataclass
class Score:
    regs: int
    spill_vals: float
    eff: float          # fraction of FP64 peak
    speedup_vs_m4: float


def score(dag: DAG | None = None, calib: PtxasCalibration | None = None,
          eff: EffModel | None = None, *, held_derivs: bool = True) -> Score:
    """Score a (candidate) fused-RHS DAG: predicted ptxas regs, spill, eff, and speedup vs the
    current M4 baseline. THIS IS THE E-GRAPH EXTRACTION COST FUNCTION (minimise regs -> maximise
    eff). Lower regs -> less spill -> more occupancy -> higher eff."""
    if dag is None:
        dag = build_dag()
    if calib is None:
        calib = calibrate(dag)
    if eff is None:
        eff = _eff_model(calib)
    alg, drv = peak_live(dag)
    regs = calib.predict_regs(alg, drv, held_derivs=held_derivs)
    spill = max(0.0, regs - REG_CAP) / REGS_PER_FP64
    e = eff.eff(regs)
    return Score(regs=regs, spill_vals=spill, eff=e, speedup_vs_m4=e / E_M4)


def calibration_sweep_spec() -> str:
    """The GPU task that VALIDATES the (currently exactly-determined) calibration: emit >=3 kernel
    variants of known naive peak-live, compile, read `ptxas -v`, check the linear map holds."""
    return (
        "GPU calibration sweep (one Marylou push) to validate the 2-anchor fit:\n"
        "  1. Emit 4-5 fused-kernel variants with DIFFERENT known naive alg-peak. Cheapest source:\n"
        "     the existing staged/reassoc variants (`staging.generate_staged`, "
        "`predict_reassociation`)\n"
        "     and a maxrregcount-uncapped build of each -> each has a CPU-computed peak_live().\n"
        "  2. Compile with `-Xptxas -v`; record (naive_alg_peak, naive_drv_peak, ptxas_regs, spill)\n"
        "     AND wall-clock time/eval (the regs->speed conversion, NOT yet calibrated).\n"
        "  3a. Linear-fit ptxas_regs ~ f_alg*alg + f_drv*drv; check residuals + that f_alg,f_drv\n"
        "      match the 2-anchor values (0.60 / 0.33 regs/value). Non-linear -> richer reg model.\n"
        "  3b. Fit the EffModel (w0, spill_alpha) to time-vs-regs -- pins the headline 'off-spill\n"
        "      ~1.5x', currently an UNCALIBRATED guess (alpha=0.05).\n"
        "  -> until then: trust the REG predictions (calibrated); treat eff/speedup as indicative."
    )


def main() -> None:
    dag = build_dag()
    calib = calibrate(dag)
    alg, drv = peak_live(dag)
    da, dd = calib.discount()
    eff = _eff_model(calib)
    sc = score(dag, calib, eff)

    print(">> BSSN RHS liveness analyzer + ptxas calibration (Step 3.4 item-4)")
    print(f"   naive peak-live (min-liveness schedule): alg={alg} values, deriv={drv} values")
    print(f"   calibration fit to anchors [standalone {ANCHOR_STANDALONE_REGS}, "
          f"fused {ANCHOR_FUSED_REGS}]:")
    print(f"     f_alg={calib.f_alg:.3f} regs/value  (ptxas holds {da*100:.0f}% of naive alg peak)")
    print(f"     f_drv={calib.f_drv:.3f} regs/value  (ptxas holds {dd*100:.0f}% of naive deriv peak"
          f" -> reload-discounted)")
    print(f"   self-check: predict standalone={calib.predict_regs(alg, drv, held_derivs=False)} "
          f"(meas 255), fused={calib.predict_regs(alg, drv)} (meas 276)")
    print()
    print(f"   M4 baseline score: {sc.regs} regs, spill {sc.spill_vals:.0f} fp64 vals, "
          f"eff {sc.eff*100:.2f}%, speedup-vs-M4 {sc.speedup_vs_m4:.2f}x (==1 by anchor)")
    print()
    print("   what an e-graph that cuts the alg peak-live would buy (holding deriv peak):")
    print("   [regs = CALIBRATED to anchors; eff/vs-M4 SUPPORTED by Stage-A sweep (255 regs fastest,")
    print("    spill ~0.13-0.17 ms/KB, off-spill ~1.5-1.7x) -- exact floor awaits a reduced-width DAG]")
    print(f"   {'alg peak':>9} {'ptxas regs':>11} {'spill vals':>11} {'eff %':>7} {'vs M4':>7}")
    for target in (alg, 380, 340, 300, 255, 210):
        regs = calib.predict_regs(target, drv)
        e = eff.eff(regs)
        sp = max(0.0, regs - REG_CAP) / REGS_PER_FP64
        tag = "  (0-spill)" if regs <= REG_CAP else ""
        print(f"   {target:>9} {regs:>11} {sp:>11.0f} {e*100:>7.2f} {e/E_M4:>6.2f}x{tag}")
    print()
    print(calibration_sweep_spec())


if __name__ == "__main__":
    main()
