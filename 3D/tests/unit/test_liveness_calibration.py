"""Step 3.4 item-4: the CPU liveness analyzer / ptxas calibration that scores e-graph candidates.

Pins the calibration anchors (255 standalone, 276 fused), the monotonic regs->eff relation the
extraction cost function relies on, and that `score` runs on the real DAG.
"""

import pytest

from bssn3d import liveness as lv
from bssn3d.staging import build_dag


@pytest.fixture(scope="module")
def dag():
    return build_dag()


def test_calibration_reproduces_both_anchors(dag):
    calib = lv.calibrate(dag)
    alg, drv = lv.peak_live(dag)
    # exactly-determined fit -> must hit both anchors on the nose
    assert calib.predict_regs(alg, drv, held_derivs=False) == lv.ANCHOR_STANDALONE_REGS
    assert calib.predict_regs(alg, drv, held_derivs=True) == lv.ANCHOR_FUSED_REGS


def test_deriv_is_reload_discounted(dag):
    # the physical finding: derivs press registers LESS than algebra temps (reloaded from L2)
    calib = lv.calibrate(dag)
    assert calib.f_drv < calib.f_alg
    da, dd = calib.discount()
    assert 0.0 < dd < da <= 1.0          # ptxas holds <100% of either; derivs least


def test_score_monotonic_in_width(dag):
    # the extraction cost function must reward smaller peak-live: fewer regs -> higher eff
    calib = lv.calibrate(dag)
    eff = lv._eff_model(calib)
    wide = lv.score(dag, calib, eff)
    # a hypothetical narrower DAG (smaller alg peak) must score better
    regs_narrow = calib.predict_regs(300, lv.peak_live(dag)[1])
    assert regs_narrow < wide.regs
    assert eff.eff(regs_narrow) > wide.eff      # less spill -> more eff


def test_crossing_spill_cliff_is_a_win(dag):
    # dropping below the 255-reg cap (0 spill) must strictly beat the spilling baseline
    calib = lv.calibrate(dag)
    eff = lv._eff_model(calib)
    drv = lv.peak_live(dag)[1]
    spilling = calib.predict_regs(422, drv)         # baseline ~276
    no_spill = calib.predict_regs(360, drv)         # below cap
    assert spilling > lv.REG_CAP >= no_spill
    assert eff.eff(no_spill) > eff.eff(spilling)


def test_m4_baseline_scores_to_anchor(dag):
    sc = lv.score(dag)
    assert sc.regs == lv.ANCHOR_FUSED_REGS
    assert abs(sc.speedup_vs_m4 - 1.0) < 1e-9       # M4 is the anchor by construction
    assert abs(sc.eff - lv.E_M4) < 1e-6
