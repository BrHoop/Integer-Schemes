"""MSRK integrator assessment tests (Sanches et al. arXiv:2603.05763).

Locks in the correctness of the three multistep RK schemes on 2D MCS:
  * coefficients + companion-matrix stability reproduce the paper's published
    ASR imaginary-axis intercepts (Table 1),
  * each method is 4th-order in time (RK4 startup preserves the order),
  * the effective-CFL ordering matches the paper's finding that RK4-2 beats RK4
    near the stability limit while RK4-3 does not.
"""
import numpy as np
import pytest

from mcs2d import msrk
from mcs2d.msrk_analysis import convergence_order, max_stable_cfl
from mcs2d.main import load_parameters
from pathlib import Path

_PARAMS = str(Path(__file__).resolve().parents[2] / "params.toml")

PAPER_INTERCEPT = {"rk4": np.sqrt(8.0), "rk4_2_1": 2.53865,
                   "rk4_2_2": 2.46201, "rk4_3": 1.30711}


@pytest.mark.parametrize("method", ["rk4", "rk4_2_1", "rk4_2_2", "rk4_3"])
def test_asr_intercept_matches_paper(method):
    """Coefficients + companion build reproduce Table 1 intercepts to 1e-4."""
    ic = msrk.imag_axis_intercept(method)
    assert abs(ic - PAPER_INTERCEPT[method]) < 1e-4, \
        f"{method}: intercept {ic:.5f} != paper {PAPER_INTERCEPT[method]:.5f}"


def test_coefficient_consistency():
    """sum(b)=1 and c_i = sum_j a_ij held in exact arithmetic at import."""
    for name, m in msrk.METHODS.items():
        assert abs(sum(m.b) - 1.0) < 1e-12, f"{name}: sum(b) != 1"


@pytest.mark.parametrize("method", ["rk4_2_1", "rk4_2_2", "rk4_3"])
def test_fourth_order_in_time(method):
    """Self-convergence in dt at fixed grid → integrator order ~4."""
    order = convergence_order(method, nx=48, t_phys=0.5, base_steps=160)
    assert 3.7 < order < 4.3, f"{method}: temporal order {order:.2f} not ~4"


def test_ecf_ordering():
    """Near the stability limit RK4-2 variants beat RK4; RK4-3 does not.

    ECF = CFL_max / n_stages.  Confirms the paper's headline regime result on
    the MCS spectrum: stage-count reduction only pays off if stability allows
    the larger step — RK4-3's small ASR makes it the least efficient here.
    """
    params = load_parameters(_PARAMS)
    nx = params.get("Nx", 512)
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / nx
    # shared coarse spectrum (fast) — ordering is robust to n_k
    from mcs2d.msrk_analysis import _semidiscrete_eigs
    eigs = _semidiscrete_eigs(params, dx, dy, n_k=48)
    ecf = {k: max_stable_cfl(k, params, dx, dy, eigs=eigs) / msrk.STAGES[k]
           for k in ["rk4", "rk4_2_1", "rk4_2_2", "rk4_3"]}
    assert ecf["rk4_2_2"] > ecf["rk4"], "RK4-2(2) should beat RK4 on ECF"
    assert ecf["rk4_2_1"] > ecf["rk4"], "RK4-2(1) should beat RK4 on ECF"
    assert ecf["rk4_3"] < ecf["rk4"], "RK4-3 should be worse than RK4 on ECF"
