"""Guard the Step 3.2f Increment-1b reference math (ricci_ref.py).

The CUDA warp-cooperative Ricci kernel is validated against this reference, so the
reference must be correct. The decisive check is against an INDEPENDENT SymPy symbolic
Ricci (symbolic differentiation vs the closed-form chain-rule expansion). SymPy is a
dev-only oracle, so the test skips cleanly if it is not installed.
"""

import numpy as np
import pytest

from bssn3d.cuda import ricci_ref as rr


def test_ricci_symmetric_and_flat():
    # flat metric (identity, zero derivatives) -> Ricci = 0
    gt = np.array([1.0, 0, 0, 1.0, 0, 1.0])
    assert np.abs(rr.ricci(gt, np.zeros(18), np.zeros(36))).max() < 1e-14


def test_ricci_vs_sympy():
    pytest.importorskip("sympy")
    # closed-form pointwise Ricci == SymPy symbolic Ricci of an analytic metric
    assert rr._sympy_check(seed=5, n=4) < 1e-9


def test_field_jet_spd():
    # generated test points are SPD conformal metrics with consistent 2-jets
    rng = np.random.default_rng(9)
    for _ in range(50):
        gt, _, _ = rr._field_jet(rng)
        assert np.linalg.eigvalsh(rr._mat(gt)).min() > 0.0
