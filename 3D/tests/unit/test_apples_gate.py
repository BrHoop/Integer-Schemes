"""Step 3.4 item-1: guard the apples-gate COMPARISON LOGIC (not the physics — that's the heavy
GPU run). `run` is monkeypatched with synthetic constraint trajectories so this is <5s, no XLA
compile. Pins: identical -> PASS, beyond-tol drift -> FAIL, blow-up-point mismatch -> FAIL.
"""

import pytest

from bssn3d import apples_gate as ag


def _rows(H, M=1e-6, a=1.0, dev=1e-7, finite=True, n=4):
    return [dict(H=H, M=M, max_alpha=a, max_dev=dev, finite=finite) for _ in range(n)]


def _patch(monkeypatch, ref_rows, cand_rows):
    def fake_run(id_name, **kw):
        return ref_rows if kw.get("scheme") == "verbatim" else cand_rows
    monkeypatch.setattr(ag, "run", fake_run)


def test_rel_is_floored():
    # near-zero reference must not blow the ratio: |0-0|/(0+floor) == 0
    assert ag._rel(0.0, 0.0, 1e-12) == 0.0
    assert ag._rel(1.0 + 1e-9, 1.0, 1e-12) == pytest.approx(1e-9, rel=1e-3)


def test_identical_trajectories_pass(monkeypatch):
    rows = _rows(7.1e-6)
    _patch(monkeypatch, rows, rows)
    assert ag.gate("candidate", tol=1e-6, quick=True) is True


def test_roundoff_drift_within_tol_passes(monkeypatch):
    _patch(monkeypatch, _rows(7.1e-6), _rows(7.1e-6 * (1 + 1e-9)))
    assert ag.gate("candidate", tol=1e-6, quick=True) is True


def test_drift_beyond_tol_fails(monkeypatch):
    # 1e-3 relative drift >> 1e-6 tol -> FAIL
    _patch(monkeypatch, _rows(7.1e-6), _rows(7.1e-6 * (1 + 1e-3)))
    assert ag.gate("candidate", tol=1e-6, quick=True) is False


def test_blowup_point_mismatch_fails(monkeypatch):
    # candidate diverges (fewer samples) -> verdict mismatch -> FAIL even if early samples agree
    _patch(monkeypatch, _rows(7.1e-6, n=8), _rows(7.1e-6, n=4))
    assert ag.gate("candidate", tol=1e-6, quick=True) is False


def test_finiteness_disagreement_fails(monkeypatch):
    ref = _rows(7.1e-6, n=4)
    cand = _rows(7.1e-6, n=4)
    cand[-1]["finite"] = False                 # candidate went non-finite where ref stayed finite
    _patch(monkeypatch, ref, cand)
    assert ag.gate("candidate", tol=1e-6, quick=True) is False
