"""Phase-4 Step 4.1 — the accuracy gate: long-run BFP-compressed MCS vs the analytic solution.

The decisive numerics question: does storing the RHS intermediates at reduced mantissa width
change the long-term result? Evolve the analytic birefringent wave for many RK4 steps at fp64
and at BFP48/40/32, and compare each to the exact solution. The gate passes if the compressed
runs track the fp64 run — i.e. the compression error stays far below the discretization
(truncation) error, so it is effectively free.

CPU-only; modest N and step count to stay inside the local box's memory.
"""
import jax
jax.config.update("jax_enable_x64", True)
import pytest

from mcs3d.compression_emulator import evolution_error_vs_analytic, default_params

N = 16
STEPS = 80


@pytest.fixture(scope="module")
def errors():
    p = default_params(N=N, order=6)
    widths = (53, 48, 40, 32, 24)
    errs = {w: evolution_error_vs_analytic(p, w, STEPS) for w in widths}
    print(f"\nMCS evolution error vs analytic (Ez interior, N={N}, {STEPS} RK4 steps):")
    for w in widths:
        tag = "fp64" if w >= 53 else f"BFP{w}"
        print(f"  {tag:6s} ({w:2d}-bit): {errs[w]:.6e}")
    return errs


def test_fp64_baseline_is_sane(errors):
    """The fp64 run has a finite, small discretization error (didn't blow up)."""
    assert 0.0 < errors[53] < 1.0


def test_bfp48_tracks_fp64(errors):
    """BFP48 storage is indistinguishable from fp64 over the run (the headline)."""
    assert errors[48] == pytest.approx(errors[53], rel=1e-3)


def test_bfp40_within_gate(errors):
    """BFP40 (5 B) stays within 2x of the fp64 error — the proposed accuracy gate."""
    assert errors[40] < 2.0 * errors[53]


def test_bfp32_within_gate(errors):
    """BFP32 (4 B, half of fp64) — measure against the gate; at this resolution the
    discretization error still dominates the per-step compression noise."""
    assert errors[32] < 2.0 * errors[53]
