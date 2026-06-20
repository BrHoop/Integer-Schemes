"""Phase-4 — BSSN accuracy gate for BFP storage compression (does BFP32 hold on the REAL RHS?).

Fast guards (single-eval + constraint perturbation on strong-field Gowdy). The full evolution
constraint-growth check is in `bssn3d.compression_accuracy.evolution_constraint_growth` (run
manually; too slow for CI). Measured 2026-06-19: BFP48 lossless, BFP32 ~7e-10 RHS / ~7e-8 momentum-
constraint perturbation, and a 60-step Gowdy run tracks fp64 to 6-7 digits with NO accelerated
constraint growth.
"""
import jax
jax.config.update("jax_enable_x64", True)

from bssn3d.compression_accuracy import single_eval_rhs_error, constraint_perturbation
from mcs_common.bfp_compress import BFP48, BFP32, FP32_MANT

N = 16


def test_fp64_reference_is_exact():
    assert single_eval_rhs_error(N, 53) == 0.0


def test_bfp48_lossless_on_bssn_rhs():
    assert single_eval_rhs_error(N, BFP48) < 1e-13


def test_single_eval_monotone():
    e48 = single_eval_rhs_error(N, BFP48)
    e32 = single_eval_rhs_error(N, BFP32)
    e24 = single_eval_rhs_error(N, FP32_MANT)
    assert e48 < e32 < e24


def test_bfp32_constraint_perturbation_small():
    """BFP32 storage of the constraint derivatives perturbs (H, |M|) far below the truncation-level
    baseline — the cancellation-sensitivity check on the under-protected momentum channel."""
    H0, M0, dH, dM = constraint_perturbation(N, BFP32)
    assert dH < 1e-7 and dM < 1e-6        # measured ~2.6e-9 / 7.4e-8


def test_bfp24_visibly_worse_than_bfp32():
    _, _, dH32, dM32 = constraint_perturbation(N, BFP32)
    _, _, dH24, dM24 = constraint_perturbation(N, FP32_MANT)
    assert dH24 > dH32 * 100              # 24-bit (fp32 mantissa) clearly degrades
