"""Phase-4 Step 4.1: single-eval accuracy of the BFP compression emulator on the 3D MCS RHS.

Confirms the emulator round-trips correctly (fp64 width == exact reference) and that the
single-RHS-eval error falls as the stored mantissa width grows, with BFP48 effectively
lossless and the fp32-mantissa floor (24 bits, the ruled-out precision) clearly worse.
CPU-only.
"""
import jax
jax.config.update("jax_enable_x64", True)
import pytest

from mcs3d.compression_emulator import single_eval_rhs_error, default_params
from mcs_common.bfp_compress import BFP48, BFP40, BFP32, FP32_MANT


def test_fp64_reference_path_is_exact():
    """mant_bits>=53 is a pass-through: compressed == reference to round-off."""
    assert single_eval_rhs_error(default_params(N=12), mant_bits=53) < 1e-14


def test_single_eval_error_monotone_in_mantissa():
    p = default_params(N=12)
    errs = {mb: single_eval_rhs_error(p, mb) for mb in (FP32_MANT, BFP32, BFP40, BFP48)}
    print("\nsingle-eval RHS error vs mantissa width:")
    for mb in (FP32_MANT, BFP32, BFP40, BFP48):
        print(f"  {mb:2d}-bit: {errs[mb]:.3e}")
    assert errs[BFP48] < errs[BFP40] < errs[BFP32] < errs[FP32_MANT]


def test_bfp48_is_effectively_lossless():
    """BFP48 storage of the derivatives leaves the RHS eval near fp64 (the headline)."""
    assert single_eval_rhs_error(default_params(N=12), BFP48) < 1e-11


def test_fp32_mantissa_floor_is_visibly_worse():
    """24-bit (fp32 mantissa) — the precision ruled out for secular growth — is ~1e-6+,
    confirming the usable window lies above it."""
    assert single_eval_rhs_error(default_params(N=12), FP32_MANT) > 1e-8
