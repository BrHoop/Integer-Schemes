"""Unit tests for the tunable-width BFP storage primitive (Phase 4, Step 4.1).

These pin the numerics of the compression round-trip so the emulator built on top of it is
trustworthy. CPU-only, fast.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mcs_common.bfp_compress import (
    quantize, block_exponent, mantissa_bytes, BFP48, BFP40, BFP32, FP32_MANT,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_roundtrip_error_scales_with_mantissa():
    """Relative-to-block-max error ~ 2^-mant_bits, and monotone decreasing in width."""
    x = jnp.asarray(_rng().standard_normal(4096))
    scale = float(jnp.max(jnp.abs(x)))
    errs = {}
    for mb in (FP32_MANT, BFP32, BFP40, BFP48):
        q = quantize(x, mb)
        errs[mb] = float(jnp.max(jnp.abs(q - x))) / scale
        assert errs[mb] < 2.0 ** (-(mb - 1)), (mb, errs[mb])
    assert errs[BFP48] < errs[BFP40] < errs[BFP32] < errs[FP32_MANT]


def test_fp64_width_is_near_exact():
    x = jnp.asarray(_rng(1).standard_normal(1024))
    rel = float(jnp.max(jnp.abs(quantize(x, 52) - x)) / jnp.max(jnp.abs(x)))
    assert rel < 1e-14


def test_zero_block_is_safe():
    x = jnp.zeros(16)
    assert float(jnp.max(jnp.abs(quantize(x, BFP48)))) == 0.0


def test_static_exponent_saturates_not_wraps():
    """A fixed (too-small) exponent must SATURATE an outlier, never wrap to the wrong sign --
    this is the realistic on-device failure mode the fp64 escape hatch must catch."""
    ref = jnp.asarray([0.1, 0.2, 0.15])
    exp = block_exponent(ref, BFP48)
    x = jnp.asarray([0.1, 0.2, 1.0e6])  # the 1e6 far exceeds the reference scale
    q = quantize(x, BFP48, exp=exp)
    assert float(q[-1]) > 0.0  # same sign, not wrapped
    assert float(q[-1]) <= (2.0 ** (BFP48 - 1)) * (2.0 ** (-float(exp)))


def test_block_float_small_value_precision_loss():
    """Documents the block-float failure mode that motivates per-scale-class grouping (4.2 B):
    a small value sharing a block with a large max loses relative precision."""
    x = jnp.asarray([1.0e8, 1.0e-3])  # 11 orders apart in one shared-exponent block
    q = quantize(x, BFP48)
    big_rel = float(abs(q[0] - x[0]) / x[0])
    small_rel = float(abs(q[1] - x[1]) / x[1])
    assert big_rel < 1e-12
    assert small_rel > big_rel * 1e3


def test_mantissa_bytes():
    assert mantissa_bytes(BFP48) == 6
    assert mantissa_bytes(BFP32) == 4
    assert mantissa_bytes(FP32_MANT) == 3
