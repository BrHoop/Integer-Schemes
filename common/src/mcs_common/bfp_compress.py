"""Tunable-width block-floating-point (BFP) storage compression — Phase-4 emulator core.

Generalizes `bfp48.py` to an arbitrary mantissa width so the compression emulator can sweep
the accuracy <-> footprint knob (BFP48 / BFP40 / BFP32 ...). STORAGE ONLY: idle values are
parked in a compact shared-exponent integer mantissa and unpacked to fp64 to compute. All
arithmetic stays fp64 (computing *in* BFP would need int64 accumulation -> no win). The GPU
kernel (Step 4.3) will pack idle temps to SMEM/registers; this module measures the *numerics*
of that round-trip on the CPU, with no GPU.

See docs/phases/phase_4_compression/step_4.1_cpu_emulator.md and memory
`phase-4-rescope-bfp-compression`.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reference mantissa widths (bits). fp64 carries 53 significant bits (52 stored + implicit).
# FP32_MANT (24) is the precision that was ruled out for secular growth — kept as the lower
# bracket: if 24 fails the gate and 48 passes, the usable window lies between them.
BFP48, BFP40, BFP32, FP32_MANT = 48, 40, 32, 24

# Byte footprint of one stored value (mantissa only; the exponent is shared per block and
# amortized). fp64 = 8 B.
_BYTES = {48: 6, 40: 5, 32: 4, 24: 3}


def mantissa_bytes(mant_bits):
    """Per-value stored bytes for `mant_bits` (exponent amortized over the block)."""
    return _BYTES.get(mant_bits, (mant_bits + 7) // 8)


def block_exponent(arr, mant_bits):
    """Static shared power-of-two exponent for a block of same-scale values.

    Chosen so the block's largest magnitude lands near full scale of a signed
    `mant_bits`-bit integer. Returns the exponent as a float (matching bfp48's convention).
    """
    max_val = jnp.max(jnp.abs(arr))
    full = 2.0 ** (mant_bits - 1) - 1.0
    return jnp.where(max_val == 0.0, 0.0, jnp.floor(jnp.log2(full / max_val)))


class QuantizingDeriv:
    """Wrap a SpatialDerivative so every compute_d1/d2/ko OUTPUT is BFP-round-tripped.

    Models "derivative intermediates stored at `mant_bits` precision, computed in fp64" with the
    RHS/constraint code untouched (both MCS and BSSN route all derivatives through `diff_op`). Each
    derivative array gets its own block exponent (per-field-per-derivative = scale-class grouping B
    at field granularity). `mant_bits >= 53` is the exact fp64 reference (pass-through), so reference
    and compressed runs share one code path. Delegates everything else (notably `.ng`) to `inner`.
    """

    def __init__(self, inner, mant_bits):
        self.inner = inner
        self.mant_bits = mant_bits

    def _q(self, x):
        return x if self.mant_bits >= 53 else quantize(x, self.mant_bits)

    def compute_d1(self, *a, **k):
        return self._q(self.inner.compute_d1(*a, **k))

    def compute_d2(self, *a, **k):
        return self._q(self.inner.compute_d2(*a, **k))

    def compute_ko(self, *a, **k):
        return self._q(self.inner.compute_ko(*a, **k))

    def __getattr__(self, name):
        return getattr(self.inner, name)


def quantize(arr, mant_bits, exp=None):
    """Round-trip `arr` through BFP storage of `mant_bits` mantissa bits (fp64 in, fp64 out).

    Storage model: ``m = round(arr * 2**exp)`` saturated to a signed `mant_bits`-bit integer;
    reconstructed value ``= m * 2**-exp``.

    ``exp=None`` computes the ideal static exponent from THIS block's max -> the accuracy
    *ceiling* for the grouping (best case; if even this fails the gate the scheme is dead).
    Pass a precomputed ``exp`` (e.g. a per-scale-class exponent fixed from a representative
    state) to model the GPU hot loop faithfully: it cannot afford a per-call max-reduction, so
    a fixed exponent + saturation on outliers is the real on-device behavior.
    """
    if exp is None:
        exp = block_exponent(arr, mant_bits)
    lim = 2.0 ** (mant_bits - 1)
    m = jnp.clip(jnp.round(arr * (2.0 ** exp)), -lim, lim - 1.0)
    return m * (2.0 ** (-exp))
