import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def pack(arr):
    """Pack a float64 array to BFP48: one shared exponent + int16 mantissa triplets.

    Args:
        arr: float64 array of any shape (all elements share one exponent)
    Returns:
        mantissas: same shape as arr with a trailing dim-3, dtype int16
        exponent:  scalar int32
    """
    max_val = jnp.max(jnp.abs(arr))
    scale_exp = jnp.where(max_val == 0, 0.0, jnp.floor(jnp.log2((2.0**47 - 1) / max_val)))
    exponent = scale_exp.astype(jnp.int32)
    scaled_int = jnp.floor(arr * (2.0 ** scale_exp) + 0.5).astype(jnp.int64)

    p0 = (scaled_int & 0xFFFF).astype(jnp.int16)
    p1 = ((scaled_int >> 16) & 0xFFFF).astype(jnp.int16)
    p2 = ((scaled_int >> 32) & 0xFFFF).astype(jnp.int16)
    return jnp.stack([p0, p1, p2], axis=-1), exponent


def unpack(mantissas, exponent):
    """Unpack BFP48 to float64.

    Args:
        mantissas: int16 array, last dim is 3 (the three 16-bit limbs)
        exponent:  scalar int32
    Returns:
        float64 array with the trailing dim-3 collapsed
    """
    p0 = mantissas[..., 0].astype(jnp.int64) & 0xFFFF
    p1 = mantissas[..., 1].astype(jnp.int64) & 0xFFFF
    p2 = mantissas[..., 2].astype(jnp.int64) & 0xFFFF
    combined = p0 | (p1 << 16) | (p2 << 32)
    combined = jnp.where(combined >= (1 << 47), combined - (1 << 48), combined)
    return combined * (2.0 ** (-exponent.astype(jnp.float64)))


def unpack_to_int64(mantissas):
    """Decode BFP48 mantissas to signed int64 without any float conversion.

    Used inside the fused Pallas kernel: the int64 mantissas feed directly
    into Ozaki RNS modular reduction, bypassing the float scaling step that
    would otherwise be required.

    Args:
        mantissas: int16 array, last dim is 3
    Returns:
        int64 array with the trailing dim-3 collapsed
    """
    p0 = mantissas[..., 0].astype(jnp.int64) & 0xFFFF
    p1 = mantissas[..., 1].astype(jnp.int64) & 0xFFFF
    p2 = mantissas[..., 2].astype(jnp.int64) & 0xFFFF
    combined = p0 | (p1 << 16) | (p2 << 32)
    return jnp.where(combined >= (1 << 47), combined - (1 << 48), combined)
