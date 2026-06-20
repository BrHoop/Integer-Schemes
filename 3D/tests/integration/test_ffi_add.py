"""M1 gate: the JAX-FFI "hello world" custom call (first FFI in the repo).

The kernel runs on CUDA, so the round-trip assertion is **GPU-only** — it is skipped unless the
`cuda/ffi_add.so` is built (on the GPU host via `build_ffi.sh`) and a GPU device is present. The
registration-error path is CPU-checkable and runs everywhere.
"""

import jax
import pytest

from bssn3d import ffi_add


def _has_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def test_missing_lib_raises_clear_error():
    """Without the built .so (e.g. local CPU dev), ffi_add raises an actionable FileNotFoundError
    pointing at the build script — not a cryptic ctypes/registration failure."""
    if ffi_add._LIB.exists():
        pytest.skip("ffi_add.so is built; error-path test is for the unbuilt case")
    import jax.numpy as jnp
    ffi_add._registered = False
    with pytest.raises(FileNotFoundError, match="build_ffi.sh"):
        ffi_add.ffi_add(jnp.ones(4), jnp.ones(4))


@pytest.mark.skipif(not (ffi_add._LIB.exists() and _has_gpu()),
                    reason="needs cuda/ffi_add.so built + a GPU device")
def test_ffi_add_matches_numpy():
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    a = jnp.arange(1024, dtype=jnp.float64)
    b = 3.0 * jnp.ones(1024, dtype=jnp.float64)
    c = jax.jit(ffi_add.ffi_add)(a, b)
    c.block_until_ready()
    assert float(jnp.max(jnp.abs(c - (a + b)))) == 0.0
