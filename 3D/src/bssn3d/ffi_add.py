"""M1 — Python side of the JAX-FFI "hello world" (the first FFI in the repo).

Loads the `cuda/ffi_add.so` shared library, registers its `extern "C"` handler `Add` as an XLA
custom-call target, and exposes `ffi_add(a, b) == a + b` computed on the GPU via `jax.ffi`. This
de-risks the FFI plumbing in isolation before Step 3.2e's 2.5D derivative kernel rides the same
path (build → register → `ffi_call`).

GPU-only: the `.so` is built on the GPU host (`cuda/build_ffi.sh`) and the custom call runs on
CUDA. Locally (CPU) there is no `.so` and no device — `_register()` raises a clear message. Run
the self-check on the GPU host with `python -m bssn3d.ffi_add`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import jax
import jax.ffi

_LIB = Path(__file__).resolve().parent / "cuda" / "ffi_add.so"
_TARGET = "bssn_ffi_add"
_registered = False


def _register() -> None:
    """Load the shared lib and register its handler as a CUDA FFI target (idempotent)."""
    global _registered
    if _registered:
        return
    if not _LIB.exists():
        raise FileNotFoundError(
            f"{_LIB} not built. On the GPU host run: bash {_LIB.parent}/build_ffi.sh")
    lib = ctypes.cdll.LoadLibrary(str(_LIB))
    jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(lib.Add), platform="CUDA")
    _registered = True


def ffi_add(a: jax.Array, b: jax.Array) -> jax.Array:
    """``a + b`` computed by the CUDA FFI kernel. Same shape/dtype (fp64) in and out."""
    _register()
    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
    return jax.ffi.ffi_call(_TARGET, out_type)(a, b)


def _selfcheck() -> int:
    """Run the FFI kernel under jit and compare to a + b. Returns a process exit code."""
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    a = jnp.arange(1024, dtype=jnp.float64)
    b = 3.0 * jnp.ones(1024, dtype=jnp.float64)
    c = jax.jit(ffi_add)(a, b)
    c.block_until_ready()
    err = float(jnp.max(jnp.abs(c - (a + b))))
    ok = err == 0.0
    print(f"[M1 ffi_add] platform={jax.devices()[0].platform}  n={a.size}  "
          f"max|ffi - (a+b)| = {err:.3e}  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
