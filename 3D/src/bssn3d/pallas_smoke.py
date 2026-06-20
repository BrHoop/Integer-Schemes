"""Minimal Pallas/Triton GPU compile smoke test.

Isolates "does Triton compile ANYTHING on this GPU" from "is our BSSN kernel too big".
Uses NONE of the BSSN code — just the bare Pallas primitives. Run on the GPU node:

    python -m bssn3d.pallas_smoke

Reading the result:
  * Rung 1 (trivial add) hangs/fails  → Triton/jaxlib/CUDA install is broken on this GPU.
    Reinstall (jax[cuda12]==0.9.2) and re-run; the kernel is not the problem.
  * Rung 1 ok, Rung 2 (broadcast-multiply-reduce) hangs → the deriv PRIMITIVE is the
    issue on this Triton/GPU, not kernel size.
  * Both fast → Triton compiles fine here; the BSSN compile problem is kernel SIZE, and
    we attack that (capped sweep / split / cheaper deriv expression).
"""

import os
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

if not os.environ.get("JAX_PALLAS_USE_MOSAIC_GPU"):
    jax.config.update("jax_pallas_use_mosaic_gpu", False)

print(f"jax {jax.__version__} | backend {jax.default_backend()} | devices {jax.devices()}",
      flush=True)


def _time(label, fn, *args):
    t0 = time.time()
    out = fn(*args)
    out.block_until_ready()
    dt = time.time() - t0
    print(f"  {label:<34} compiled+ran in {dt:7.2f}s   (checksum {float(jnp.sum(out)):.4e})",
          flush=True)
    return out


# -------- Rung 1: trivial elementwise add (does Triton compile at all?) --------
def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def pallas_add(x, y):
    return pl.pallas_call(
        _add_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x, y)


# -------- Rung 2: broadcast-multiply-reduce (the exact deriv primitive) --------
# Contract (8,16) matrix M with axis 0 of a (16,16) tile -> (8,16): the same
# expand_dims * reshape * sum the tiled derivative kernel uses (no dot/slice).
def _bmr_kernel(m_ref, x_ref, o_ref):
    m = m_ref[...]                                   # (8, 16)
    xe = jnp.expand_dims(x_ref[...], 0)              # (1, 16, 16)
    o_ref[...] = (xe * m.reshape(8, 16, 1)).sum(axis=1)


def pallas_bmr(m, x):
    return pl.pallas_call(
        _bmr_kernel, out_shape=jax.ShapeDtypeStruct((8, 16), x.dtype))(m, x)


def main():
    n = 256
    print("Rung 1 — trivial Pallas add (Triton-compiles-at-all):", flush=True)
    a = jnp.arange(n, dtype=jnp.float32)
    _time("pallas add (256 f32)", pallas_add, a, a)

    print("Rung 2 — broadcast-multiply-reduce (deriv primitive):", flush=True)
    m = jnp.ones((8, 16), dtype=jnp.float32)
    x = jnp.arange(16 * 16, dtype=jnp.float32).reshape(16, 16)
    _time("bmr (8,16)x(16,16)", pallas_bmr, m, x)

    print(">> Both fast → Triton compiles fine on this GPU; the BSSN issue is kernel SIZE.\n"
          ">> Rung 1 hangs → install/toolchain broken (reinstall jax[cuda12]==0.9.2).",
          flush=True)


if __name__ == "__main__":
    main()
