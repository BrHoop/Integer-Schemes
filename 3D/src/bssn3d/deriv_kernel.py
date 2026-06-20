"""M2a — Python side of the device derivative stage (the FFI wrapper + the validation gate).

`device_derivative_bundle(state, dx, dy, dz)` returns the same ``{grad_*: array}`` dict as
`derivative_bundle`, but computed on the GPU by `cuda/deriv_2p5d.so` (built from `deriv_2p5d.cu`).
This is the correctness baseline for Step 3.2e: it validates the device derivative math + the
FFI 24-in/138-out workload against the CPU `derivative_bundle` (= the M0 reference). The SMEM 2.5D
streaming (wall A) is M2b; it will reuse this wrapper's name mapping and validation.

GPU-only: the `.so` is built on the GPU host (`cuda/build_deriv.sh`). Run the self-check there with
`python -m bssn3d.deriv_kernel`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict

import jax
# MUST precede the imports below: SpatialDerivative bakes its stencil arrays as jnp.array at
# class-definition (import) time, so x64 has to be on BEFORE mcs_common.derivatives is imported
# (transitively via gen_deriv_kernel / derivative_bundle) — else the FD coefficients are fp32 and
# the derivative_bundle reference is silently degraded to ~1e-7 (the M2a "FAIL" was this, not the
# kernel). Matches the gen_algebra_cuda.py convention.
jax.config.update("jax_enable_x64", True)
import jax.ffi

from .cuda.gen_deriv_kernel import deriv_order
from ._bssn_rhs_generated import FIELD_INPUTS
from .derivative_bundle import field_dict

_CUDA = Path(__file__).resolve().parent / "cuda"
# variant -> (shared lib, FFI target name, extern-"C" handler symbol)
_VARIANTS = {
    "global": ("deriv_2p5d.so", "bssn_deriv", "Deriv"),          # M2a (1 thread/pt, global mem)
    "smem":   ("deriv_smem.so", "bssn_deriv_smem", "DerivSmem"),  # M2b (2.5D SMEM streaming)
}
_LIB = _CUDA / _VARIANTS["global"][0]   # back-compat: tests reference _LIB for the M2a lib
_registered: set = set()
_NAMES = [s[0] for s in deriv_order()]          # the 138 deriv names in kernel/out order
N_DERIV = len(_NAMES)


def _register(variant: str) -> None:
    if variant in _registered:
        return
    libname, target, sym = _VARIANTS[variant]
    lib_path = _CUDA / libname
    if not lib_path.exists():
        raise FileNotFoundError(
            f"{lib_path} not built. On the GPU host: bash {_CUDA}/build_deriv.sh")
    lib = ctypes.cdll.LoadLibrary(str(lib_path))
    jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
    _registered.add(variant)


def device_derivative_bundle(state, dx: float, dy: float, dz: float,
                             variant: str = "global") -> Dict[str, jax.Array]:
    """The 138 derivatives computed on the GPU; same keys/shapes as ``derivative_bundle``.

    ``variant``: "global" = M2a (1 thread/point, global-memory neighbours);
    "smem" = M2b (2.5D SMEM-streaming, the wall-A geometry). Both share the deriv-name mapping."""
    _register(variant)
    target = _VARIANTS[variant][1]
    F = field_dict(state)
    import jax.numpy as jnp
    fields = jnp.stack([F[name] for name in FIELD_INPUTS])      # (24, Sx, Sy, Sz)
    out_type = jax.ShapeDtypeStruct((N_DERIV,) + fields.shape[1:], fields.dtype)
    out = jax.ffi.ffi_call(target, out_type)(
        fields, dx=float(dx), dy=float(dy), dz=float(dz))
    return {name: out[d] for d, name in enumerate(_NAMES)}


def _selfcheck() -> int:
    """Compare the device bundle to the CPU derivative_bundle on a random state (round-off bar)."""
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import jax.numpy as jnp
    from .grid import Grid
    from .state import BSSNState
    from .derivative_bundle import derivative_bundle
    from mcs_common.derivatives import SpatialDerivative

    g = Grid.from_domain(16, order=6)                 # ng=4 (8th-order KO), 6th-order FD
    diff = SpatialDerivative(order=6, ko_order=8)
    assert diff.C1.dtype == jnp.float64, (
        f"FD stencils are {diff.C1.dtype}, not fp64 — x64 was enabled too late (after "
        "mcs_common.derivatives was imported); the bundle reference would be fp32-degraded.")
    rng = np.random.default_rng(11)
    state = BSSNState(jnp.array(rng.standard_normal((24,) + g.shape)))

    ref = {k: jnp.asarray(v) for k, v in
           derivative_bundle(state, diff, g.dx, g.dy, g.dz).items()}

    def _worst(bundle):
        w, where = 0.0, ""
        for k in ref:
            e = float(jnp.max(jnp.abs(ref[k] - bundle[k])))
            if e > w:
                w, where = e, k
        return w, where

    rc = 0
    avail = [v for v in _VARIANTS if (_CUDA / _VARIANTS[v][0]).exists()]
    print(f"platform={jax.devices()[0].platform}  grid={g.shape}  {N_DERIV} derivs")
    results = {}
    for v in avail:
        dev = jax.jit(lambda s, vv=v: device_derivative_bundle(BSSNState(s), g.dx, g.dy, g.dz, vv))(state.data)
        results[v] = dev
        w, where = _worst(dev)
        tag = "M2a global" if v == "global" else "M2b smem"
        ok = w <= 1e-11
        rc |= 0 if ok else 1
        print(f"  [{tag:11s}] max|device - bundle| = {w:.3e} (at {where})  -> {'PASS' if ok else 'FAIL'}")
    if "global" in results and "smem" in results:           # M2b must match M2a
        cross = max(float(jnp.max(jnp.abs(results['smem'][k] - results['global'][k]))) for k in ref)
        print(f"  [M2b vs M2a ] max|smem - global| = {cross:.3e}  -> {'PASS' if cross <= 1e-11 else 'FAIL'}")
        rc |= 0 if cross <= 1e-11 else 1
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
