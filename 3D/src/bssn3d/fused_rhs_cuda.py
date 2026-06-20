"""M4 — Python side of the fused CUDA BSSN RHS (the thesis target): validate + wall-clock A/B.

`device_rhs_fused` calls the fused kernel (`cuda/rhs_fused.so`): per point it computes the 138
derivatives into registers and runs the 1c algebra in ONE kernel, no derivative HBM round-trip.
Validated vs `BSSNSolver.rhs` (verbatim) to round-off; the gate is `--compare` wall-clock vs the
verbatim-XLA RHS (the 31.27 ms @128³ baseline). x64 set at module top BEFORE the FD-stencil import
(the fp32-stencil trap — see memory `x64-stencil-import-order`).

  python -m bssn3d.fused_rhs_cuda                 # validate vs verbatim (round-off)
  python -m bssn3d.fused_rhs_cuda --compare --n 128   # wall-clock M4 vs verbatim XLA
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time
from pathlib import Path
from typing import Dict

import jax
jax.config.update("jax_enable_x64", True)        # BEFORE the FD-stencil import below
import jax.ffi
import jax.numpy as jnp

from .state import BSSNState, PhysicsParams, VAR_NAMES
from .grid import Grid
from . import initial_data as bid
from .rhs import BSSNSolver
from ._bssn_rhs_generated import FIELD_INPUTS, OUTPUT_FIELDS
from .derivative_bundle import field_dict

_LIB = Path(__file__).resolve().parent / "cuda" / "rhs_fused.so"
_TARGET = "bssn_rhs_fused_ffi"
_OUT_FIELDS = OUTPUT_FIELDS                       # OUT[k] = d/dt of this field (kernel write order)
NOUT = len(_OUT_FIELDS)
_registered = False


def _register() -> None:
    global _registered
    if _registered:
        return
    if not _LIB.exists():
        raise FileNotFoundError(f"{_LIB} not built. On the GPU host: bash {_LIB.parent}/build_fused.sh")
    lib = ctypes.cdll.LoadLibrary(str(_LIB))
    jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(lib.Fused), platform="CUDA")
    _registered = True


def _pack_S(eta, cahd_c, dt, dx, ssl_h, ssl_sigma, t, lmbda, lambda_f, dy, dz) -> jax.Array:
    return jnp.array([eta, cahd_c, dt, dx, ssl_h, ssl_sigma, t,
                      lmbda[0], lmbda[1], lmbda[2], lmbda[3], lambda_f[0], lambda_f[1],
                      dx, dy, dz], dtype=jnp.float64)


def _scalars(p: PhysicsParams, dx, dy, dz, dt, t) -> jax.Array:   # kept for tests
    return _pack_S(p.eta, p.cahd_c, dt, dx, p.ssl_h, p.ssl_sigma, t, p.lmbda, p.lambda_f, dy, dz)


def cuda_fused_algebra(F, dx, dy, dz, eta, lmbda, lambda_f, cahd_c, dt, ssl_h, ssl_sigma, t):
    """The M4 production scheme callable — matches the fused `_algebra` signature (field dict +
    scalars, no D) so `BSSNSolver(scheme="cuda_fused")` dispatches the full RK4 step through the
    fused CUDA RHS. Returns ``{field_name: d/dt}`` (the bare algebra RHS; KO is a separate
    `evolve._ko` pass — fusing it in was measured SLOWER, see memory `bssn-m4-fused-rhs-win`)."""
    _register()
    fields = jnp.stack([F[n] for n in FIELD_INPUTS])           # (NF, Sx, Sy, Sz)
    S = _pack_S(eta, cahd_c, dt, dx, ssl_h, ssl_sigma, t, lmbda, lambda_f, dy, dz)
    out_type = jax.ShapeDtypeStruct((NOUT,) + fields.shape[1:], jnp.float64)
    OUT = jax.ffi.ffi_call(_TARGET, out_type)(fields, S)
    return {_OUT_FIELDS[k]: OUT[k] for k in range(NOUT)}


def device_rhs_fused(state, params: PhysicsParams, dx, dy, dz, dt, t=0.0) -> Dict[str, jax.Array]:
    """The fused BSSN RHS on the GPU; ``{field_name: d/dt}`` (matches rhs_dict). Standalone helper
    over `cuda_fused_algebra` (unpacks a state + PhysicsParams)."""
    p = params
    return cuda_fused_algebra(field_dict(state), dx, dy, dz, p.eta, p.lmbda, p.lambda_f,
                              p.cahd_c, dt, p.ssl_h, p.ssl_sigma, t)


def _setup(n: int):
    g = Grid.from_domain(n, order=6)
    p = PhysicsParams()
    dt = 0.25 * float(g.dx)
    state = bid.gauge_wave(g, amplitude=0.01)
    return g, p, dt, state


def _selfcheck() -> int:
    g, p, dt, state = _setup(16)
    solver = BSSNSolver(g, p, order=6, scheme="verbatim", dt=dt)
    ref = solver.rhs_dict(state, t=0.0, dt=dt)                 # {field: d/dt} (verbatim)
    dev = jax.jit(lambda s: device_rhs_fused(BSSNState(s), p, g.dx, g.dy, g.dz, dt, 0.0))(state.data)
    worst, where = 0.0, ""
    for k in ref:
        e = float(jnp.max(jnp.abs(jnp.asarray(ref[k]) - dev[k])))
        if e > worst:
            worst, where = e, k
    ok = worst <= 1e-8
    print(f"[M4 fused RHS] platform={jax.devices()[0].platform}  grid={g.shape}  "
          f"max|fused - verbatim| = {worst:.3e} (at {where})  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _bench(fn, d0, batches: int = 8, per_batch: int = 4, warmup: int = 3) -> float:
    """Median ms/call of a jitted `fn` driven in a feedback loop from `d0`."""
    d = d0
    for _ in range(warmup):
        d = fn(d)
    d.block_until_ready()
    ts = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(per_batch):
            d = fn(d)
        d.block_until_ready()
        ts.append((time.perf_counter() - t0) / per_batch)
    return statistics.median(ts) * 1e3


def _time_pair(n: int):
    """(verbatim ms, M4 ms, grid-shape) at interior size n; jits both, times the RHS eval."""
    g, p, dt, state = _setup(n)
    solver = BSSNSolver(g, p, order=6, scheme="verbatim", dt=dt)
    verb = jax.jit(lambda s: solver.rhs(BSSNState(s), t=0.0, dt=dt).data)
    m4 = jax.jit(lambda s: jnp.stack(
        [device_rhs_fused(BSSNState(s), p, g.dx, g.dy, g.dz, dt, 0.0)[n2] for n2 in VAR_NAMES]))
    return _bench(verb, state.data), _bench(m4, state.data), g.shape


def _compare(n: int) -> None:
    tv, tm, shape = _time_pair(n)
    print(f">> RHS wall-clock @ {shape} (n={n}):")
    print(f"   verbatim XLA : {tv:8.3f} ms/eval")
    print(f"   M4 fused CUDA: {tm:8.3f} ms/eval")
    print(f"   speedup      : {tv / tm:6.2f}×  ({'WIN' if tm < tv else 'loss'})")


def _scale(ns) -> None:
    """Sweep grid sizes → does the M4 win hold (or grow) toward production resolution?"""
    print(f">> M4 fused RHS vs verbatim XLA — scaling sweep ({jax.devices()[0].platform})")
    print(f"  {'n':>4} {'grid':>16} {'Mpts':>7} {'verbatim ms':>12} {'M4 ms':>9} {'speedup':>8}")
    for n in ns:
        try:
            tv, tm, shape = _time_pair(n)
            mpts = (shape[0] * shape[1] * shape[2]) / 1e6
            print(f"  {n:>4} {str(shape):>16} {mpts:7.2f} {tv:12.3f} {tm:9.3f} "
                  f"{tv / tm:7.2f}×")
        except Exception as e:                       # OOM / compile failure at large n
            print(f"  {n:>4}  -> skipped ({type(e).__name__}: {str(e)[:50]})")


def _rk4_compare(n: int, ko_sigma: float = 0.1) -> None:
    """The production question: is the full RK4 STEP faster with M4? Times evolve.step (4 RHS +
    KO + algebraic enforcement + BC) for verbatim vs cuda_fused, after a one-step correctness check."""
    from .evolve import BSSNEvolution
    g, p, dt, state = _setup(n)
    evo_v = BSSNEvolution(g, p, order=6, ko_sigma=ko_sigma, scheme="verbatim")
    evo_m = BSSNEvolution(g, p, order=6, ko_sigma=ko_sigma, scheme="cuda_fused")

    sv = evo_v.step(state, 0.0, dt).data
    sm = evo_m.step(state, 0.0, dt).data
    err = float(jnp.max(jnp.abs(sv - sm)))
    print(f">> RK4-step correctness: max|cuda_fused - verbatim| = {err:.3e}  "
          f"-> {'PASS' if err <= 1e-8 else 'FAIL'}")

    step_v = jax.jit(lambda s: evo_v.step(BSSNState(s), 0.0, dt).data)
    step_m = jax.jit(lambda s: evo_m.step(BSSNState(s), 0.0, dt).data)
    tv = _bench(step_v, state.data)
    tm = _bench(step_m, state.data)
    print(f">> full RK4 STEP wall-clock @ {g.shape} (n={n}, ko_sigma={ko_sigma}):")
    print(f"   verbatim XLA : {tv:8.3f} ms/step")
    print(f"   M4 cuda_fused: {tm:8.3f} ms/step")
    print(f"   speedup      : {tv / tm:6.2f}×  ({'WIN' if tm < tv else 'loss'})")
    print("   (step = 4 RHS evals + KO + enforcement + BC; non-RHS parts dilute the RHS 2.28×)")


def _evolve_check(n: int, nsteps: int = 64, ko_sigma: float = 0.1) -> None:
    """Stability over many steps: evolve cuda_fused and verbatim `nsteps` and confirm cuda_fused
    stays finite/bounded and tracks verbatim (the one-step round-off accumulates, not blows up)."""
    from .evolve import BSSNEvolution
    from .state import ALPHA
    g, p, dt, state = _setup(n)
    sv = BSSNEvolution(g, p, order=6, ko_sigma=ko_sigma, scheme="verbatim").evolve(state, dt, nsteps)
    sm = BSSNEvolution(g, p, order=6, ko_sigma=ko_sigma, scheme="cuda_fused").evolve(state, dt, nsteps)
    sv.data.block_until_ready(); sm.data.block_until_ready()
    finite = bool(jnp.all(jnp.isfinite(sm.data)))
    maxa = float(jnp.max(jnp.abs(sm.data[ALPHA])))
    drift = float(jnp.max(jnp.abs(sv.data - sm.data)))
    print(f">> {nsteps}-step evolution (cuda_fused, n={n}, ko_sigma={ko_sigma}):")
    print(f"   finite = {finite}   max|alpha| = {maxa:.5f}   (bounded => stable)")
    print(f"   drift vs verbatim after {nsteps} steps = {drift:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 fused BSSN RHS — validate / wall-clock compare")
    ap.add_argument("--compare", action="store_true", help="wall-clock M4 vs verbatim XLA at --n")
    ap.add_argument("--scale", action="store_true", help="sweep grid sizes (the --ns list)")
    ap.add_argument("--rk4", action="store_true", help="full RK4 STEP: verbatim vs cuda_fused")
    ap.add_argument("--evolve", action="store_true", help="multi-step stability of cuda_fused")
    ap.add_argument("--nsteps", type=int, default=64)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ns", type=int, nargs="+", default=[48, 64, 96, 128, 160, 192],
                    help="grid sizes for --scale")
    args = ap.parse_args()
    if jax.devices()[0].platform != "gpu":
        print(">> CPU backend: the fused kernel is GPU-only. Run on the GPU host.")
        return
    if args.evolve:
        _evolve_check(args.n, args.nsteps)
    elif args.rk4:
        _rk4_compare(args.n)
    elif args.scale:
        _scale(args.ns)
    elif args.compare:
        _compare(args.n)
    else:
        raise SystemExit(_selfcheck())


if __name__ == "__main__":
    main()
