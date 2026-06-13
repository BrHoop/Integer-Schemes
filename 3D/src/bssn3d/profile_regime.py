"""GPU regime profiling for the 3D BSSN RHS — Phase 3.2a.

Answers the one question Phase 3 hinges on with the cluster's permission-free tools
(Nsight banned, perf counters locked → no direct DRAM/SM/TC counters): **is the BSSN
RHS kernel DRAM-bound or compute-bound?** NVML exposes coarse utilizations without
counters, so `nvidia-smi dmon -s u` (SM% = kernel executing, MEM% = memory-controller
busy ≈ DRAM-activity tell) gives the DRAM-vs-compute verdict, and a JAX/XLA trace gives
the per-op device-time breakdown.

The 3.2a use: profile the **staged-XLA** RHS (`--scheme staged`, 0 spill but 107
kernels) and confirm it is *still* memory-bound (MEM% high) despite the despill — the
number that motivates the Pallas register-resident kernel (3.2c). A/B against
`--scheme verbatim` (the Phase-2 baseline). The regime flip Phase 3 must show is
MEM%↓ / SM%↑ once the Pallas kernel keeps intermediates on-chip.

This profiles the RHS evaluation itself (the kernel of interest), looped via a tiny
`s + eps*rhs(s)` feedback so XLA can't DCE it and the state can't blow up over the
sampling window (regime is independent of the RHS *values*).

Examples (H200 node):
  python -m bssn3d.profile_regime --smi --scheme staged --n 128 --seconds 15
  python -m bssn3d.profile_regime --smi --scheme verbatim --n 128
  python -m bssn3d.profile_regime --jax-trace traces/bssn_staged --scheme staged
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from mcs_common.jax_config import setup as _jax_setup
_jax_setup()                      # x64 + persistent compile cache, before any jit

import jax

from .grid import Grid
from .state import PhysicsParams, BSSNState
from .rhs import BSSNSolver
from . import initial_data as bid

# Tiny feedback coefficient: exercises the full RHS kernel each step while keeping the
# state from diverging over thousands of steps (the regime is value-independent).
_EPS = 1e-8


def _build(n: int, order: int, scheme: str):
    grid = Grid.from_domain(n, order=order)
    solver = BSSNSolver(grid, PhysicsParams(), order=order, scheme=scheme)
    state = bid.gauge_wave(grid, amplitude=0.01)
    step = jax.jit(lambda s: BSSNState(s.data + _EPS * solver.rhs(s).data))
    return solver, state, step


def _warm(state, step, warmup):
    for _ in range(warmup):
        state = step(state)
    state.data.block_until_ready()
    return state


def smi_capture(n, order, scheme, seconds, warmup):
    """Loop the RHS step for ~`seconds` while `nvidia-smi dmon -s u` samples; report
    steady-state SM%/MEM% (the permission-free DRAM-vs-compute tell)."""
    _, state, step = _build(n, order, scheme)
    state = _warm(state, step, warmup)

    try:
        proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
            stdout=subprocess.PIPE, text=True, bufsize=1)
    except FileNotFoundError:
        print(">> nvidia-smi not found — cannot sample utilization.")
        return

    samples, col = [], {"sm": None, "mem": None}

    def reader():
        for line in proc.stdout:
            toks = line.lstrip("#").split()
            if "sm" in toks and "mem" in toks:           # header row
                col["sm"], col["mem"] = toks.index("sm"), toks.index("mem")
                continue
            if col["sm"] is None or not toks or not toks[0].isdigit():
                continue
            try:
                samples.append((float(toks[col["sm"]]), float(toks[col["mem"]])))
            except (IndexError, ValueError):
                pass

    threading.Thread(target=reader, daemon=True).start()

    t0, nsteps = time.time(), 0
    while time.time() - t0 < seconds:
        for _ in range(64):
            state = step(state)
        state.data.block_until_ready()
        nsteps += 64
    proc.terminate()
    time.sleep(0.7)                                       # let the reader drain

    steady = samples[2:] if len(samples) > 3 else samples  # drop ramp samples
    print(f"\n>> nvidia-smi dmon: {len(steady)} steady samples over ~{seconds}s "
          f"({nsteps} RHS steps, {n}^3, scheme={scheme})")
    if not steady:
        print("   (no samples parsed — check `nvidia-smi dmon -s u` output format)")
        return
    sm = statistics.median(s for s, _ in steady)
    mem = statistics.median(m for _, m in steady)
    print(f"   SM  utilization : {sm:5.0f}%   (kernel executing)")
    print(f"   MEM utilization : {mem:5.0f}%   (memory-controller busy → DRAM tell)")
    if mem >= 60:
        v = (f"DRAM/memory-bound  → {scheme} moves its working set through HBM "
             "(inter-kernel intermediates / register spill / the 138-array deriv read)")
    elif sm >= 60:
        v = "compute-bound      → regime flipped; the on-chip target is reached"
    else:
        v = "underutilized      → grow --n (latency-bound at this size) before judging"
    print(f"   coarse verdict  : {v}")
    print(">> Paste these SM/MEM lines + verdict back.")


def jax_trace(n, order, scheme, steps, warmup, outdir):
    """Emit a JAX/XLA profiler trace (open in Perfetto or TensorBoard)."""
    _, state, step = _build(n, order, scheme)
    state = _warm(state, step, warmup)
    os.makedirs(outdir, exist_ok=True)
    jax.profiler.start_trace(outdir)
    for _ in range(steps):
        state = step(state)
    state.data.block_until_ready()
    jax.profiler.stop_trace()
    print(f">> JAX trace → {outdir}  (open *.trace.json.gz in Perfetto, or "
          f"`tensorboard --logdir {outdir}`)")


def run(n, order, scheme, steps, warmup):
    """Plain loop — for external live monitoring (nvtop)."""
    _, state, step = _build(n, order, scheme)
    state = _warm(state, step, warmup)
    for _ in range(steps):
        state = step(state)
    state.data.block_until_ready()


def main():
    ap = argparse.ArgumentParser(description="BSSN RHS GPU regime profiling (allowed tools only)")
    ap.add_argument("--n", type=int, default=128, help="grid (N = Nx = Ny = Nz)")
    ap.add_argument("--order", type=int, default=6, help="FD order (ghost zones)")
    ap.add_argument("--scheme",
                    choices=["verbatim", "staged", "pallas", "fused", "fused_fp64"],
                    default="staged", help="RHS algebra variant to profile")
    ap.add_argument("--steps", type=int, default=300, help="loop steps (plain / jax-trace)")
    ap.add_argument("--warmup", type=int, default=20, help="warmup steps")
    ap.add_argument("--seconds", type=float, default=15.0, help="--smi sampling duration")
    ap.add_argument("--smi", action="store_true",
                    help="sample SM%%/MEM%% with nvidia-smi dmon (DRAM-vs-compute)")
    ap.add_argument("--jax-trace", metavar="DIR", default=None,
                    help="emit a JAX/XLA profiler trace into DIR")
    a = ap.parse_args()

    print(f">> bssn3d.profile_regime | backend={jax.default_backend()} "
          f"| {a.n}^3 | order={a.order} | scheme={a.scheme}", flush=True)
    if jax.default_backend() != "gpu":
        print(">> CPU backend: SM%/MEM% are GPU-only. Run on a GPU node; building "
              "once to check it compiles ...")
        run(a.n, a.order, a.scheme, steps=2, warmup=1)
        print(">> built + ran 2 steps on CPU (no regime numbers).")
        return
    if a.smi:
        smi_capture(a.n, a.order, a.scheme, a.seconds, a.warmup)
    elif a.jax_trace:
        jax_trace(a.n, a.order, a.scheme, a.steps, a.warmup, a.jax_trace)
    else:
        run(a.n, a.order, a.scheme, a.steps, a.warmup)
    print(">> done.", flush=True)


if __name__ == "__main__":
    main()
