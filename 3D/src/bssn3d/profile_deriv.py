"""Wall-A baseline profile for the M2a device derivative kernel (Step 3.2e).

M2a reads neighbours straight from HBM (1 thread/point, no on-chip reuse) and writes all 138
derivatives back to HBM — so it is the **memory-bound baseline** that exhibits wall A in full.
This harness samples the permission-free regime tell (`nvidia-smi dmon` SM%/MEM%, Nsight banned)
and the wall-clock per eval, so M2b's SMEM 2.5D streaming has a concrete A/B to beat (MEM%↓ and a
lower HBM read once the halo is SMEM/L2-served).

Mirrors `profile_regime.py` (same sampler, same `s + eps·f(s)` feedback so XLA can't DCE the
kernel). Uses `mcs_common.jax_config.setup()` to enable x64 BEFORE importing the FD stencils
(else the fp32-stencil trap — see memory `x64-stencil-import-order`).

  python -m bssn3d.profile_deriv --smi  --n 128 --seconds 15
  python -m bssn3d.profile_deriv --time --n 128
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
_jax_setup()                          # x64 + persistent compile cache, before any FD-stencil import

import jax

from .grid import Grid
from .state import BSSNState
from . import initial_data as bid
from .deriv_kernel import device_derivative_bundle, N_DERIV

_EPS = 1e-9


def _build(n: int, variant: str = "global"):
    grid = Grid.from_domain(n, order=6)
    dx, dy, dz = float(grid.dx), float(grid.dy), float(grid.dz)
    data0 = bid.gauge_wave(grid, amplitude=0.01).data

    def step(data):
        D = device_derivative_bundle(BSSNState(data), dx, dy, dz, variant)  # 138 × (Sx,Sy,Sz)
        acc = sum(D.values())                                       # depends on ALL 138 → no DCE
        return data + _EPS * acc[None, ...]                         # bounded feedback, all 24 fields

    return grid, data0, jax.jit(step)


def _warm(data, step, warmup):
    for _ in range(warmup):
        data = step(data)
    data.block_until_ready()
    return data


def smi_capture(n: int, seconds: int, warmup: int, variant: str) -> None:
    grid, data, step = _build(n, variant)
    data = _warm(data, step, warmup)
    try:
        proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
                                stdout=subprocess.PIPE, text=True, bufsize=1)
    except FileNotFoundError:
        print(">> nvidia-smi not found — cannot sample utilization.")
        return

    samples, col = [], {"sm": None, "mem": None}

    def reader():
        for line in proc.stdout:
            toks = line.lstrip("#").split()
            if "sm" in toks and "mem" in toks:
                col["sm"], col["mem"] = toks.index("sm"), toks.index("mem")
                continue
            if col["sm"] is None or not toks or not toks[0].isdigit():
                continue
            try:
                samples.append((float(toks[col["sm"]]), float(toks[col["mem"]])))
            except (IndexError, ValueError):
                pass

    threading.Thread(target=reader, daemon=True).start()
    t0, ncalls = time.time(), 0
    while time.time() - t0 < seconds:
        for _ in range(8):
            data = step(data)
        data.block_until_ready()
        ncalls += 8
    proc.terminate()
    time.sleep(0.7)

    steady = samples[2:] if len(samples) > 3 else samples
    print(f"\n>> nvidia-smi dmon: {len(steady)} steady samples over ~{seconds}s "
          f"({ncalls} deriv evals, {grid.shape}, {N_DERIV} derivs)")
    if not steady:
        print("   (no samples parsed — check `nvidia-smi dmon -s u` output format)")
        return
    sm = statistics.median(s for s, _ in steady)
    mem = statistics.median(m for _, m in steady)
    print(f"   SM  utilization : {sm:5.0f}%   (kernel executing)")
    print(f"   MEM utilization : {mem:5.0f}%   (memory-controller busy → DRAM tell)")
    if mem >= 60:
        verdict = (f"DRAM/memory-bound ({variant}: redundant HBM reads + 138-array deriv write). "
                   + ("the M2a baseline for M2b to beat." if variant == "global"
                      else "M2b is still write-bound on the deriv output (the fused M4 kills that)."))
    else:
        verdict = "compute/underutilized → grow --n (24 fields ≫ L2 only at large N)"
    print(f"   coarse verdict  : {verdict}")
    print(f">> Paste these SM/MEM lines back ({variant}).")


def time_capture(n: int, batches: int, per_batch: int, warmup: int, variant: str) -> None:
    grid, data, step = _build(n, variant)
    data = _warm(data, step, warmup)
    times = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(per_batch):
            data = step(data)
        data.block_until_ready()
        times.append((time.perf_counter() - t0) / per_batch)
    ms = statistics.median(times) * 1e3
    tag = "M2a global" if variant == "global" else "M2b smem"
    print(f">> {tag} deriv eval: {ms:.3f} ms/eval median  ({grid.shape}, {N_DERIV} derivs, "
          f"{batches}×{per_batch} reps)")


def main() -> None:
    ap = argparse.ArgumentParser(description="M2a derivative-kernel wall-A baseline profile")
    ap.add_argument("--n", type=int, default=128, help="interior cells/axis (full grid = n+2*ng)")
    ap.add_argument("--smi", action="store_true", help="sample SM%%/MEM%% (default if neither set)")
    ap.add_argument("--time", action="store_true", help="wall-clock ms/eval")
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--per-batch", type=int, default=8)
    ap.add_argument("--variant", choices=["global", "smem"], default="global",
                    help="global = M2a (1 thread/pt); smem = M2b (2.5D streaming)")
    args = ap.parse_args()

    if jax.devices()[0].platform != "gpu":
        print(">> CPU backend: this is a GPU regime/timing profile. Run on the GPU host.")
        return
    if args.time:
        time_capture(args.n, args.batches, args.per_batch, args.warmup, args.variant)
    else:
        smi_capture(args.n, args.seconds, args.warmup, args.variant)


if __name__ == "__main__":
    main()
