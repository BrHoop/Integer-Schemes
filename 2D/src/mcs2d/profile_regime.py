"""
GPU regime profiling for the fused FP64 step — Step 1.2 T5.

This cluster bans NVIDIA Nsight (ncu/nsys) and locks GPU performance counters
(ERR_NVGPUCTRPERM), so DRAM throughput / SM occupancy / TC utilization are not
directly measurable.  The allowed, permission-free tools still answer the one
question we need — *is the step DRAM-bound or compute-bound?* — because NVML
exposes coarse utilizations without counters:

  * `--smi`        run the step loop while `nvidia-smi dmon` samples SM% and
                   MEM%.  MEM% = memory-controller busy % ≈ a DRAM-activity tell.
  * `--jax-trace`  emit a JAX/XLA profiler trace (Perfetto/TensorBoard) for the
                   per-op device-time breakdown (the JAX analog of the PyTorch
                   profiler; ideal for before/after-fusion comparison).
  * (no flag)      just run the loop — e.g. to eyeball `nvtop` in another shell.

Examples (on an H200 node):
  python -m mcs2d.profile_regime --smi --nx 512 --seconds 15
  python -m mcs2d.profile_regime --jax-trace traces/jax --nx 512 --steps 300
  # or, live: salloc a node, `ssh` to it, `module load nvtop; nvtop`, run no-flag.

NB: the steady-state regime was already established from an nsys *trace* (graph
trace: 99.2% GPU-busy, 100% theoretical occupancy, 0 B shared memory, stencil via
global dynamic_slice → DRAM-bound).  `--smi` is the permission-free confirmation
and the go-forward tool for Step 1.7 (watch MEM% drop as the kernel is optimized).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import statistics
import subprocess
import threading
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from mcs_common.jax_config import setup as _jax_setup
_jax_setup()                      # x64 + persistent compile cache, before any jit

import jax

from mcs2d.main import MaxwellChernSimons2D, InitialData, load_parameters


def _build(nx: int, params_file: str):
    params = load_parameters(params_file)
    params.update({
        "scheme": "fused_floating_point", "Nx": nx, "Ny": nx, "Nt": 1,
        "id_type": "birefringent", "bc_type": "periodic", "sponge_strength": 0.0,
    })
    dx = (params["xmax"] - params["xmin"]) / nx
    sim = MaxwellChernSimons2D(dx, dx, params["Lambda"], params)
    state = InitialData(sim, params).generate()
    step = jax.jit(lambda s: sim.step_rk4(s, sim.dt))
    return sim, state, step


def _warm(state, step, warmup):
    for _ in range(warmup):
        state = step(state)
    state.data.block_until_ready()
    return state


def run(nx, steps, warmup, params_file):
    """Plain loop — for external live monitoring (nvtop)."""
    sim, state, step = _build(nx, params_file)
    state = _warm(state, step, warmup)
    for _ in range(steps):
        state = step(state)
    state.data.block_until_ready()
    return state


def smi_capture(nx, seconds, warmup, params_file):
    """Run the step loop for ~`seconds` while `nvidia-smi dmon -s u` samples;
    report steady-state SM% and MEM% (the permission-free DRAM-vs-compute tell)."""
    sim, state, step = _build(nx, params_file)
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

    t0, n = time.time(), 0
    while time.time() - t0 < seconds:
        for _ in range(64):
            state = step(state)
        state.data.block_until_ready()
        n += 64
    proc.terminate()
    time.sleep(0.7)                                       # let the reader drain

    steady = samples[2:] if len(samples) > 3 else samples  # drop ramp samples
    print(f"\n>> nvidia-smi dmon: {len(steady)} steady samples over ~{seconds}s "
          f"({n} steps, {nx}²)")
    if not steady:
        print("   (no samples parsed — check `nvidia-smi dmon -s u` output format)")
        return
    sm = statistics.median(s for s, _ in steady)
    mem = statistics.median(m for _, m in steady)
    print(f"   SM  utilization : {sm:5.0f}%   (kernel executing)")
    print(f"   MEM utilization : {mem:5.0f}%   (memory-controller busy → DRAM tell)")
    if mem >= 60:
        v = "DRAM/memory-bound  → 1.3 lever = tiling + temporal fusion (cut traffic)"
    elif sm >= 60:
        v = "compute-bound      → tensor cores help directly"
    else:
        v = "underutilized      → occupancy / launch first"
    print(f"   coarse verdict  : {v}")


def jax_trace(nx, steps, warmup, params_file, outdir):
    """Emit a JAX/XLA profiler trace (open in Perfetto or TensorBoard)."""
    sim, state, step = _build(nx, params_file)
    state = _warm(state, step, warmup)
    os.makedirs(outdir, exist_ok=True)
    jax.profiler.start_trace(outdir)
    for _ in range(steps):
        state = step(state)
    state.data.block_until_ready()
    jax.profiler.stop_trace()
    print(f">> JAX trace → {outdir}  (open the *.trace.json.gz in Perfetto, or "
          f"`tensorboard --logdir {outdir}`)")


def main():
    ap = argparse.ArgumentParser(description="GPU regime profiling (allowed tools only)")
    ap.add_argument("--nx", type=int, default=512, help="grid (Nx = Ny)")
    ap.add_argument("--steps", type=int, default=300, help="loop steps (plain / jax-trace)")
    ap.add_argument("--warmup", type=int, default=40, help="warmup steps")
    ap.add_argument("--seconds", type=float, default=15.0, help="--smi sampling duration")
    ap.add_argument("--smi", action="store_true",
                    help="sample SM%%/MEM%% with nvidia-smi dmon (DRAM-vs-compute)")
    ap.add_argument("--jax-trace", metavar="DIR", default=None,
                    help="emit a JAX/XLA profiler trace into DIR")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[2] / "params.toml"))
    a = ap.parse_args()
    print(f">> profile_regime: fused_floating_point  {a.nx}²", flush=True)
    if a.smi:
        smi_capture(a.nx, a.seconds, a.warmup, a.params)
    elif a.jax_trace:
        jax_trace(a.nx, a.steps, a.warmup, a.params, a.jax_trace)
    else:
        run(a.nx, a.steps, a.warmup, a.params)
    print(">> done.", flush=True)


if __name__ == "__main__":
    main()
