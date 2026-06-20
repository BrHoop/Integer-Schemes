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

**Wall-clock is the rigorous metric.** The SM%/MEM% regime read is a coarse,
counter-free *proxy* (Nsight banned, perf counters locked) — it can say "off HBM"
but not "faster." `--time`/`--compare` measure time-per-step directly, which is the
number Phase 3 must actually win: a kernel can flip the regime (MEM%↓/SM%↑) yet be
*slower* in wall-clock because the de-spill bought it with redundant FLOP (recompute
+ halo). `--compare` reports speedup vs the verbatim baseline so a pyrrhic win shows.

Examples (H200 node):
  python -m bssn3d.profile_regime --smi --scheme staged --n 128 --seconds 15
  python -m bssn3d.profile_regime --time --scheme fused_tiled --n 128
  python -m bssn3d.profile_regime --compare --n 128            # A/B all schemes
  python -m bssn3d.profile_regime --compare --rk4 --n 128      # full RK4 step
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
from .evolve import BSSNEvolution
from . import initial_data as bid

ALL_SCHEMES = ["verbatim", "staged", "pallas", "fused", "fused_fp64", "fused_tiled"]
# Schemes that actually LOWER on GPU (Triton). `fused`/`fused_fp64` are the whole-grid
# jnp.pad/lax.slice prototypes — non-pow2 → they do NOT lower (CPU-interpret-only); they
# clutter a GPU `--compare` with expected FAILED rows. `--compare` defaults to this set.
GPU_COMPARE_SCHEMES = ["verbatim", "staged", "pallas", "fused_tiled"]

# Tiny feedback coefficient: exercises the full RHS kernel each step while keeping the
# state from diverging over thousands of steps (the regime is value-independent).
_EPS = 1e-8


def _build(n: int, order: int, scheme: str):
    grid = Grid.from_domain(n, order=order)
    solver = BSSNSolver(grid, PhysicsParams(), order=order, scheme=scheme)
    state = bid.gauge_wave(grid, amplitude=0.01)
    step = jax.jit(lambda s: BSSNState(s.data + _EPS * solver.rhs(s).data))
    return solver, state, step


def _build_rk4(n: int, order: int, scheme: str, ko_sigma: float = 0.0):
    """Full RK4 step (4 RHS evals + KO + algebraic enforcement + BC sync) on the
    given scheme — the true time-per-step the production loop pays, vs `_build`'s
    bare single RHS eval."""
    grid = Grid.from_domain(n, order=order)
    evo = BSSNEvolution(grid, PhysicsParams(), order=order,
                        ko_sigma=ko_sigma, scheme=scheme)
    state = bid.gauge_wave(grid, amplitude=0.01)
    dt = 0.25 * grid.dx
    step = jax.jit(lambda s: evo.step(s, 0.0, dt))
    return evo, state, step


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
    print(f"   MEM utilization : {mem:5.0f}%   (memory-controller BUSY-TIME — NOT bandwidth %)")

    # Roofline disambiguation: MEM% is controller-busy-TIME; scattered SPILL/local traffic inflates
    # it without saturating bandwidth. Estimate the achieved field-I/O bandwidth (24 read + 24 write
    # RHS arrays, interior floor) and compare to HBM peak — if MEM% is high but BW% is low, the
    # memory activity is spill/halo re-reads, not bandwidth-binding (the Nsight counters we lack
    # would show this directly; the roofline is the permission-free proxy).
    PEAK_BW = 3.35e12                       # H100 SXM HBM3 (GB/s class)
    per_eval = seconds / max(nsteps, 1)
    bytes_rhs = 48 * (n ** 3) * 8           # 24 fields read + 24 outputs written (interior floor)
    bw = bytes_rhs / per_eval
    bw_frac = bw / PEAK_BW
    print(f"   field-I/O BW    : {bw/1e9:5.0f} GB/s = {bw_frac*100:4.1f}% of HBM peak "
          f"(interior floor; halo/spill add traffic but are L1/L2-served)")

    if bw_frac >= 0.6:
        v = "BANDWIDTH-bound    → field I/O genuinely saturates HBM; reduce DATA moved (precision/BFP48)"
    elif mem >= 60 and bw_frac < 0.30:
        v = ("SPILL/latency-bound → MEM% high but field-I/O BW low ⇒ the memory activity is register "
             "spill + halo re-reads (L1/L2), NOT bandwidth. The e-graph (cut peak-live) is the lever.")
    elif sm >= 60:
        v = "compute-bound      → regime flipped; the on-chip target is reached"
    else:
        v = "underutilized      → grow --n (latency-bound at this size) before judging"
    print(f"   verdict         : {v}")
    print(">> Paste these SM/MEM/BW lines + verdict back.")


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


def _time_step(state, step, batches, per_batch):
    """Median/min per-step wall-clock (seconds), measured in `batches` runs of
    `per_batch` steps each, blocking once per batch so async dispatch can't hide
    the work. The median is the headline; min is the noise-floor (best case)."""
    state.data.block_until_ready()
    per = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(per_batch):
            state = step(state)
        state.data.block_until_ready()
        per.append((time.perf_counter() - t0) / per_batch)
    return state, per


def _compile_time(step, state):
    """Seconds for the first (compiling) call — the one-time Pallas/XLA compile
    cost. Cached + shape-keyed in practice, but worth reporting."""
    t0 = time.perf_counter()
    s = step(state)
    s.data.block_until_ready()
    return s, time.perf_counter() - t0


def time_capture(n, order, scheme, batches, per_batch, warmup, rk4, ko_sigma):
    """Wall-clock per-step for one scheme — the rigorous, counter-free metric
    (independent of the banned Nsight counters and the coarse SM%/MEM% proxy).
    This is the number Phase 3 must actually win: time-per-step, not regime."""
    builder = _build_rk4 if rk4 else _build
    args = (n, order, scheme, ko_sigma) if rk4 else (n, order, scheme)
    _, state, step = builder(*args)
    kind = "RK4 step" if rk4 else "RHS eval"

    state, compile_s = _compile_time(step, state)
    state = _warm(state, step, warmup)
    state, per = _time_step(state, step, batches, per_batch)

    med, lo = statistics.median(per), min(per)
    npts = n ** 3
    print(f"\n>> wall-clock | scheme={scheme} | {n}^3 = {npts:,} pts | {kind}")
    print(f"   compile (first call) : {compile_s:8.2f} s")
    print(f"   per step  (median)   : {med * 1e3:8.3f} ms")
    print(f"   per step  (min)      : {lo * 1e3:8.3f} ms")
    print(f"   throughput (median)  : {npts / med / 1e6:8.1f} Mpts/s")
    print(">> Paste these back.")
    return med


def compare_schemes(n, order, schemes, batches, per_batch, warmup, rk4, ko_sigma):
    """A/B every scheme on wall-clock per-step and report speedup vs the verbatim
    baseline (the do-nothing reference: what you'd run if Phase 3 didn't exist).
    A scheme that flips the regime but is slower here is a *pyrrhic* win — this
    table is the honest scoreboard the regime metric can't give."""
    kind = "RK4 step" if rk4 else "RHS eval"
    print(f">> scheme comparison | {n}^3 | {kind} | "
          f"{batches}×{per_batch} steps/scheme", flush=True)
    rows = []
    for sc in schemes:
        try:
            builder = _build_rk4 if rk4 else _build
            args = (n, order, sc, ko_sigma) if rk4 else (n, order, sc)
            _, state, step = builder(*args)
            state, compile_s = _compile_time(step, state)
            state = _warm(state, step, warmup)
            state, per = _time_step(state, step, batches, per_batch)
            rows.append([sc, statistics.median(per), min(per), compile_s, None])
            print(f"   {sc:<12} ok ({statistics.median(per) * 1e3:.2f} ms)", flush=True)
        except Exception as exc:                              # noqa: BLE001
            rows.append([sc, None, None, None, repr(exc)[:70]])
            print(f"   {sc:<12} FAILED: {repr(exc)[:70]}", flush=True)

    base = next((r[1] for r in rows if r[0] == "verbatim" and r[1] is not None), None)
    npts = n ** 3
    print(f"\n   {'scheme':<12} {'median ms':>10} {'min ms':>9} "
          f"{'Mpts/s':>9} {'vs verbatim':>12} {'compile s':>10}")
    print("   " + "-" * 66)
    for sc, med, lo, comp, err in rows:
        if med is None:
            print(f"   {sc:<12} {'—':>10} {'—':>9} {'—':>9} {'—':>12} "
                  f"{'—':>10}   {err}")
            continue
        spd = f"{base / med:6.2f}x" if base else "    n/a"
        print(f"   {sc:<12} {med * 1e3:10.3f} {lo * 1e3:9.3f} "
              f"{npts / med / 1e6:9.1f} {spd:>12} {comp:10.2f}")
    print("\n>> 'vs verbatim' > 1 = faster than the unoptimized baseline. The "
          "regime flip is only a win if this column beats 1. Paste this table back.")


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
                    choices=["verbatim", "staged", "pallas", "fused", "fused_fp64",
                             "fused_tiled", "cuda_fused"],
                    default="staged", help="RHS algebra variant to profile")
    ap.add_argument("--steps", type=int, default=300, help="loop steps (plain / jax-trace)")
    ap.add_argument("--warmup", type=int, default=20, help="warmup steps")
    ap.add_argument("--seconds", type=float, default=15.0, help="--smi sampling duration")
    ap.add_argument("--smi", action="store_true",
                    help="sample SM%%/MEM%% with nvidia-smi dmon (DRAM-vs-compute)")
    ap.add_argument("--jax-trace", metavar="DIR", default=None,
                    help="emit a JAX/XLA profiler trace into DIR")
    ap.add_argument("--time", action="store_true",
                    help="wall-clock per-step for --scheme (the rigorous metric)")
    ap.add_argument("--compare", action="store_true",
                    help="wall-clock A/B across ALL schemes; speedup vs verbatim")
    ap.add_argument("--rk4", action="store_true",
                    help="time the full RK4 step (4 RHS + KO + enforce) not a bare RHS eval")
    ap.add_argument("--ko-sigma", type=float, default=0.0,
                    help="KO dissipation strength for --rk4 timing")
    ap.add_argument("--batches", type=int, default=10, help="timing batches (median over these)")
    ap.add_argument("--per-batch", type=int, default=50, help="steps per timing batch")
    ap.add_argument("--schemes", default=None,
                    help="comma-list for --compare (default = GPU-lowering set, i.e. drops "
                         "the non-pow2 fused/fused_fp64 prototypes that can't lower)")
    a = ap.parse_args()

    print(f">> bssn3d.profile_regime | backend={jax.default_backend()} "
          f"| {a.n}^3 | order={a.order} | scheme={a.scheme}", flush=True)

    # Wall-clock timing works on any backend (it IS the counter-free metric); only
    # the SM%/MEM% regime read is GPU-only.
    if a.compare:
        if jax.default_backend() != "gpu":
            print(">> NOTE: CPU timings do not reflect the GPU regime — smoke-test only.")
        schemes = a.schemes.split(",") if a.schemes else GPU_COMPARE_SCHEMES
        compare_schemes(a.n, a.order, schemes, a.batches, a.per_batch,
                        a.warmup, a.rk4, a.ko_sigma)
        print(">> done.", flush=True)
        return
    if a.time:
        if jax.default_backend() != "gpu":
            print(">> NOTE: CPU timings do not reflect the GPU regime — smoke-test only.")
        time_capture(a.n, a.order, a.scheme, a.batches, a.per_batch,
                     a.warmup, a.rk4, a.ko_sigma)
        print(">> done.", flush=True)
        return

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
