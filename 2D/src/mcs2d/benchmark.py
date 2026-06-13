"""
Scan-based scheme benchmark for the MCS 2D solver.

Compares fused_floating_point and fused_ozaki at production grid sizes (default
256² and 512²).  floating_point (unfused, untiled) is omitted — fused_floating_point
is identical FP64 arithmetic and strictly faster, so it is the FP64 baseline.
Unfused ozaki is also omitted — fused_ozaki uses the same INT8 pipeline and is
strictly equal-or-faster.

Methodology
-----------
Each measurement compiles a function that runs N_SCAN_STEPS steps via
`jax.lax.scan`.  This is the key difference from a naive Python loop: every
step happens on-device with no Python→JAX dispatch between steps, so the
measured time reflects true GPU throughput.

Reported per (scheme, grid):
  compile_s          — first scan call (includes XLA compile; reused across runs
                       via persistent compile cache)
  per_step_us        — median steady-state per-step time (Python overhead nil)
  rhs_per_step_us    — same, for RHS-only path
  throughput_Mpts_s  — Nx² / per_step
  end_to_end_s       — compile_s + N_REPORT_STEPS × per_step  (true wall clock
                       for a realistic-length simulation)
  state_MB           — Nx_tot · Ny_tot · 10 · 8 B
  peak_device_MB     — jax.devices()[0].memory_stats()
  l2_err             — Ez error vs analytic oracle after N_CORRECT steps

Usage:
    python mcs2d/benchmark.py [params.toml] [output_dir]

Edit GRID_SIZES at the top of this file if you want a different sweep.
"""

import os
import sys
import csv
import time
import statistics
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_dir  = str(Path(__file__).resolve().parent)
_root = str(Path(__file__).resolve().parent.parent)
for _p in [_dir, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcs_common.jax_config import setup as _jax_setup
_jax_setup()

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcs2d.main import (
    MaxwellChernSimons2D, InitialData,
    get_physical, load_parameters, l2norm,
)


# ── Configuration ──────────────────────────────────────────────────────────────

GRID_SIZES = [256, 512]   # production-scale only; small grids weren't bottlenecked
# `ozaki` (unfused) is omitted: `fused_ozaki` uses the same INT8 RNS pipeline
# and is strictly equal-or-faster.  Re-add `ozaki` here if you want to quantify
# the fusion-only benefit on the INT8 path.
# `pallas_ozaki` is the single-kernel-per-tile version that keeps CRT
# intermediates in shared memory (no L2 round-trips between Garner levels).
# `floating_point` (unfused, untiled) is omitted: `fused_floating_point` is
# identical FP64 arithmetic and strictly faster — it is now the FP baseline.
# Re-add "fused_ozaki", "pallas_ozaki" here to also benchmark the INT8 path.
SCHEMES    = ["fused_floating_point"]

# Each measurement compiles a function that runs N_SCAN_STEPS via lax.scan,
# eliminating Python-loop dispatch overhead between steps.  This is the key
# difference from a naive loop: it measures the GPU's actual throughput, not
# the Python→JAX dispatch latency.
N_SCAN_STEPS    = 100   # steps per scan call (amortizes dispatch overhead)
N_REPS          = 3     # repeat scan calls; report median for steady state
N_REPORT_STEPS  = 2000  # extrapolated end-to-end horizon (compile + steady·N)
N_CORRECT       = 100   # oracle correctness check

NF = 10  # number of MCS fields


# ── GPU / device metadata ──────────────────────────────────────────────────────

# Theoretical peak HBM bandwidth (GB/s).  Used for bandwidth-efficiency %.
_BW_TABLE = {
    "H200 NVL": 4800, "H200": 4800,            # Hopper HBM3e, ~4.8 TB/s
    "H100 SXM": 3350, "H100 PCIe": 2000, "H100": 3350,
    "GH200": 4900, "B200": 8000, "GB200": 8000,
    "A100": 2039, "V100": 900,
    "RTX 5090": 1792, "RTX 4090": 1008, "RTX 3090": 936, "RTX 3080": 760,
    "RTX 2080": 448, "L40": 864, "T4": 320,
}
# Peak compute for the roofline ceilings: (FP64 vector GFLOP/s, INT8 TC GOP/s).
# FP64 = CUDA-core vector path (today's stencil); INT8 TC = the thesis headroom
# (dense; 2:4 sparse roughly doubles it).
_COMPUTE_TABLE = {
    "H200": (34000, 1979000), "H100": (34000, 1979000),
    "GH200": (34000, 1979000), "B200": (40000, 4500000),
    "A100": (9700, 624000), "RTX 5090": (1700, 838000),
    "RTX 4090": (1300, 660000),
}


def _match_table(device, table):
    for key, val in table.items():
        if key.lower() in device.lower():
            return val
    return None


def gpu_info():
    info = {
        "backend":          jax.default_backend(),
        "device":           str(jax.devices()[0]),
        "vram_gb":          None,
        "peak_bw_GBs":      None,
        "peak_fp64_GFLOPs": None,
        "peak_int8_TOPs":   None,
    }

    # 1) pynvml (best: exact name + VRAM).  2) jax device_kind.  3) nvidia-smi.
    name, vram = None, None
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        nm = pynvml.nvmlDeviceGetName(h)
        name = nm.decode() if isinstance(nm, bytes) else nm
        vram = round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 1024**3, 1)
    except Exception:
        pass
    if not name:
        try:
            dk = jax.devices()[0].device_kind          # e.g. "NVIDIA H200"
            if dk and "cuda" not in dk.lower():
                name = dk
        except Exception:
            pass
    if not name or vram is None:
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                nm, mem = out.splitlines()[0].split(",")
                name = name or nm.strip()
                vram = vram if vram is not None else round(float(mem) / 1024, 1)
        except Exception:
            pass

    if name:
        info["device"] = name
    info["vram_gb"] = vram
    info["peak_bw_GBs"] = _match_table(info["device"], _BW_TABLE)
    comp = _match_table(info["device"], _COMPUTE_TABLE)
    if comp:
        info["peak_fp64_GFLOPs"], info["peak_int8_TOPs"] = comp
    return info


def _device_peak_mem_mb():
    """Peak device memory in MB since last JAX program start, or None."""
    try:
        stats = jax.devices()[0].memory_stats()
        b = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
        return round(b / 1e6, 1) if b else None
    except Exception:
        return None


def _rhs_cost(rhs_fn, state):
    """XLA HLO cost model of ONE RHS eval: GFLOPs, MB accessed, and arithmetic
    intensity (FLOP/byte → compute- vs memory-bound).  Best-effort ({} if N/A)."""
    try:
        ca = jax.jit(rhs_fn).lower(state).compile().cost_analysis()
    except Exception:
        return {}
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else {}
    if not isinstance(ca, dict):
        return {}
    flops = float(ca.get("flops", 0.0) or 0.0)
    byts = float(ca.get("bytes accessed", 0.0) or 0.0)
    out = {"rhs_GFLOP": round(flops / 1e9, 4), "rhs_MB_accessed": round(byts / 1e6, 2)}
    if byts > 0:
        out["rhs_flop_per_byte"] = round(flops / byts, 3)
        out["bound"] = "compute" if flops / byts > 10.0 else "memory"
    return out


def _step_cost(step_fn, state):
    """XLA HLO cost model of ONE full RK4 step (FLOPs + bytes accessed).

    CAVEAT: cost-model "bytes accessed" counts ALL loads/stores, including the
    temporal fusion's redundant on-chip halo reads (empirically step bytes ≈
    120× state) — most of which hit L1/L2/shared, not HBM.  So the derived
    GFLOP/s, GB/s and %peakBW are UPPER BOUNDS that include on-chip traffic, NOT
    DRAM bandwidth.  Real DRAM throughput requires hardware counters (locked on this cluster).
    Still useful for the roofline framing + the FLOP/byte regime.  ({} if N/A.)"""
    try:
        ca = jax.jit(step_fn).lower(state).compile().cost_analysis()
    except Exception:
        return {}
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else {}
    if not isinstance(ca, dict):
        return {}
    flops = float(ca.get("flops", 0.0) or 0.0)
    byts = float(ca.get("bytes accessed", 0.0) or 0.0)
    out = {"step_GFLOP": round(flops / 1e9, 4), "step_MB_accessed": round(byts / 1e6, 2)}
    if byts > 0:
        out["step_flop_per_byte"] = round(flops / byts, 3)
    return out


# ── Birefringent oracle ────────────────────────────────────────────────────────

def _make_oracle(params):
    Lx    = params["xmax"] - params["xmin"]
    Ly    = params["ymax"] - params["ymin"]
    k_x   = 2.0 * jnp.pi / Lx
    k_y   = 2.0 * jnp.pi / Ly
    k     = jnp.sqrt(k_x**2 + k_y**2)
    m_cs  = params.get("id_m_cs", params.get("Lambda", 1.0) * 2.0)
    E0    = params.get("id_amp", 1.0)
    omega = jnp.sqrt(k**2 + m_cs * k)
    return lambda X, Y, t: E0 * jnp.sin(k_x * X + k_y * Y - omega * t)


# ── Single (scheme, grid) measurement ─────────────────────────────────────────

def measure(scheme, nx, base_params):
    """Scan-based measurement.

    All step-time numbers are amortized over N_SCAN_STEPS inside a single
    `lax.scan`, so they measure GPU compute throughput — not Python dispatch.

    Returns metrics:
      compile_s         — one-time cost (first scan call, including XLA compile)
      per_step_us       — steady-state, median over N_REPS scan calls
      rhs_per_step_us   — same idea for the RHS-only path (derivative cost)
      throughput_Mpts_s — Nx² / per_step
      end_to_end_s      — compile_s + N_REPORT_STEPS · per_step (true wall clock)
    """
    params = dict(base_params)
    params.update({
        "scheme":          scheme,
        "Nx": nx, "Ny": nx,
        "Nt": 1,
        "id_type":         "birefringent",
        "bc_type":         "periodic",
        "sponge_strength": 0.0,
    })

    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / nx
    sim   = MaxwellChernSimons2D(dx, dy, params["Lambda"], params)
    state = InitialData(sim, params).generate()
    state_bytes = sim.Nx_tot * sim.Ny_tot * NF * 8
    cost = _rhs_cost(sim.rhs, state)   # HLO FLOP/byte for one RHS (compute vs memory bound)
    step_cost = _step_cost(lambda s: sim.step_rk4(s, sim.dt), state)  # per-STEP HBM traffic

    # Scan-based runners — all stepping happens on the device per call.
    @jax.jit
    def scan_steps(s):
        return jax.lax.scan(lambda c, _: (sim.step_rk4(c, sim.dt), None),
                            s, None, length=N_SCAN_STEPS)[0]

    @jax.jit
    def scan_rhs(s):
        # N_SCAN_STEPS RHS calls; output discarded (last carry returned).
        return jax.lax.scan(lambda c, _: (sim.rhs(c), None),
                            s, None, length=N_SCAN_STEPS)[0]

    # ── First call: compile + execute.  Time captures cold compile cost. ──
    t0 = time.perf_counter()
    state = scan_steps(state)
    state.data.block_until_ready()
    compile_s = time.perf_counter() - t0
    peak_mem_mb = _device_peak_mem_mb()

    # ── Steady-state: median over N_REPS subsequent scan calls ────────────
    step_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        state = scan_steps(state)
        state.data.block_until_ready()
        step_times.append(time.perf_counter() - t0)
    per_step_us = statistics.median(step_times) / N_SCAN_STEPS * 1e6

    # ── RHS-only: same scan pattern ───────────────────────────────────────
    scan_rhs(state).data.block_until_ready()  # compile
    rhs_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        r = scan_rhs(state); r.data.block_until_ready()
        rhs_times.append(time.perf_counter() - t0)
    rhs_per_step_us = statistics.median(rhs_times) / N_SCAN_STEPS * 1e6

    throughput_Mpts_s = (nx * nx) / (per_step_us * 1e-6 * 1e6)

    # ── HLO-cost-model throughput (UPPER BOUND, incl. on-chip — see _step_cost).
    # hlo_bytes_pct_peakBW is filled later once peak BW is known.
    per_step_s = per_step_us * 1e-6
    hlo = {}
    if step_cost.get("step_GFLOP"):
        hlo["hlo_GFLOPs"] = round(step_cost["step_GFLOP"] / per_step_s, 1)
    if step_cost.get("step_MB_accessed"):
        hlo["hlo_GBs"] = round(
            step_cost["step_MB_accessed"] / 1e3 / per_step_s, 1)

    # ── End-to-end wall: realistic simulation horizon ─────────────────────
    end_to_end_s = compile_s + N_REPORT_STEPS * per_step_us * 1e-6

    # ── Correctness: L2 vs birefringent oracle (separate fresh IC) ────────
    state_c = InitialData(sim, params).generate()
    # Compile a smaller scan to hit exactly N_CORRECT steps.
    @jax.jit
    def scan_correct(s):
        return jax.lax.scan(lambda c, _: (sim.step_rk4(c, sim.dt), None),
                            s, None, length=N_CORRECT)[0]
    state_c = scan_correct(state_c); state_c.data.block_until_ready()

    x_p = sim.x[sim.ng:-sim.ng]
    y_p = sim.y[sim.ng:-sim.ng]
    X, Y = jnp.meshgrid(x_p, y_p, indexing='ij')
    l2_err = float(l2norm(
        get_physical(state_c.data[sim.EZ], sim.ng)
        - _make_oracle(params)(X, Y, N_CORRECT * sim.dt)
    ))

    return {
        "scheme":            scheme,
        "nx":                nx,
        "compile_s":         compile_s,
        "per_step_us":       per_step_us,
        "rhs_per_step_us":   rhs_per_step_us,
        "throughput_Mpts_s": throughput_Mpts_s,
        f"end_to_end_s_N{N_REPORT_STEPS}": end_to_end_s,
        "state_MB":          round(state_bytes / 1e6, 2),
        "peak_device_MB":    peak_mem_mb,
        "l2_err":            l2_err,
        **cost,
        **step_cost,
        **hlo,
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def _f(val, fmt, na="N/A"):
    return format(val, fmt) if val is not None else na


def print_table(rows, gpu):
    e2e_key = f"end_to_end_s_N{N_REPORT_STEPS}"
    hdr = (
        f"  {'Scheme':<22} {'Nx':>4}  "
        f"{'Compile':>9}  {'Step':>10}  {'RHS':>10}  "
        f"{'Mpts/s':>8}  {f'E2E(N={N_REPORT_STEPS})':>13}  "
        f"{'StateMB':>8}  {'PeakMB':>8}  {'L2 err':>10}"
    )
    sep = "-" * len(hdr)

    print(f"\n{'='*len(hdr)}")
    print(f"  MCS 2D Benchmark — scan-based timings")
    print(f"  Device   : {gpu['device']}")
    print(f"  Backend  : {gpu['backend']}")
    if gpu["vram_gb"]: print(f"  VRAM     : {gpu['vram_gb']} GB")
    print(f"  Method   : each step time = (jit_scan_{N_SCAN_STEPS}_steps) / {N_SCAN_STEPS}")
    print(f"  Reps     : {N_REPS} scan calls; median reported")
    print(f"  E2E col  : compile_s + {N_REPORT_STEPS} · per_step  (true wall clock)")
    print(f"  L2 col   : Ez error vs analytic oracle after {N_CORRECT} steps")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print(sep)

    for r in rows:
        print(
            f"  {r['scheme']:<22} {r['nx']:>4}  "
            f"  {_f(r['compile_s'],         '7.2f')}s  "
            f"  {_f(r['per_step_us'],       '7.1f')}us  "
            f"  {_f(r['rhs_per_step_us'],   '7.1f')}us  "
            f"  {_f(r['throughput_Mpts_s'], '6.1f')}  "
            f"  {_f(r[e2e_key],             '9.2f')}s   "
            f"  {_f(r['state_MB'],          '6.1f')}  "
            f"  {_f(r['peak_device_MB'],    '6.1f') if r['peak_device_MB'] else '   N/A':>6}  "
            f"  {_f(r['l2_err'],            '.2e')}"
        )
    print(sep)
    # HLO cost-model line.  NB: cost-model "bytes accessed" counts ON-CHIP traffic
    # too (the temporal fusion's redundant halo), so these GFLOP/s, GB/s and
    # %peakBW are UPPER BOUNDS that include on-chip work — NOT DRAM bandwidth.
    # Real DRAM throughput needs HW counters (locked; use profile_regime --smi).  Trustworthy here:
    # the RHS arithmetic intensity (rhs FLOP/byte) and the memory/compute `bound`.
    print(f"  {'HLO cost (UB)':<22} {'Nx':>4}  {'rhsFLOP/B':>10}  "
          f"{'GFLOP/s*':>10}  {'GB/s*':>9}  {'%peakBW*':>9}  {'bound':>8}")
    for r in rows:
        print(
            f"  {r['scheme']:<22} {r['nx']:>4}  "
            f"  {_f(r.get('rhs_flop_per_byte'), '8.2f')}  "
            f"  {_f(r.get('hlo_GFLOPs'),    '8.0f')}  "
            f"  {_f(r.get('hlo_GBs'),       '7.0f')}  "
            f"  {_f(r.get('hlo_bytes_pct_peakBW'), '7.1f')}  "
            f"  {str(r.get('bound', '?')):>8}"
        )
    print(f"  * cost-model upper bound (incl. on-chip halo); NOT DRAM — real DRAM"
          f" needs HW counters (locked here); use `profile_regime.py --smi`")
    print(sep)


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n>> CSV  →  {path}")


_COLORS  = {"floating_point": "#2196F3", "fused_floating_point": "#9C27B0",
            "ozaki": "#FF9800", "fused_ozaki": "#4CAF50",
            "pallas_ozaki": "#E91E63"}
_MARKERS = {"floating_point": "o",       "fused_floating_point": "D",
            "ozaki": "s",       "fused_ozaki": "^",
            "pallas_ozaki": "P"}


def _color(scheme):
    """Plot colour for a scheme, with a neutral fallback so an unknown scheme
    (e.g. one added to SCHEMES without a palette entry) never crashes a plot and
    discards a completed benchmark run."""
    return _COLORS.get(scheme, "#607D8B")


def _marker(scheme):
    return _MARKERS.get(scheme, "o")


def save_throughput_plot(rows, path, gpu):
    """Throughput and steady-state per-step time vs grid size (log-log)."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    for scheme in SCHEMES:
        rs = sorted([r for r in rows if r["scheme"] == scheme], key=lambda r: r["nx"])
        if not rs:
            continue
        xs = [r["nx"] for r in rs]
        ax0.loglog(xs, [r["throughput_Mpts_s"] for r in rs],
                   marker=_marker(scheme), label=scheme, color=_color(scheme),
                   lw=2, markersize=8)
        ax1.loglog(xs, [r["per_step_us"] for r in rs],
                   marker=_marker(scheme), label=scheme, color=_color(scheme),
                   lw=2, markersize=8)

    ax0.set(xlabel="Grid Nx (= Ny)", ylabel="Throughput (Mpts/s)",
            title="Throughput vs Grid Size")
    ax0.legend(); ax0.grid(True, which="both", alpha=0.3)
    ax1.set(xlabel="Grid Nx (= Ny)", ylabel="Per-step time (µs)",
            title="Steady-state per-step time")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"MCS 2D — scan-based timings  |  {gpu['device']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f">> Plot  →  {path}")


def save_end_to_end_plot(rows, path, gpu):
    """End-to-end wall time for N_REPORT_STEPS — the 'which scheme should I use' plot."""
    e2e_key = f"end_to_end_s_N{N_REPORT_STEPS}"
    fig, ax = plt.subplots(figsize=(10, 5))

    grids   = sorted({r["nx"] for r in rows})
    schemes = [s for s in SCHEMES if any(r["scheme"] == s for r in rows)]
    bar_w   = 0.8 / len(schemes)

    for i, scheme in enumerate(schemes):
        rs = sorted([r for r in rows if r["scheme"] == scheme], key=lambda r: r["nx"])
        if not rs:
            continue
        # stack compile_s on bottom (one-time) and steady-state on top
        comp = [r["compile_s"] for r in rs]
        body = [r[e2e_key] - r["compile_s"] for r in rs]
        x    = [grids.index(r["nx"]) + (i - (len(schemes)-1)/2) * bar_w for r in rs]
        ax.bar(x, comp, bar_w, color=_color(scheme), alpha=0.45, hatch="//",
               label=f"{scheme} (compile)")
        ax.bar(x, body, bar_w, bottom=comp, color=_color(scheme), alpha=0.9,
               label=f"{scheme} (run)")
        for xi, r in zip(x, rs):
            ax.text(xi, r[e2e_key] * 1.02, f"{r[e2e_key]:.1f}s",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels([f"{n}×{n}" for n in grids])
    ax.set_ylabel(f"Wall time for {N_REPORT_STEPS} steps (s)")
    ax.set_title(f"End-to-end wall time  |  {gpu['device']}")
    ax.legend(loc="upper left", fontsize=8, ncol=len(schemes))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f">> Plot  →  {path}")


def save_compile_plot(rows, path, gpu):
    """Bar chart of one-time compile cost per scheme × grid."""
    fig, ax = plt.subplots(figsize=(10, 5))
    grids   = sorted({r["nx"] for r in rows})
    schemes = [s for s in SCHEMES if any(r["scheme"] == s for r in rows)]
    bar_w   = 0.8 / len(schemes)
    for i, scheme in enumerate(schemes):
        rs = sorted([r for r in rows if r["scheme"] == scheme], key=lambda r: r["nx"])
        x  = [grids.index(r["nx"]) + (i - (len(schemes)-1)/2) * bar_w for r in rs]
        ax.bar(x, [r["compile_s"] for r in rs], bar_w,
               color=_color(scheme), label=scheme)
        for xi, r in zip(x, rs):
            ax.text(xi, r["compile_s"] * 1.02, f"{r['compile_s']:.1f}s",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels([f"{n}×{n}" for n in grids])
    ax.set_ylabel("XLA compile time (s)")
    ax.set_title(f"One-time compile cost  |  {gpu['device']}  "
                 f"(persistent cache reuses across runs)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f">> Plot  →  {path}")


def _add_speedups(rows):
    """Add per-step and per-RHS speedup vs the fused_floating_point baseline (per grid).
    speedup > 1 means faster than the tiled FP64 reference."""
    base = {r["nx"]: r for r in rows if r["scheme"] == "fused_floating_point"}
    for r in rows:
        b = base.get(r["nx"])
        r["speedup_step_vs_fp"] = (round(b["per_step_us"] / r["per_step_us"], 3)
                                   if b and r["per_step_us"] else None)
        r["speedup_rhs_vs_fp"] = (round(b["rhs_per_step_us"] / r["rhs_per_step_us"], 3)
                                  if b and r["rhs_per_step_us"] else None)
    return rows


def _add_bandwidth(rows, gpu):
    """Add hlo_bytes_pct_peakBW = hlo_GBs / peak_bw_GBs · 100 once peak BW is known.

    WARNING: NOT DRAM efficiency.  hlo_GBs is from the cost model's "bytes
    accessed", which includes on-chip / redundant-halo traffic, so this % is an
    UPPER BOUND on memory utilization.  Real DRAM % needs HW counters (locked; use profile_regime --smi)."""
    peak = gpu.get("peak_bw_GBs")
    for r in rows:
        ach = r.get("hlo_GBs")
        r["hlo_bytes_pct_peakBW"] = (round(100.0 * ach / peak, 1)
                           if peak and ach else None)
    return rows


def save_json(rows, path, gpu):
    """Structured results for post-hoc graphing/analysis."""
    import json
    # Record the Pallas pow2-padding state — it's MANDATORY on GPU (Triton
    # requires pow2 tile size) but makes pallas_* compute on a padded tile, so
    # its per-step time is handicapped by `pad_waste×` extra cells.  Interpret
    # any pallas_* speedup with this in mind.
    pad_meta = {}
    try:
        from mcs2d.schemes import pallas_ozaki as _p
        pad_meta = {
            "pallas_pad_pow2": bool(_p._PAD_POW2),
            "pallas_H_orig": _p.H_ORIG, "pallas_H_pad": _p.H_PAD,
            "pallas_NF": _p.NF, "pallas_NF_pad": _p.NF_PAD,
            "pallas_pad_waste": round((_p.NF_PAD * _p.H_PAD**2)
                                      / (_p.NF * _p.H_ORIG**2), 3),
        }
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump({"device": gpu.get("device"), "backend": gpu.get("backend"),
                   "vram_gb": gpu.get("vram_gb"),
                   "peak_bw_GBs": gpu.get("peak_bw_GBs"),
                   "peak_fp64_GFLOPs": gpu.get("peak_fp64_GFLOPs"),
                   "peak_int8_TOPs": gpu.get("peak_int8_TOPs"),
                   "grids": GRID_SIZES,
                   "schemes": SCHEMES, "n_scan_steps": N_SCAN_STEPS,
                   **pad_meta, "rows": rows}, f, indent=2)
    print(f">> JSON  →  {path}")


def save_speedup_plot(rows, path, gpu):
    """Grouped bar chart: per-step speedup of each scheme vs fused_floating_point."""
    grids   = sorted({r["nx"] for r in rows})
    schemes = [s for s in SCHEMES if any(r["scheme"] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(grids))
    w = 0.8 / max(1, len(schemes))
    for i, s in enumerate(schemes):
        ys = []
        for g in grids:
            r = next((r for r in rows if r["scheme"] == s and r["nx"] == g), None)
            ys.append((r or {}).get("speedup_step_vs_fp") or 0.0)
        bars = ax.bar(x + i * w, ys, w, label=s)
        ax.bar_label(bars, fmt="%.2f×", fontsize=8, padding=2)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="fused_floating_point baseline")
    ax.set_xticks(x + w * (len(schemes) - 1) / 2)
    ax.set_xticklabels([f"{g}²" for g in grids])
    ax.set_ylabel("per-step speedup vs fused_floating_point  (×, higher = faster)")
    ax.set_title(f"Scheme speedup vs fused FP64 baseline  —  {gpu.get('device', '')}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f">> Plot  →  {path}")


def save_roofline_plot(rows, path, gpu):
    """Roofline: the trustworthy parts are the HBM/FP64/INT8 ceilings and ridges
    (FP64 ≈7, INT8 ≈410) and the RHS arithmetic intensity (≈3.2 FLOP/byte) — which
    sits far left of both ridges, quantifying how much temporal fusion must raise
    intensity for tensor cores to pay off.

    The plotted point's HEIGHT (hlo_GFLOPs) is a cost-model UPPER BOUND — it
    includes the temporal fusion's redundant on-chip work — so it is drawn hollow
    and the true achieved point needs HW counters (locked here).  x uses the RHS
    intensity (single-pass, cleaner) rather than the redundancy-deflated step IA."""
    peak_bw   = gpu.get("peak_bw_GBs")
    peak_fp64 = gpu.get("peak_fp64_GFLOPs")
    peak_int8 = gpu.get("peak_int8_TOPs")
    # x: RHS intensity (clean single-pass) preferred; y: cost-model UB perf.
    pts = [(r.get("rhs_flop_per_byte") or r.get("step_flop_per_byte"),
            r.get("hlo_GFLOPs"), r) for r in rows]
    pts = [(x, y, r) for (x, y, r) in pts if x and y]
    if not (peak_bw and peak_fp64 and pts):
        raise ValueError("roofline needs peak BW + FP64 ceiling + >=1 point "
                         "(rhs_flop_per_byte & hlo_GFLOPs)")

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.logspace(-1, 3.2, 200)                       # 0.1 .. ~1600 FLOP/byte
    ax.plot(xs, peak_bw * xs, "k-", lw=1.2, label=f"HBM {peak_bw/1000:.1f} TB/s")
    ax.axhline(peak_fp64, color="C0", ls="--", lw=1.2,
               label=f"FP64 vector {peak_fp64/1000:.0f} TFLOP/s")
    r_fp64 = peak_fp64 / peak_bw
    ax.axvline(r_fp64, color="C0", ls=":", lw=0.8, alpha=0.6)
    ax.text(r_fp64, peak_fp64 * 1.15, f"FP64 ridge {r_fp64:.1f}",
            fontsize=7, color="C0", ha="center")
    if peak_int8:
        ax.axhline(peak_int8, color="C3", ls="--", lw=1.2,
                   label=f"INT8 TC {peak_int8/1000:.0f} TOP/s (thesis headroom)")
        r_int8 = peak_int8 / peak_bw
        ax.axvline(r_int8, color="C3", ls=":", lw=0.8, alpha=0.6)
        ax.text(r_int8, peak_int8 * 1.15, f"INT8 ridge {r_int8:.0f}",
                fontsize=7, color="C3", ha="center")
    for i, (x, y, r) in enumerate(sorted(pts, key=lambda t: t[0])):
        # hollow marker — the height is a cost-model upper bound, not measured perf
        ax.plot(x, y, "o", mfc="none", mec=_color(r["scheme"]), mew=1.6, ms=11)
        ax.annotate(f"{r['scheme']} {r['nx']}²  (HLO-UB)",
                    (x, y), fontsize=7, ha="left", va="top",
                    xytext=(8, -6 - 12 * i), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)  —  RHS, single-pass")
    ax.set_ylabel("performance  (GFLOP/s · GOP/s INT8)")
    ax.set_title(f"Roofline — {gpu.get('device', '')}  (MCS FD step)")
    ax.text(0.02, 0.02,
            "○ height = cost-model upper bound (incl. on-chip halo); "
            "true point needs HW counters (locked here)",
            transform=ax.transAxes, fontsize=6.5, color="grey", va="bottom")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f">> Plot  →  {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _render_plots(rows, out_dir, gpu):
    """Write all benchmark figures.  Each plot is guarded so a single failing
    figure cannot discard the others or the (already-saved) CSV/JSON data."""
    plots = [
        ("benchmark_throughput.png", save_throughput_plot),
        ("benchmark_end_to_end.png", save_end_to_end_plot),
        ("benchmark_compile.png",    save_compile_plot),
        ("benchmark_speedup.png",    save_speedup_plot),
        ("benchmark_roofline.png",   save_roofline_plot),
    ]
    for name, fn in plots:
        try:
            fn(rows, os.path.join(out_dir, name), gpu)
        except Exception as exc:
            print(f">> Plot FAILED ({name}): {exc}  [data is safe in CSV/JSON]")


def replot(json_path, out_dir=None):
    """Regenerate all figures from a saved benchmark_results.json — no GPU re-run.

    Use this when the timing run completed (CSV/JSON written) but plotting failed,
    or to re-style figures locally from data pulled off the supercomputer.
    """
    import json
    with open(json_path) as f:
        blob = json.load(f)
    rows = blob["rows"]
    gpu = {"device": blob.get("device"), "backend": blob.get("backend"),
           "vram_gb": blob.get("vram_gb")}
    # Peak compute/BW: from the JSON if present, else re-derive from the device name.
    dev = gpu["device"] or ""
    gpu["peak_bw_GBs"] = blob.get("peak_bw_GBs") or _match_table(dev, _BW_TABLE)
    comp = _match_table(dev, _COMPUTE_TABLE) or (None, None)
    gpu["peak_fp64_GFLOPs"] = blob.get("peak_fp64_GFLOPs") or comp[0]
    gpu["peak_int8_TOPs"]   = blob.get("peak_int8_TOPs")   or comp[1]
    out_dir = out_dir or str(Path(json_path).resolve().parent)
    os.makedirs(out_dir, exist_ok=True)
    _add_speedups(rows)
    _add_bandwidth(rows, gpu)
    _render_plots(rows, out_dir, gpu)
    print(f">> Replotted from {json_path} into {out_dir}/")


def main():
    # Default to the canonical 2D/params.toml (full config incl. Order); there is
    # no params.toml next to this script, and load_parameters' missing-file
    # fallback dict omits several keys the benchmark/oracle need.
    parfile = (sys.argv[1] if len(sys.argv) > 1
               else str(Path(__file__).resolve().parents[2] / "params.toml"))
    out_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(_dir) / "output")
    os.makedirs(out_dir, exist_ok=True)

    base_params = load_parameters(parfile)
    gpu         = gpu_info()

    total = len(GRID_SIZES) * len(SCHEMES)
    print(f"\n{'='*60}")
    print(f"  MCS 2D Scheme Benchmark (scan-based)")
    print(f"  Device  : {gpu['device']}  ({gpu['backend']})")
    if gpu["vram_gb"]: print(f"  VRAM    : {gpu['vram_gb']} GB")
    if gpu["peak_bw_GBs"]:
        print(f"  Peak BW : {gpu['peak_bw_GBs']/1000:.1f} TB/s  "
              f"(FP64 {(_f(gpu['peak_fp64_GFLOPs'],'.0f') if gpu['peak_fp64_GFLOPs'] else '?')} GFLOP/s, "
              f"INT8-TC {(_f(gpu['peak_int8_TOPs'],'.0f') if gpu['peak_int8_TOPs'] else '?')} GOP/s)")
    else:
        print(f"  Peak BW : UNKNOWN (device '{gpu['device']}' not in _BW_TABLE; "
              f"roofline + hlo_bytes_pct_peakBW will be N/A)")
    print(f"  Grids   : {GRID_SIZES}")
    print(f"  Schemes : {SCHEMES}")
    print(f"  Per cfg : compile + {N_REPS} × scan({N_SCAN_STEPS} steps) "
          f"+ {N_CORRECT}-step oracle check")
    print(f"  E2E col : compile + {N_REPORT_STEPS} steady steps (true wall clock)")
    print(f"{'='*60}")

    rows = []
    done = 0
    e2e_key = f"end_to_end_s_N{N_REPORT_STEPS}"
    for nx in GRID_SIZES:
        for scheme in SCHEMES:
            done += 1
            print(f"\n[{done}/{total}]  {scheme:<22}  {nx}×{nx} ...", flush=True)
            try:
                row = measure(scheme, nx, base_params)
                rows.append(row)
                print(
                    f"         compile={row['compile_s']:.2f}s  "
                    f"step={row['per_step_us']:.1f}us  "
                    f"rhs={row['rhs_per_step_us']:.1f}us  "
                    f"E2E={row[e2e_key]:.1f}s  "
                    f"L2={row['l2_err']:.2e}"
                )
            except Exception as exc:
                print(f"         FAILED: {exc}")

    if rows:
        _add_speedups(rows)
        _add_bandwidth(rows, gpu)
        print_table(rows, gpu)
        # Save data FIRST so a plotting failure can never lose the timing run.
        save_csv(rows,  os.path.join(out_dir, "benchmark_results.csv"))
        save_json(rows, os.path.join(out_dir, "benchmark_results.json"), gpu)
        _render_plots(rows, out_dir, gpu)


if __name__ == "__main__":
    # `--replot <results.json> [out_dir]` regenerates figures from saved data
    # (no GPU re-run); otherwise run the full benchmark.
    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        replot(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        main()
