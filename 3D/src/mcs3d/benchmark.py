"""
Scan-based scheme benchmark + roofline for the MCS 3D solver (Phase 1, Step 1.4).

The 3D port of `2D/src/mcs2d/benchmark.py`.  3D currently has a single production
scheme (`floating_point`); the fused/Ozaki tiled paths are Phase-4 work, so the
sweep is a single scheme over a few cube sizes.

Methodology (unchanged from 2D)
-------------------------------
Each measurement compiles a function that runs N_SCAN_STEPS steps via
`jax.lax.scan`, so every step happens on-device with no Python->JAX dispatch
between steps: the measured time reflects true GPU throughput, not dispatch
latency.

GPU/CPU
-------
The timing numbers are only meaningful on the H200 (`profile_regime.py --smi`
gives the DRAM-vs-compute regime; Nsight is banned on the cluster).  The plumbing
runs on CPU too -- the cost-model (HLO FLOP/byte) and roofline framing are
device-independent and CPU-safe; only the wall-clock throughput needs the GPU.

Usage:
    python -m mcs3d.benchmark [params.toml] [output_dir]
    python -m mcs3d.benchmark --replot <benchmark_results.json> [out_dir]
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

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcs3d.main import (
    MaxwellChernSimons3D, InitialData,
    get_physical, load_parameters, l2norm,
)


# ── Configuration ──────────────────────────────────────────────────────────────

GRID_SIZES = [64, 128]          # cube edge N (N^3 points); modest for 3D HBM
SCHEMES    = ["floating_point"]  # 3D's only production scheme (fused = Phase 4)

N_SCAN_STEPS    = 50    # steps per scan call (amortizes dispatch overhead)
N_REPS          = 3     # repeat scan calls; report median for steady state
N_REPORT_STEPS  = 2000  # extrapolated end-to-end horizon (compile + steady*N)
N_CORRECT       = 50    # oracle correctness check

NF = 10  # number of MCS fields


# ── GPU / device metadata (shared with the 2D benchmark) ───────────────────────

_BW_TABLE = {
    "H200 NVL": 4800, "H200": 4800,
    "H100 SXM": 3350, "H100 PCIe": 2000, "H100": 3350,
    "GH200": 4900, "B200": 8000, "GB200": 8000,
    "A100": 2039, "V100": 900,
    "RTX 5090": 1792, "RTX 4090": 1008, "RTX 3090": 936, "RTX 3080": 760,
    "RTX 2080": 448, "L40": 864, "T4": 320,
}
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
            dk = jax.devices()[0].device_kind
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
    try:
        stats = jax.devices()[0].memory_stats()
        b = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
        return round(b / 1e6, 1) if b else None
    except Exception:
        return None


def _rhs_cost(rhs_fn, state):
    """XLA HLO cost model of ONE RHS eval: GFLOPs, MB accessed, and arithmetic
    intensity (FLOP/byte -> compute- vs memory-bound).  Best-effort ({} if N/A)."""
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

    CAVEAT (same as 2D): the cost model's "bytes accessed" counts ALL loads/stores
    including on-chip/halo traffic, so the derived GFLOP/s and GB/s are UPPER
    BOUNDS, not DRAM bandwidth.  Real DRAM needs HW counters (locked on the
    cluster); use `profile_regime.py --smi`."""
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


# ── Birefringent oracle (Ez component, 3D) ─────────────────────────────────────

def _make_oracle(params):
    Lx = params["xmax"] - params["xmin"]
    Ly = params["ymax"] - params["ymin"]
    Lz = params["zmax"] - params["zmin"]
    kx = 2.0 * np.pi / Lx; ky = 2.0 * np.pi / Ly; kz = 2.0 * np.pi / Lz
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    m_cs = params.get("id_m_cs", params.get("Lambda", 1.0) * 2.0)
    E0 = params.get("id_amp", 1.0)
    omega = np.sqrt(k**2 + m_cs * k)
    nf = np.sqrt(kx**2 + ky**2)
    e1z = 0.0
    e2z = nf / k
    # Ez = E0*(e1z*cos - e2z*sin) = -E0*e2z*sin(phase)
    def ez(X, Y, Z, t):
        phase = kx * X + ky * Y + kz * Z - omega * t
        return E0 * (e1z * np.cos(phase) - e2z * np.sin(phase))
    return ez


# ── Single (scheme, grid) measurement ─────────────────────────────────────────

def measure(scheme, nx, base_params):
    """Scan-based measurement; all step-time numbers are amortized over
    N_SCAN_STEPS inside one `lax.scan`, so they measure GPU throughput."""
    params = dict(base_params)
    params.update({
        "scheme":          scheme,
        "Nx": nx, "Ny": nx, "Nz": nx,
        "Nt": 1,
        "id_type":         "birefringent",
        "bc_type":         "periodic",
        "sponge_strength": 0.0,
    })

    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / nx
    dz = (params["zmax"] - params["zmin"]) / nx
    sim   = MaxwellChernSimons3D(dx, dy, dz, params["Lambda"], params)
    state = InitialData(sim, params).generate()
    state_bytes = sim.Nx_tot * sim.Ny_tot * sim.Nz_tot * NF * 8
    cost = _rhs_cost(sim.rhs, state)
    step_cost = _step_cost(lambda s: sim.step_rk4(s, sim.dt), state)

    @jax.jit
    def scan_steps(s):
        return jax.lax.scan(lambda c, _: (sim.step_rk4(c, sim.dt), None),
                            s, None, length=N_SCAN_STEPS)[0]

    @jax.jit
    def scan_rhs(s):
        return jax.lax.scan(lambda c, _: (sim.rhs(c), None),
                            s, None, length=N_SCAN_STEPS)[0]

    t0 = time.perf_counter()
    state = scan_steps(state)
    state.data.block_until_ready()
    compile_s = time.perf_counter() - t0
    peak_mem_mb = _device_peak_mem_mb()

    step_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        state = scan_steps(state)
        state.data.block_until_ready()
        step_times.append(time.perf_counter() - t0)
    per_step_us = statistics.median(step_times) / N_SCAN_STEPS * 1e6

    scan_rhs(state).data.block_until_ready()
    rhs_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        r = scan_rhs(state); r.data.block_until_ready()
        rhs_times.append(time.perf_counter() - t0)
    rhs_per_step_us = statistics.median(rhs_times) / N_SCAN_STEPS * 1e6

    throughput_Mpts_s = (nx ** 3) / (per_step_us * 1e-6 * 1e6)

    per_step_s = per_step_us * 1e-6
    hlo = {}
    if step_cost.get("step_GFLOP"):
        hlo["hlo_GFLOPs"] = round(step_cost["step_GFLOP"] / per_step_s, 1)
    if step_cost.get("step_MB_accessed"):
        hlo["hlo_GBs"] = round(step_cost["step_MB_accessed"] / 1e3 / per_step_s, 1)

    end_to_end_s = compile_s + N_REPORT_STEPS * per_step_us * 1e-6

    # Correctness: L2 vs birefringent oracle (Ez), fresh IC.
    state_c = InitialData(sim, params).generate()
    @jax.jit
    def scan_correct(s):
        return jax.lax.scan(lambda c, _: (sim.step_rk4(c, sim.dt), None),
                            s, None, length=N_CORRECT)[0]
    state_c = scan_correct(state_c); state_c.data.block_until_ready()

    x_p = np.asarray(sim.x[sim.ng:-sim.ng])
    y_p = np.asarray(sim.y[sim.ng:-sim.ng])
    z_p = np.asarray(sim.z[sim.ng:-sim.ng])
    X, Y, Z = np.meshgrid(x_p, y_p, z_p, indexing="ij")
    l2_err = float(np.sqrt(np.mean(
        (np.asarray(get_physical(state_c.data[sim.EZ], sim.ng))
         - _make_oracle(params)(X, Y, Z, N_CORRECT * sim.dt)) ** 2)))

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
    hdr = (f"  {'Scheme':<18} {'N':>4}  {'Compile':>9}  {'Step':>10}  {'RHS':>10}  "
           f"{'Mpts/s':>8}  {f'E2E(N={N_REPORT_STEPS})':>13}  "
           f"{'StateMB':>8}  {'L2 err':>10}")
    sep = "-" * len(hdr)
    print(f"\n{'='*len(hdr)}")
    print(f"  MCS 3D Benchmark — scan-based timings")
    print(f"  Device   : {gpu['device']}  ({gpu['backend']})")
    if gpu["vram_gb"]: print(f"  VRAM     : {gpu['vram_gb']} GB")
    print(f"  Method   : each step time = (jit_scan_{N_SCAN_STEPS}_steps) / {N_SCAN_STEPS}")
    print(f"{'='*len(hdr)}")
    print(hdr); print(sep)
    for r in rows:
        print(f"  {r['scheme']:<18} {r['nx']:>4}  "
              f"  {_f(r['compile_s'],'7.2f')}s  "
              f"  {_f(r['per_step_us'],'7.1f')}us  "
              f"  {_f(r['rhs_per_step_us'],'7.1f')}us  "
              f"  {_f(r['throughput_Mpts_s'],'6.1f')}  "
              f"  {_f(r[e2e_key],'9.2f')}s   "
              f"  {_f(r['state_MB'],'6.1f')}  "
              f"  {_f(r['l2_err'],'.2e')}")
    print(sep)
    print(f"  {'HLO cost (UB)':<18} {'N':>4}  {'rhsFLOP/B':>10}  {'GFLOP/s*':>10}  "
          f"{'GB/s*':>9}  {'%peakBW*':>9}  {'bound':>8}")
    for r in rows:
        print(f"  {r['scheme']:<18} {r['nx']:>4}  "
              f"  {_f(r.get('rhs_flop_per_byte'),'8.2f')}  "
              f"  {_f(r.get('hlo_GFLOPs'),'8.0f')}  "
              f"  {_f(r.get('hlo_GBs'),'7.0f')}  "
              f"  {_f(r.get('hlo_bytes_pct_peakBW'),'7.1f')}  "
              f"  {str(r.get('bound','?')):>8}")
    print(f"  * cost-model upper bound (incl. on-chip halo); NOT DRAM — use "
          f"`profile_regime.py --smi` for the real regime")
    print(sep)


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n>> CSV  →  {path}")


_COLORS  = {"floating_point": "#2196F3"}
_MARKERS = {"floating_point": "o"}


def _color(scheme):
    return _COLORS.get(scheme, "#607D8B")


def _marker(scheme):
    return _MARKERS.get(scheme, "o")


def save_throughput_plot(rows, path, gpu):
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
    ax0.set(xlabel="Grid N (= Nx = Ny = Nz)", ylabel="Throughput (Mpts/s)",
            title="Throughput vs Grid Size")
    ax0.legend(); ax0.grid(True, which="both", alpha=0.3)
    ax1.set(xlabel="Grid N", ylabel="Per-step time (µs)",
            title="Steady-state per-step time")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"MCS 3D — scan-based timings  |  {gpu['device']}", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f">> Plot  →  {path}")


def save_roofline_plot(rows, path, gpu):
    """Roofline: HBM/FP64/INT8 ceilings + RHS arithmetic intensity.  The plotted
    height (hlo_GFLOPs) is a cost-model UPPER BOUND (incl. on-chip halo), drawn
    hollow; the true point needs HW counters (locked here)."""
    peak_bw = gpu.get("peak_bw_GBs")
    peak_fp64 = gpu.get("peak_fp64_GFLOPs")
    peak_int8 = gpu.get("peak_int8_TOPs")
    pts = [(r.get("rhs_flop_per_byte") or r.get("step_flop_per_byte"),
            r.get("hlo_GFLOPs"), r) for r in rows]
    pts = [(x, y, r) for (x, y, r) in pts if x and y]
    if not (peak_bw and peak_fp64 and pts):
        raise ValueError("roofline needs peak BW + FP64 ceiling + >=1 point")
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.logspace(-1, 3.2, 200)
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
        ax.plot(x, y, "o", mfc="none", mec=_color(r["scheme"]), mew=1.6, ms=11)
        ax.annotate(f"{r['scheme']} {r['nx']}³  (HLO-UB)", (x, y), fontsize=7,
                    ha="left", va="top", xytext=(8, -6 - 12 * i),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)  —  RHS, single-pass")
    ax.set_ylabel("performance  (GFLOP/s · GOP/s INT8)")
    ax.set_title(f"Roofline — {gpu.get('device','')}  (MCS 3D FD step)")
    ax.text(0.02, 0.02, "○ height = cost-model upper bound (incl. on-chip halo)",
            transform=ax.transAxes, fontsize=6.5, color="grey", va="bottom")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f">> Plot  →  {path}")


def _add_bandwidth(rows, gpu):
    peak = gpu.get("peak_bw_GBs")
    for r in rows:
        ach = r.get("hlo_GBs")
        r["hlo_bytes_pct_peakBW"] = (round(100.0 * ach / peak, 1)
                                     if peak and ach else None)
    return rows


def save_json(rows, path, gpu):
    import json
    with open(path, "w") as f:
        json.dump({"device": gpu.get("device"), "backend": gpu.get("backend"),
                   "vram_gb": gpu.get("vram_gb"),
                   "peak_bw_GBs": gpu.get("peak_bw_GBs"),
                   "peak_fp64_GFLOPs": gpu.get("peak_fp64_GFLOPs"),
                   "peak_int8_TOPs": gpu.get("peak_int8_TOPs"),
                   "grids": GRID_SIZES, "schemes": SCHEMES,
                   "n_scan_steps": N_SCAN_STEPS, "rows": rows}, f, indent=2)
    print(f">> JSON  →  {path}")


def _render_plots(rows, out_dir, gpu):
    plots = [
        ("benchmark_throughput.png", save_throughput_plot),
        ("benchmark_roofline.png",   save_roofline_plot),
    ]
    for name, fn in plots:
        try:
            fn(rows, os.path.join(out_dir, name), gpu)
        except Exception as exc:
            print(f">> Plot FAILED ({name}): {exc}  [data is safe in CSV/JSON]")


def replot(json_path, out_dir=None):
    import json
    with open(json_path) as f:
        blob = json.load(f)
    rows = blob["rows"]
    gpu = {"device": blob.get("device"), "backend": blob.get("backend"),
           "vram_gb": blob.get("vram_gb")}
    dev = gpu["device"] or ""
    gpu["peak_bw_GBs"] = blob.get("peak_bw_GBs") or _match_table(dev, _BW_TABLE)
    comp = _match_table(dev, _COMPUTE_TABLE) or (None, None)
    gpu["peak_fp64_GFLOPs"] = blob.get("peak_fp64_GFLOPs") or comp[0]
    gpu["peak_int8_TOPs"]   = blob.get("peak_int8_TOPs")   or comp[1]
    out_dir = out_dir or str(Path(json_path).resolve().parent)
    os.makedirs(out_dir, exist_ok=True)
    _add_bandwidth(rows, gpu)
    _render_plots(rows, out_dir, gpu)
    print(f">> Replotted from {json_path} into {out_dir}/")


def _find_params():
    """Locate 3D/params.toml robustly.

    `__file__`-relative resolution (`parents[2]`) only works for an in-tree
    editable install; a copying/site-packages install puts the package under
    `.venv/.../site-packages/mcs3d/`, where `parents[2]` lands in the venv, not
    the repo.  So try the in-tree path first, then CWD-relative candidates (run
    from the repo root or from 3D/).  Returns the first that exists, else the
    in-tree path (load_parameters then warns + uses its defaults dict)."""
    candidates = [
        Path(__file__).resolve().parents[2] / "params.toml",  # in-tree editable
        Path.cwd() / "3D" / "params.toml",                    # run from repo root
        Path.cwd() / "params.toml",                           # run from 3D/
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main():
    parfile = sys.argv[1] if len(sys.argv) > 1 else str(_find_params())
    # Default output next to the params file (i.e. 3D/src/mcs3d/output), so it is
    # in the repo tree where `sync.sh pull results` looks -- NOT in site-packages.
    default_out = str(Path(parfile).resolve().parent / "src" / "mcs3d" / "output")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_out
    os.makedirs(out_dir, exist_ok=True)

    base_params = load_parameters(parfile)
    gpu = gpu_info()

    print(f"\n{'='*60}")
    print(f"  MCS 3D Scheme Benchmark (scan-based)")
    print(f"  Device  : {gpu['device']}  ({gpu['backend']})")
    if gpu["peak_bw_GBs"]:
        print(f"  Peak BW : {gpu['peak_bw_GBs']/1000:.1f} TB/s")
    else:
        print(f"  Peak BW : UNKNOWN (device '{gpu['device']}' not in table; "
              f"roofline %peakBW will be N/A)")
    print(f"  Grids   : {GRID_SIZES}   Schemes : {SCHEMES}")
    print(f"{'='*60}")

    rows = []
    e2e_key = f"end_to_end_s_N{N_REPORT_STEPS}"
    total = len(GRID_SIZES) * len(SCHEMES); done = 0
    for nx in GRID_SIZES:
        for scheme in SCHEMES:
            done += 1
            print(f"\n[{done}/{total}]  {scheme:<18}  {nx}³ ...", flush=True)
            try:
                row = measure(scheme, nx, base_params)
                rows.append(row)
                print(f"         compile={row['compile_s']:.2f}s  "
                      f"step={row['per_step_us']:.1f}us  "
                      f"E2E={row[e2e_key]:.1f}s  L2={row['l2_err']:.2e}")
            except Exception as exc:
                print(f"         FAILED: {exc}")

    if rows:
        _add_bandwidth(rows, gpu)
        print_table(rows, gpu)
        save_csv(rows,  os.path.join(out_dir, "benchmark_results.csv"))
        save_json(rows, os.path.join(out_dir, "benchmark_results.json"), gpu)
        _render_plots(rows, out_dir, gpu)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        replot(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        main()
