"""
profile_ozaki.py — the one command that tells us everything about the Pallas
INT8 / BFP48 Ozaki RHS.

It answers, in order of importance:

  1. WALL-CLOCK  — is `pallas_ozaki` actually faster than `fused_floating_point`
     (the XLA-auto-fused FP64 baseline)?  Per-RHS, at production grids.

  2. THE AMDAHL WALL  — within the Ozaki kernel, how is time split across
        residue-conversion  /  int8-GEMM  /  Garner-CRT  ?
     This is THE number.  int8-GEMM rides the hardware curve (gets faster on
     every new GPU); the RNS residue-conversion and CRT reconstruction do NOT.
     If the GEMM dominates, the scheme gets faster for free on future hardware;
     if the conversion+CRT dominate, no amount of int8 progress saves it.
     Measured by build-time truncation of the kernel pipeline (profile_trunc).

  3. ROOFLINE  — compute- vs memory-bound (XLA FLOP/byte; note the Pallas custom
     call is partly opaque to the cost model — see the caveat printed inline).

  4. MODULI SCALING  — time vs k_full (vary the RNS moduli): slope = per-modulus
     cost, intercept = fixed (load + unpack + PDE assembly) cost.

  5. MEMORY TRANSFER  — BFP48 (3×int16 = 6 B/cell) vs fp64 (8 B/cell) per RHS.

  6. --trace  — emit a JAX profiler trace so you can confirm int8 IMMA/WGMMA
     kernels actually fire on the GPU (open in chrome://tracing / perfetto).

The profiler times the RAW RHS kernels (make_*_rhs), not the full RK4/BC stack,
so it isolates exactly the kernel under study.

Usage:
    python mcs2d/profile_ozaki.py [--grids 256 512] [--reps 30]
                                  [--stage-grid 256] [--extras] [--trace]
                                  [params.toml] [output_dir]

`--extras` also profiles fused_ozaki for context.
"""

import os
import sys
import csv
import json
import time
import argparse
import statistics
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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

# Reuse the benchmark's device/cost helpers (single source of truth).
from mcs2d.benchmark import gpu_info, _rhs_cost, _device_peak_mem_mb
from mcs2d.main import load_parameters
from mcs2d.schemes import pallas_ozaki as POZ

NF = 10
# Theoretical dense throughput for the headline GPUs (TOPS int8 / TFLOPS fp64),
# used only to print the int8/fp64 ratio that motivates the scheme.
_TOPS = {"H200": (1979.0, 67.0), "H100": (1979.0, 67.0), "A100": (624.0, 19.5)}


# ── RHS builders (raw kernels, isolated from RK4/BC) ─────────────────────────────

def _pde_args(nx, params, ko_override=None):
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / nx
    cs = params.get("enable_cs", 1.0)
    L  = params.get("Lambda", 0.1)
    K1 = params.get("K1", 100.0)
    K2 = params.get("K2", 100.0)
    ko = params.get("ko_sigma", 0.05) if ko_override is None else ko_override
    return (nx, nx, dx, dy, cs, L, K1, K2, ko)


def build_rhs(scheme, nx, params, mods_ext=None, profile_trunc=None, ko_override=None):
    """Return a raw RHS fn  data:(NF,nx,nx) → (NF,nx,nx)  for `scheme`.

    `ko_override`/`mods_ext` shrink the (huge, fully-unrolled) Ozaki kernel for a
    fast Triton compile: ko_override=0.0 drops the 20 KO derivative calls (~half
    the kernel); fewer ext moduli cut the GEMM/Garner count."""
    args = _pde_args(nx, params, ko_override=ko_override)
    if scheme == "pallas_ozaki":
        return POZ.make_pallas_ozaki_rhs(*args, mods_ext=mods_ext,
                                         profile_trunc=profile_trunc)
    if scheme == "fused_floating_point":
        from mcs2d.schemes.fused_rhs_fp import make_fused_rhs
        return make_fused_rhs(*args)
    if scheme == "fused_ozaki":
        from mcs2d.schemes.fused_rhs_ozaki import make_fused_ozaki_rhs
        return make_fused_ozaki_rhs(*args, mods_ext=mods_ext)
    raise ValueError(f"unknown scheme {scheme}")


# ── Timing ───────────────────────────────────────────────────────────────────────

def time_call(fn, data, reps, warmup=3):
    """Median wall time (µs) of one jitted `fn(data)`, GPU-synced.

    Single application per call (no output→input feedback), so values stay
    bounded and XLA can't hoist; launch overhead is included but negligible
    against a full-grid kernel.  Returns (median_us, compile_s)."""
    f = jax.jit(fn)
    t0 = time.perf_counter()
    jax.block_until_ready(f(data))
    compile_s = time.perf_counter() - t0
    for _ in range(warmup - 1):
        jax.block_until_ready(f(data))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(data))
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), compile_s


def _rand(nx, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((NF, nx, nx)))


# ── Staged compile diagnostic (localise a compile hang) ──────────────────────────

def _trivial_int8_pallas():
    """Compile+run a minimal int8 Pallas MMA — confirms int8 IMMA lowers at all,
    independent of the Ozaki kernel."""
    from jax.experimental import pallas as pl
    M = K = N = 32
    a = jnp.ones((M, K), jnp.int8)
    b = jnp.ones((K, N), jnp.int8)
    def k(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.int32)
    fn = pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct((M, N), jnp.int32))
    jax.block_until_ready(jax.jit(fn)(a, b))


def _int8_gemm_probe(n_gemms, M, K, N):
    """Compile a kernel with `n_gemms` int8 GEMMs at OUR shapes, accumulated.
    Maps int8-GEMM count → compile time, isolating it from residues/Garner —
    so we can tell whether step 2's hang is GEMM COUNT (→ must stop unrolling)
    or GEMM SHAPE (M=16 → pad fix)."""
    from jax.experimental import pallas as pl
    a = jnp.ones((M, K), jnp.int8)
    b = jnp.ones((K, N), jnp.int8)
    def k(a_ref, b_ref, o_ref):
        acc = jnp.zeros((M, N), jnp.int32)
        for _ in range(n_gemms):
            acc = acc + jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.int32)
        o_ref[...] = acc
    fn = pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct((M, N), jnp.int32))
    jax.block_until_ready(jax.jit(fn)(a, b))


def _int8_gemm_distinct(n, M, K, N):
    """n int8 GEMMs with DISTINCT operands (ref-indexed → no CSE) — the realistic
    case the previous probe hid by reusing one (a,b)."""
    from jax.experimental import pallas as pl
    A = (jnp.arange(n * M * K, dtype=jnp.int32) % 7 - 3).astype(jnp.int8).reshape(n, M, K)
    b = jnp.ones((K, N), jnp.int8)
    def kern(a_ref, b_ref, o_ref):
        acc = jnp.zeros((M, N), jnp.int32)
        for i in range(n):
            acc = acc + jnp.dot(a_ref[i], b_ref[...], preferred_element_type=jnp.int32)
        o_ref[...] = acc
    fn = pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((M, N), jnp.int32))
    jax.block_until_ready(jax.jit(fn)(A, b))


def run_diagnose(params, nx=64):
    """Compile (block_until_ready) progressively larger pallas_ozaki kernels so we
    can see WHICH stage explodes/hangs.  Watch the prints; Ctrl-C on the hang."""
    data = _rand(nx, seed=7)
    print(f"\n[diagnose] staged compile @ {nx}² — watch which line never finishes "
          f"(Ctrl-C there):")

    print("  [0] trivial int8 Pallas MMA (32³) … ", end="", flush=True)
    try:
        t0 = time.perf_counter(); _trivial_int8_pallas()
        print(f"OK ({time.perf_counter()-t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:160]}")

    # int8-GEMM count scaling at OUR exact shapes (BS=16,H_PAD=32).  If 1 & 8
    # compile fast but ~120 hangs → it's COUNT (unrolling), not shape; if even a
    # single (16,32)x(32,32) hangs → it's the M=16 SHAPE.
    H = POZ.H_PAD
    for shape in [(POZ.BS, H, H), (H, H, POZ.BS)]:   # axis-0 and axis-1 GEMM shapes
        for n in (1, 8, 40, 120):
            print(f"  [g] {n:>3}× int8 GEMM {shape[0]}×{shape[1]}·{shape[1]}×{shape[2]} … ",
                  end="", flush=True)
            try:
                t0 = time.perf_counter(); _int8_gemm_probe(n, *shape)
                print(f"OK ({time.perf_counter()-t0:.1f}s)")
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {str(e)[:120]}")

    # DISTINCT-operand GEMMs (no CSE) — the realistic case.  If this blows up at
    # ~120 while the reused-operand probe above didn't, distinct-MMA count is the
    # wall (→ batch the modulus loop into ONE dot).
    for n in (8, 40, 120):
        print(f"  [gd] {n:>3}× DISTINCT int8 GEMM 16×32·32×32 … ", end="", flush=True)
        try:
            t0 = time.perf_counter(); _int8_gemm_distinct(n, POZ.BS, POZ.H_PAD, POZ.H_PAD)
            print(f"OK ({time.perf_counter()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:120]}")

    # Each row is a strictly larger slice of the real kernel.
    configs = [
        ("1 trunc0 ko0 k6  (limbs+crop, NO gemm/garner)",
         dict(profile_trunc=0, ko_override=0.0, mods_ext=[])),
        ("1b trunc1 ko0 k6 (+ residues, NO gemm)",
         dict(profile_trunc=1, ko_override=0.0, mods_ext=[])),
        ("2 trunc2 ko0 k6  (+ int8 GEMM + bias + mod)",
         dict(profile_trunc=2, ko_override=0.0, mods_ext=[])),
        ("3 full   ko0 k6  (+ Garner CRT; full math, no KO)",
         dict(profile_trunc=None, ko_override=0.0, mods_ext=[])),
        ("4 full   ko0 k8  (+ 2 moduli)",
         dict(profile_trunc=None, ko_override=0.0, mods_ext=[239, 233])),
        ("5 full   ko-ON k8 (FULL production kernel)",
         dict(profile_trunc=None, ko_override=None, mods_ext=[239, 233])),
    ]
    for name, kw in configs:
        print(f"  [{name}] compiling … ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            fn = build_rhs("pallas_ozaki", nx, params, **kw)
            jax.block_until_ready(jax.jit(fn)(data))
            print(f"OK ({time.perf_counter()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:160]}")


# ── 1. Wall-clock scheme comparison ──────────────────────────────────────────────

def run_wallclock(schemes, grids, params, reps, mods_ext=None, ko_override=None):
    rows = []
    for nx in grids:
        data = _rand(nx)
        base = None
        for s in schemes:
            try:
                # mods_ext/ko_override only shrink the Ozaki kernel; FP schemes
                # ignore them.
                kw = ({} if s == "fused_floating_point"
                      else {"mods_ext": mods_ext, "ko_override": ko_override})
                print(f"    {s:<22} {nx}²  compiling… "
                      f"(Triton compile of the unrolled Ozaki kernel can be slow)",
                      flush=True)
                fn = build_rhs(s, nx, params, **kw)
                us, comp = time_call(fn, data, reps)
                cost = _rhs_cost(fn, data)
                row = {"scheme": s, "nx": nx, "rhs_us": round(us, 2),
                       "compile_s": round(comp, 2), **cost}
                if s == "fused_floating_point":
                    base = us
                rows.append(row)
                print(f"    {s:<22} {nx}²  rhs={us:8.1f}µs  compile={comp:6.2f}s  "
                      f"{cost.get('rhs_flop_per_byte','?')} FLOP/B ({cost.get('bound','?')})")
            except Exception as exc:
                print(f"    {s:<22} {nx}²  FAILED: {exc}")
        # speedup vs fused_floating_point baseline
        for row in rows:
            if row["nx"] == nx and base:
                row["speedup_vs_fused_fp"] = round(base / row["rhs_us"], 3)
    return rows


# ── 2. The Amdahl wall: in-kernel stage breakdown ────────────────────────────────

def run_stage_breakdown(nx, params, reps, mods_ext=None, ko_override=None):
    """Truncate the Ozaki kernel at each pipeline stage; attribute time by diff.

    T0 = limb load + unpack (+ PDE assembly, constant across all)
    T1 = + residue conversion        → residue = T1 − T0
    T2 = + int8 GEMM + bias + reduce → gemm    = T2 − T1
    full = + Garner CRT + unscale    → crt     = full − T2
    """
    data = _rand(nx, seed=1)
    levels = [("T0_load",     0),
              ("T1_residues", 1),
              ("T2_gemm",     2),
              ("full",     None)]
    t = {}
    for name, trunc in levels:
        fn = build_rhs("pallas_ozaki", nx, params, mods_ext=mods_ext,
                       profile_trunc=trunc, ko_override=ko_override)
        us, _ = time_call(fn, data, reps)
        t[name] = us
        print(f"    {name:<14} {us:9.1f} µs")

    load     = t["T0_load"]
    residue  = max(0.0, t["T1_residues"] - t["T0_load"])
    gemm     = max(0.0, t["T2_gemm"]     - t["T1_residues"])
    crt      = max(0.0, t["full"]        - t["T2_gemm"])
    total    = t["full"]
    rns      = residue + gemm + crt        # the RNS pipeline (excludes load+assembly)

    def pct(x, d):  return round(100.0 * x / d, 1) if d > 0 else None
    out = {
        "nx": nx,
        "us": {k: round(v, 2) for k, v in t.items()},
        "stage_us": {"load_assembly": round(load, 2), "residue": round(residue, 2),
                     "gemm": round(gemm, 2), "crt": round(crt, 2)},
        "pct_of_total": {"load_assembly": pct(load, total), "residue": pct(residue, total),
                         "gemm": pct(gemm, total), "crt": pct(crt, total)},
        # THE number: GEMM share of the RNS pipeline (the part that rides int8 HW).
        "gemm_pct_of_rns": pct(gemm, rns),
        "gemm_pct_of_total": pct(gemm, total),
        "hw_curve_pct": pct(gemm, total),                 # rides int8 hardware curve
        "fixed_overhead_pct": pct(load + residue + crt, total),  # does NOT
    }
    return out


# ── 4. Moduli scaling (per-modulus cost) ─────────────────────────────────────────

def run_moduli_scaling(nx, params, reps, ko_override=None):
    """Vary k_full via mods_ext; time tells per-modulus slope + fixed intercept."""
    data = _rand(nx, seed=2)
    # base has 6 moduli; ext adds 0..3.  Defaults from CrtFloatConverter.
    variants = {6: [], 7: [239], 8: [239, 233], 9: [239, 233, 229]}
    pts = {}
    for k, ext in variants.items():
        try:
            print(f"    k_full={k}  compiling…", flush=True)
            fn = build_rhs("pallas_ozaki", nx, params, mods_ext=ext,
                           ko_override=ko_override)
            us, _ = time_call(fn, data, reps)
            pts[k] = round(us, 2)
            print(f"    k_full={k}  rhs={us:9.1f} µs")
        except Exception as exc:
            print(f"    k_full={k}  FAILED: {exc}")
    slope = intercept = None
    if len(pts) >= 2:
        ks = np.array(sorted(pts)); ys = np.array([pts[k] for k in ks])
        slope, intercept = np.polyfit(ks, ys, 1)
    return {"nx": nx, "us_by_k": pts,
            "us_per_modulus": round(float(slope), 2) if slope is not None else None,
            "fixed_us": round(float(intercept), 2) if intercept is not None else None}


# ── 5. Memory-transfer accounting ────────────────────────────────────────────────

def mem_accounting(grids):
    """BFP48 (6 B/cell) vs fp64 (8 B/cell) kernel-input transfer per RHS."""
    out = []
    for nx in grids:
        bs, ng = POZ.BS, POZ.NG
        nbx = (nx + (bs - nx % bs) % bs) // bs
        n_patches = nbx * nbx
        cells = n_patches * NF * (POZ.H_ORIG ** 2)     # real (haloed) cells read
        fp64_mb  = cells * 8 / 1e6
        bfp48_mb = cells * 6 / 1e6
        out.append({"nx": nx, "n_patches": n_patches,
                    "fp64_input_MB": round(fp64_mb, 2),
                    "bfp48_input_MB": round(bfp48_mb, 2),
                    "transfer_saving_pct": 25.0})
    return out


# ── 6. Optional profiler trace ───────────────────────────────────────────────────

def run_trace(nx, params, out_dir, reps=8):
    tdir = os.path.join(out_dir, "trace")
    os.makedirs(tdir, exist_ok=True)
    data = _rand(nx, seed=3)
    fns = {"pallas_ozaki": jax.jit(build_rhs("pallas_ozaki", nx, params)),
           "fused_floating_point": jax.jit(build_rhs("fused_floating_point", nx, params))}
    for f in fns.values():           # compile out of the trace
        jax.block_until_ready(f(data))
    jax.profiler.start_trace(tdir)
    for _ in range(reps):
        for f in fns.values():
            jax.block_until_ready(f(data))
    jax.profiler.stop_trace()
    print(f">> Trace →  {tdir}  (open the .xplane.pb in TensorBoard / perfetto;"
          f" look for int8 'mma'/'wgmma'/'igemm' kernels in pallas_ozaki)")


# ── Reporting ────────────────────────────────────────────────────────────────────

def _bar(pct, width=40):
    n = int(round((pct or 0) / 100 * width))
    return "█" * n + "·" * (width - n)


def report(results, gpu):
    sb = results["stage_breakdown"]
    print("\n" + "=" * 72)
    print("  OZAKI PROFILE — summary")
    print(f"  Device : {gpu['device']}  ({gpu['backend']})")
    if gpu["backend"] == "cpu":
        print("  ⚠️  CPU/interpret mode: the int8 GEMM runs as an ordinary CPU"
              " matmul — NO tensor cores.\n      The Amdahl split below is"
              " MEANINGLESS here; it only answers the question on a GPU (H200).")
    dev = gpu["device"]
    for k, (i8, f64) in _TOPS.items():
        if k.lower() in dev.lower():
            print(f"  int8/fp64 dense throughput ratio ≈ {i8/f64:.0f}×  "
                  f"({i8:.0f} TOPS int8 / {f64:.0f} TFLOPS fp64)")
            break
    print("=" * 72)

    print("\n  [1] WALL-CLOCK (raw RHS, vs fused_floating_point):")
    for r in results["wallclock"]:
        sp = r.get("speedup_vs_fused_fp")
        print(f"      {r['scheme']:<22} {r['nx']}²  {r['rhs_us']:8.1f}µs"
              f"  {('%.2f×'%sp) if sp else '   —':>8} vs fused_fp")

    if sb.get("skipped"):
        print("\n  [2] AMDAHL WALL — skipped (--quick).  Re-run without --quick "
              "for the GEMM/residue/CRT split.")
    else:
        print(f"\n  [2] AMDAHL WALL — in-kernel split @ {sb['nx']}²:")
        for stage, key in [("load+assembly", "load_assembly"), ("residue-conv", "residue"),
                           ("int8 GEMM", "gemm"), ("Garner CRT", "crt")]:
            p = sb["pct_of_total"][key]
            print(f"      {stage:<14} {sb['stage_us'][key]:8.1f}µs  {_bar(p)} {p:5.1f}%")
        print(f"      {'─'*60}")
        print(f"      GEMM share of RNS pipeline : {sb['gemm_pct_of_rns']:5.1f}%  "
              f"(the part that rides the int8 hardware curve)")
        print(f"      Rides hardware curve       : {sb['hw_curve_pct']:5.1f}% of kernel")
        print(f"      Fixed RNS overhead         : {sb['fixed_overhead_pct']:5.1f}% of kernel"
              f"  (does NOT improve on new GPUs)")
        verdict = ("GEMM-dominated → int8 progress keeps paying off"
                   if (sb["gemm_pct_of_rns"] or 0) >= 50 else
                   "overhead-dominated → faster int8 won't save it (Amdahl); "
                   "optimize residue/CRT or reconsider")
        print(f"      VERDICT: {verdict}")

    print(f"\n  [4] MODULI SCALING @ {results['moduli']['nx']}²: "
          f"{results['moduli']['us_per_modulus']}µs/modulus + "
          f"{results['moduli']['fixed_us']}µs fixed")
    print(f"      {results['moduli']['us_by_k']}")

    print("\n  [5] MEMORY TRANSFER (kernel input/RHS):")
    for m in results["memory"]:
        print(f"      {m['nx']}²  fp64={m['fp64_input_MB']:8.2f}MB  "
              f"bfp48={m['bfp48_input_MB']:8.2f}MB  (−{m['transfer_saving_pct']:.0f}%)")
    print("      NOTE: realized only when BFP48 is the CANONICAL stored state;"
          " as wired, the wrapper still reads fp64 to pack each RHS call.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--stage-grid", type=int, default=None,
                    help="grid for the Amdahl/moduli passes (default: smallest)")
    ap.add_argument("--extras", action="store_true",
                    help="also profile fused_ozaki")
    ap.add_argument("--trace", action="store_true",
                    help="emit a JAX profiler trace (confirm int8 MMA kernels)")
    ap.add_argument("--diagnose", action="store_true",
                    help="staged compile: compile progressively larger pallas_ozaki "
                         "kernels to localise a compile hang (watch which stage stalls)")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke: tiny grid, KO off, 6 moduli, wall-clock only "
                         "— just confirms pallas_ozaki COMPILES + runs on the GPU")
    ap.add_argument("--no-ko", action="store_true",
                    help="drop KO derivatives (halves the Ozaki kernel → faster compile)")
    ap.add_argument("--mods-ext", type=int, default=None,
                    help="number of extension moduli (0→k=6 … 3→k=9; default 2→k=8). "
                         "Fewer = smaller kernel = faster compile.")
    ap.add_argument("parfile", nargs="?",
                    default=str(Path(__file__).resolve().parents[2] / "params.toml"))
    ap.add_argument("out_dir", nargs="?", default=str(Path(_dir) / "output"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.quick:
        a.grids = [64]
        a.reps = min(a.reps, 5)
        a.no_ko = True
        if a.mods_ext is None:
            a.mods_ext = 0

    # ext-moduli override: None→default (k=8); else take the first N of [239,233,229].
    mods_ext = None if a.mods_ext is None else [239, 233, 229][:a.mods_ext]
    ko_override = 0.0 if a.no_ko else None

    params = load_parameters(a.parfile)
    gpu = gpu_info()

    if a.diagnose:
        print(f"\n{'='*60}\n  OZAKI COMPILE DIAGNOSTIC\n  Device : {gpu['device']} "
              f"({gpu['backend']})\n{'='*60}")
        run_diagnose(params)
        return

    sgrid = a.stage_grid or min(a.grids)
    schemes = ["pallas_ozaki", "fused_floating_point"]
    if a.extras:
        schemes += ["fused_ozaki"]

    print(f"\n{'='*60}\n  OZAKI PROFILER\n  Device : {gpu['device']} ({gpu['backend']})")
    if gpu["vram_gb"]:
        print(f"  VRAM   : {gpu['vram_gb']} GB")
    print(f"  Grids  : {a.grids}   stage/moduli grid: {sgrid}   reps: {a.reps}")
    print(f"  ext-moduli: {mods_ext if mods_ext is not None else 'default(k=8)'}  "
          f"KO: {'OFF' if a.no_ko else 'on'}  quick: {a.quick}")
    print(f"  int8 hardcoded ON: {POZ.USE_INT8_GEMM} | pad_pow2: {POZ._PAD_POW2} "
          f"(H {POZ.H_ORIG}→{POZ.H_PAD}, NF {POZ.NF}→{POZ.NF_PAD})\n{'='*60}")

    print("\n[1] Wall-clock ...")
    wallclock = run_wallclock(schemes, a.grids, params, a.reps,
                              mods_ext=mods_ext, ko_override=ko_override)
    if a.quick:
        # Quick mode: wall-clock only — the point is just "does it compile + run?"
        print("\n[quick] skipping Amdahl/moduli passes (each is another full "
              "compile).  Drop --quick for the full profile once this works.")
        stage_breakdown = {"nx": sgrid, "stage_us": {}, "pct_of_total": {},
                           "gemm_pct_of_rns": None, "gemm_pct_of_total": None,
                           "hw_curve_pct": None, "fixed_overhead_pct": None,
                           "skipped": True}
        moduli = {"nx": sgrid, "us_by_k": {}, "us_per_modulus": None, "fixed_us": None}
    else:
        print(f"\n[2] Amdahl wall (stage breakdown) @ {sgrid}² ...")
        stage_breakdown = run_stage_breakdown(sgrid, params, a.reps,
                                              mods_ext=mods_ext, ko_override=ko_override)
        print(f"\n[4] Moduli scaling @ {sgrid}² ...")
        moduli = run_moduli_scaling(sgrid, params, a.reps, ko_override=ko_override)
    memory = mem_accounting(a.grids)

    results = {"device": gpu["device"], "backend": gpu["backend"],
               "int8_on": bool(POZ.USE_INT8_GEMM), "pad_pow2": bool(POZ._PAD_POW2),
               "H_orig": POZ.H_ORIG, "H_pad": POZ.H_PAD,
               "wallclock": wallclock, "stage_breakdown": stage_breakdown,
               "moduli": moduli, "memory": memory}

    report(results, gpu)

    jpath = os.path.join(a.out_dir, "profile_ozaki.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n>> JSON  →  {jpath}")
    save_stage_plot(stage_breakdown, os.path.join(a.out_dir, "profile_ozaki_stages.png"), gpu)

    if a.trace:
        print("\n[6] Profiler trace ...")
        run_trace(sgrid, params, a.out_dir)


def save_stage_plot(sb, path, gpu):
    if sb.get("skipped") or not sb.get("stage_us"):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = ["load+\nassembly", "residue\nconv", "int8\nGEMM", "Garner\nCRT"]
    keys = ["load_assembly", "residue", "gemm", "crt"]
    vals = [sb["stage_us"][k] for k in keys]
    colors = ["#9E9E9E", "#FF9800", "#4CAF50", "#E91E63"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars, fmt="%.0fµs", padding=2)
    ax.set_ylabel("time per RHS (µs)")
    ax.set_title(f"Ozaki kernel stage breakdown @ {sb['nx']}²  |  {gpu['device']}\n"
                 f"GEMM = {sb['gemm_pct_of_total']}% of kernel "
                 f"({sb['gemm_pct_of_rns']}% of RNS) — rides int8 HW curve")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f">> Plot  →  {path}")


if __name__ == "__main__":
    main()
