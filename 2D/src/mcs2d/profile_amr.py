"""
Advanced GPU profiler for the AMR step.

Goal: the MOST data possible, attributable down to individual building-block
functions, so we can see exactly what costs time and whether each piece is
compute- or memory-bound.

For every "phase" (the full step + each building-block kernel in isolation) it
collects, in one run:

  * timing      — median per-call µs over `--repeats` (steady-state, robust)
  * trace       — its OWN jax.profiler.trace (own XLA module) → per-op-family
                  GPU-time breakdown (concatenate / select / add / gather / …),
                  kernel COUNT and average kernel µs (latency- vs throughput-bound),
                  and GPU-busy time
  * cost model  — XLA HLO cost_analysis: FLOPs, bytes accessed, and arithmetic
                  intensity (FLOP/byte) → compute-bound vs memory-bound
  * memory      — device peak / in-use bytes for the phase

Building-block phases (function-level attribution):
  full_step, rhs, rhs_no_ko (isolates KO), sync_root (periodic within-level),
  sync_within (5.1 same-level sibling sync), sync_cross (cross-level prolongation),
  restrict (6th-order restriction).

Dynamic regime (`--dynamic`): evolve a localized feature with periodic regrid and
record the time series the static profile can't show — per-chunk GPU step time,
per-regrid HOST time, per-level occupancy, slot fragmentation, cap-growth, and
recompiles.  This is the evidence base for the compaction (4.6) go/no-go.

Outputs: a rich JSON (`--out`) and a human-readable Markdown report (`--report`),
plus the per-phase Perfetto traces (open at https://ui.perfetto.dev).

Usage (on the supercomputer, after `./sync.sh push`)
----------------------------------------------------
    cd /home/bmh74/Integer-Schemes
    python 2D/src/mcs2d/profile_amr.py --nbx 4 --nby 4 --levels 4 \
        --out traces/amr/static.json --report traces/amr/static.md
    python 2D/src/mcs2d/profile_amr.py --nbx 4 --nby 4 --levels 4 --dynamic \
        --out traces/amr/dynamic.json --report traces/amr/dynamic.md
Then:  ./sync.sh pull traces
"""

import os
import glob
import gzip
import json
import time
import shutil
import argparse
import collections
import subprocess
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Small utilities ───────────────────────────────────────────────────────────

def _nvidia_smi(tag):
    if shutil.which("nvidia-smi") is None:
        print(f"   [{tag}] nvidia-smi unavailable (CPU run?)")
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        print(f"   [{tag}] {out}")
    except Exception:
        pass


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _bench_us(fn, fn_args, iters, repeats):
    """Steady-state per-call µs: dispatch `iters` calls, block once; median over
    `repeats` runs (robust to outliers / first-call jitter)."""
    import jax
    r = fn(*fn_args); jax.block_until_ready(r)          # compile
    per_call = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            r = fn(*fn_args)
        jax.block_until_ready(r)
        per_call.append((time.perf_counter() - t0) / iters * 1e6)
    return _median(per_call)


def _trace_phase(trace_dir, name, fn, fn_args, iters):
    """Profile one phase into its OWN subdir → its own XLA module, so the op-family
    breakdown is attributable to this phase alone.  Clears only this phase's subdir
    (never *.json or other phase dirs)."""
    import jax
    d = trace_dir / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    r = fn(*fn_args); jax.block_until_ready(r)          # ensure compiled first
    with jax.profiler.trace(str(d), create_perfetto_link=False):
        for _ in range(iters):
            r = fn(*fn_args)
        jax.block_until_ready(r)


_OP_KEYS = ["concatenate", "dynamic_update_slice", "dynamic_slice", "gather",
            "reduce", "select", "add", "transpose", "convert", "pad"]


def _classify(name):
    for k in _OP_KEYS:
        if k in name:
            return k
    if "copy" in name or "Memcpy" in name:
        return "copy/memcpy"
    return "other"


def _parse_phase_trace(phase_dir):
    """Aggregate the GPU kernels in a phase's Perfetto trace → op-family table
    (ms, %, count, avg-µs per family) + GPU-busy total + kernel count."""
    cands = sorted(glob.glob(str(Path(phase_dir) / "**" / "*.trace.json.gz"),
                             recursive=True))
    if not cands:
        return {}
    with gzip.open(cands[-1]) as fh:
        data = json.load(fh)
    ev = data.get("traceEvents", []) if isinstance(data, dict) else []
    # Find the GPU device process(es) by name (robust across nodes / JAX versions,
    # rather than hard-coding pid==1).  Fall back to any "/device:" stream.
    pid_name = {e["pid"]: e.get("args", {}).get("name", "")
                for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu_pids = {p for p, nm in pid_name.items() if "device:GPU" in nm}
    if not gpu_pids:
        gpu_pids = {p for p, nm in pid_name.items() if "/device:" in nm}
    gpu = [e for e in ev if e.get("pid") in gpu_pids
           and e.get("ph") == "X" and "dur" in e]
    if not gpu:
        return {}
    tot = sum(e["dur"] for e in gpu)
    us = collections.Counter(); n = collections.Counter()
    for e in gpu:
        c = _classify(e["name"]); us[c] += e["dur"]; n[c] += 1
    fam = {c: {"ms": round(us[c] / 1e3, 4),
               "pct": round(100 * us[c] / tot, 1) if tot else 0.0,
               "count": n[c],
               "avg_us": round(us[c] / n[c], 3)}
           for c in us}
    fam = dict(sorted(fam.items(), key=lambda kv: -kv[1]["ms"]))
    return {"gpu_busy_ms": round(tot / 1e3, 4), "kernel_count": len(gpu),
            "op_family": fam}


def _cost_analysis(fn, fn_args):
    """XLA HLO cost model: FLOPs, bytes accessed, arithmetic intensity (FLOP/byte)
    → compute-bound (high intensity) vs memory-bound (low).  Best-effort."""
    import jax
    try:
        ca = jax.jit(fn).lower(*fn_args).compile().cost_analysis()
    except Exception:
        return {}
    if ca is None:
        return {}
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else {}
    if not isinstance(ca, dict):
        return {}
    flops = float(ca.get("flops", 0.0) or 0.0)
    byts = float(ca.get("bytes accessed", 0.0) or 0.0)
    out = {"gflops": round(flops / 1e9, 4), "mbytes": round(byts / 1e6, 4)}
    if byts > 0:
        out["arith_intensity"] = round(flops / byts, 3)
        out["bound"] = "compute" if flops / byts > 10.0 else "memory"
    return out


def _mem_stats():
    """Device peak / in-use MB, if the backend reports it (GPU)."""
    import jax
    try:
        ms = jax.devices()[0].memory_stats() or {}
    except Exception:
        return {}
    if not ms:
        return {}
    return {"peak_mb": round(ms.get("peak_bytes_in_use", 0) / 1e6, 2),
            "in_use_mb": round(ms.get("bytes_in_use", 0) / 1e6, 2)}


def _frag(active_row):
    """Fragmentation of one level's active mask: (max_active_index+1)/n_active.
    1.0 = perfectly compact prefix; >1 = holes/scatter (what compaction fixes)."""
    import numpy as np
    idx = np.flatnonzero(np.asarray(active_row))
    if len(idx) == 0:
        return 0.0
    return round((int(idx[-1]) + 1) / len(idx), 3)


# ── Markdown report ────────────────────────────────────────────────────────────

def _write_report(path, results):
    L = []
    cfg = results["config"]
    L.append(f"# AMR profile — {results['device']} ({results['backend']})\n")
    L.append(f"- config: nbx×nby={cfg['nbx']}×{cfg['nby']}, LEVELS={cfg['levels']}, "
             f"BS={cfg['BS']}, NF={cfg['NF']}")
    L.append(f"- caps/level: {cfg['caps_per_level']}   active/level: {cfg['active_per_level']}")
    if "per_step_ms" in results:
        L.append(f"- **per-step: {results['per_step_ms']:.4f} ms**\n")

    if "phases" in results:
        L.append("\n## Per-phase summary\n")
        L.append("| phase | µs/call | GPU-busy ms | kernels | GFLOP | MB | FLOP/byte | bound |")
        L.append("|---|--:|--:|--:|--:|--:|--:|:--|")
        for name, p in results["phases"].items():
            c = p.get("cost", {})
            t = p.get("trace", {})
            L.append(f"| {name} | {p.get('us', 0.0):.1f} | "
                     f"{t.get('gpu_busy_ms', '-')} | {t.get('kernel_count', '-')} | "
                     f"{c.get('gflops', '-')} | {c.get('mbytes', '-')} | "
                     f"{c.get('arith_intensity', '-')} | {c.get('bound', '-')} |")
        L.append("\n## Per-phase op-family breakdown (GPU time)\n")
        for name, p in results["phases"].items():
            fam = p.get("trace", {}).get("op_family", {})
            if not fam:
                continue
            L.append(f"\n**{name}** (GPU-busy {p['trace']['gpu_busy_ms']} ms, "
                     f"{p['trace']['kernel_count']} kernels):\n")
            L.append("| op | ms | % | count | avg µs |")
            L.append("|---|--:|--:|--:|--:|")
            for op, d in fam.items():
                L.append(f"| {op} | {d['ms']} | {d['pct']} | {d['count']} | {d['avg_us']} |")

    if "dynamic" in results:
        dyn = results["dynamic"]
        L.append("\n## Dynamic regime (regrid churn)\n")
        L.append(f"- recompiles: **{dyn['recompiles']}** (expect 1 + cap-growths)")
        L.append(f"- cap-growth events: {dyn['cap_growths']}")
        L.append(f"- chunk GPU time ms: median {dyn['chunk_ms_median']:.3f}, "
                 f"max {dyn['chunk_ms_max']:.3f}")
        L.append(f"- regrid HOST time ms: median {dyn['regrid_ms_median']:.3f}, "
                 f"max {dyn['regrid_ms_max']:.3f}")
        L.append(f"- occupancy/level: min {dyn['occupancy_min']}, "
                 f"peak {dyn['occupancy_peak']}")
        L.append(f"- fragmentation/level (max over run): {dyn['frag_max']} "
                 f"(1.0=compact; >1 ⇒ compaction would help)")
        L.append("\n| regrid | step | active/level | frag/level | regrid ms |")
        L.append("|--:|--:|---|---|--:|")
        for r in dyn["timeline"]:
            L.append(f"| {r['regrid']} | {r['step']} | {r['active']} | "
                     f"{r['frag']} | {r['regrid_ms']:.3f} |")

    Path(path).write_text("\n".join(L) + "\n")


# ── Hierarchy builders ─────────────────────────────────────────────────────────

def _gaussian_root_interior(nbx, nby, NF, BS, sigma_frac=0.12, idx=2):
    """A localized Gaussian bump in field `idx` over the (nbx*BS, nby*BS) interior
    — gives a coherent feature whose gradient drives localized refinement."""
    import numpy as np
    nx, ny = nbx * BS, nby * BS
    xs = (np.arange(nx) - nx / 2) / nx
    ys = (np.arange(ny) - ny / 2) / ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    bump = np.exp(-(X**2 + Y**2) / (2 * sigma_frac**2))
    field = np.zeros((NF, nx, ny))
    field[idx] = bump
    return field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbx", type=int, default=2)
    ap.add_argument("--nby", type=int, default=2)
    ap.add_argument("--levels", type=int, default=None,
                    help="override MCS_AMR_LEVELS before import (default: env/4)")
    ap.add_argument("--profile-steps", type=int, default=30)
    ap.add_argument("--micro-iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--trace-iters", type=int, default=50)
    ap.add_argument("--out", type=str, default=None,
                    help="results JSON path (default: traces/amr/profile_results.json)")
    ap.add_argument("--report", type=str, default=None,
                    help="human-readable Markdown report path (default: alongside --out)")
    ap.add_argument("--caps", type=str, default=None,
                    help='per-level slot caps, comma-separated (e.g. "16,6,6,1").')
    ap.add_argument("--autocaps", action="store_true",
                    help="size caps to observed peak occupancy x --cap-margin")
    ap.add_argument("--cap-margin", type=float, default=1.5)
    ap.add_argument("--dynamic", action="store_true",
                    help="also profile the dynamic regime (moving box + regrid churn)")
    ap.add_argument("--dynamic-steps", type=int, default=600)
    ap.add_argument("--regrid-every", type=int, default=50)
    ap.add_argument("--dynamic-max-level", type=int, default=1,
                    help="finest level the moving box refines to (default 1 — bounded)")
    ap.add_argument("--dynamic-radius", type=float, default=1.0,
                    help="moving-box half-width in root-block units (Chebyshev)")
    ap.add_argument("--skip-full-step", action="store_true",
                    help="skip the sub-cycled full-step phase — by far the slowest "
                         "thing to COMPILE (deeply-unrolled recursion). Use when you "
                         "only want the building-block micro-benchmarks and/or --dynamic.")
    ap.add_argument("--unrolled", action="store_true",
                    help="profile full_step with the UNROLLED sub-cycled step (no M1a "
                         "prolongation factoring) instead of the default rolled one. "
                         "Run once with and once without on the SAME GPU to isolate M1a.")
    args = ap.parse_args()

    if args.levels is not None:
        os.environ["MCS_AMR_LEVELS"] = str(args.levels)

    from mcs_common.jax_config import setup as jax_setup
    jax_setup(verbose=True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from mcs2d.amr.state import (
        BS, NG, NF, LEVELS, MAX_BLOCKS, MAX_BLOCKS_PER_LEVEL, AMRState, AMRTopology,
    )
    from mcs2d.amr.evolve import (
        make_subcycled_n_level_step, make_subcycled_n_level_step_unrolled,
        make_n_level_step, amr_state_from_global,
    )
    from mcs2d.amr.kernels import (
        sync_ghosts_within_level_root_periodic, sync_ghosts_within_level,
        sync_ghosts_across_levels, restrict_all_into_parents_highorder,
    )
    from mcs2d.amr.regrid import apply_flags, regrid, REFINE, COARSEN
    from mcs2d.schemes.fused_rhs_fp import _make_kernel_fn

    nbx, nby = args.nbx, args.nby
    LAMBDA, CS, K1, K2, KO = 0.4, 1.0, 1.0, 1.0, 0.05
    dx = 10.0 / (nbx * BS)

    print(f"\n=== AMR profile: backend={jax.default_backend()}  device={jax.devices()[0]} ===")
    print(f"    nbx×nby={nbx}×{nby}  LEVELS={LEVELS}  BS={BS}  MAX_BLOCKS={MAX_BLOCKS}\n")
    _nvidia_smi("start")

    def build_hierarchy(caps, root_field=None, refine=True):
        """Depth-3 test hierarchy with per-level caps `caps`.  Deterministic.
        `root_field` (NF, nbx*BS, nby*BS) seeds the root interior (else random).
        `refine=False` returns a root-only state (for the dynamic driver to grow)."""
        rng = np.random.default_rng(0)
        blocks0 = np.zeros((caps[0], NF, BS + 2*NG, BS + 2*NG))
        if root_field is None:
            interior = rng.standard_normal((nbx*nby, NF, BS, BS))
            blocks0[:nbx*nby, :, NG:NG+BS, NG:NG+BS] = interior
        else:
            tiled = np.asarray(root_field).reshape(NF, nbx, BS, nby, BS)
            tiled = tiled.transpose(1, 3, 0, 2, 4).reshape(nbx*nby, NF, BS, BS)
            blocks0[:nbx*nby, :, NG:NG+BS, NG:NG+BS] = tiled
        blk = tuple(jnp.asarray(blocks0) if L == 0
                    else jnp.zeros((caps[L], NF, BS + 2*NG, BS + 2*NG)) for L in range(LEVELS))
        actv = tuple(jnp.zeros((caps[L],), bool).at[:nbx*nby].set(True) if L == 0
                     else jnp.zeros((caps[L],), bool) for L in range(LEVELS))
        st = AMRState(blocks=blk, active=actv)
        tp = AMRTopology(caps=list(caps))
        for s in range(nbx*nby):
            tp.add_block(0, s, ((s // nby) * BS, (s % nby) * BS))
        if refine:
            for L in range(min(LEVELS - 1, 2)):
                finder = 0 if L == 0 else int(np.flatnonzero(np.asarray(st.active[L]))[0])
                flags = np.zeros((LEVELS, max(caps)), dtype=np.int32)
                flags[L, finder] = REFINE
                st, tp = apply_flags(st, tp, flags, auto_grow=False)
        return st, tp

    default_caps = list(MAX_BLOCKS_PER_LEVEL)
    if args.caps:
        caps = [int(c) for c in args.caps.split(",")]
        assert len(caps) == LEVELS, f"--caps needs {LEVELS} entries, got {len(caps)}"
    elif args.autocaps:
        _, _tp = build_hierarchy(default_caps)
        _tp.record_occupancy()
        caps = [max(1, c) for c in _tp.recommended_caps(margin=args.cap_margin)]
    else:
        caps = default_caps
    state, topo = build_hierarchy(caps)
    ta = topo.to_jax_arrays()
    print(f"    caps per level   : {caps}")
    print("    active per level :",
          [int(np.asarray(state.active[L]).sum()) for L in range(LEVELS)])

    results = {
        "device": str(jax.devices()[0]),
        "backend": jax.default_backend(),
        "config": {
            "nbx": nbx, "nby": nby, "levels": int(LEVELS), "BS": int(BS),
            "NG": int(NG), "NF": int(NF), "max_blocks": int(MAX_BLOCKS),
            "caps_per_level": [int(c) for c in caps],
            "active_per_level": [int(np.asarray(state.active[L]).sum())
                                 for L in range(LEVELS)],
        },
    }

    trace_dir = _REPO_ROOT / "traces" / "amr"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # ── STATIC: per-phase timing + trace + cost model + memory ──────────────────
    NP = args.profile_steps
    fullstep_defs = []
    if not args.skip_full_step:
        # The sub-cycled N-level step is a deeply-unrolled recursion → by far the
        # slowest thing to COMPILE.  --skip-full-step omits it (and its `run` scan).
        _mk = (make_subcycled_n_level_step_unrolled if args.unrolled
               else make_subcycled_n_level_step)
        if args.unrolled:
            print(">> --unrolled: profiling full_step with the UNROLLED step (no M1a).")
        step_sub = _mk(dx, dx, 0.05 * dx, CS, LAMBDA, K1, K2, KO, nbx, nby)

        @jax.jit
        def run(s):
            return jax.lax.scan(lambda c, _: (step_sub(c, ta), None), s, None, length=NP)[0]

        print("\n>> Warmup (compile sub-cycled step — the slow one)…")
        t0 = time.perf_counter()
        _s = step_sub(state, ta); _s.blocks[0].block_until_ready()
        print(f"   compile wall: {time.perf_counter()-t0:.1f}s")
        fullstep_defs = [("full_step", run, (state,), 1, 1, step_sub, (state, ta))]
    else:
        print("\n>> --skip-full-step: not compiling the sub-cycled step.")

    kern    = _make_kernel_fn(dx, dx, CS, LAMBDA, K1, K2, KO)
    kern_nk = _make_kernel_fn(dx, dx, CS, LAMBDA, K1, K2, 0.0)   # KO off
    rhs_all    = jax.jit(jax.vmap(kern))
    rhs_no_ko  = jax.jit(jax.vmap(kern_nk))
    sync_root  = jax.jit(lambda b: sync_ghosts_within_level_root_periodic(b, nbx, nby))
    sync_wl    = jax.jit(lambda b: sync_ghosts_within_level(
        b, ta.neighbor_slot[1], ta.neighbor_valid[1]))
    sync_x     = jax.jit(lambda f, c: sync_ghosts_across_levels(
        f, c, ta.parent_slot[1], ta.child_cx[1], ta.child_cy[1], state.active[1]))
    restrict   = jax.jit(lambda c, f: restrict_all_into_parents_highorder(
        c, f, ta.parent_slot[1], ta.child_cx[1], ta.child_cy[1], state.active[1]))
    lvl0, lvl1 = state.blocks[0], state.blocks[1]

    # name → (fn, args, bench_iters, trace_iters, cost_fn, cost_args)
    NI, NTRC = args.micro_iters, args.trace_iters
    phase_defs = fullstep_defs + [
        ("rhs",        rhs_all,   (lvl0,),      NI, NTRC, rhs_all,  (lvl0,)),
        ("rhs_no_ko",  rhs_no_ko, (lvl0,),      NI, NTRC, rhs_no_ko,(lvl0,)),
        ("sync_root",  sync_root, (lvl0,),      NI, NTRC, sync_root,(lvl0,)),
        ("sync_within", sync_wl,  (lvl1,),      NI, NTRC, sync_wl,  (lvl1,)),
        ("sync_cross", sync_x,    (lvl1, lvl0), NI, NTRC, sync_x,   (lvl1, lvl0)),
        ("restrict",   restrict,  (lvl0, lvl1), NI, NTRC, restrict, (lvl0, lvl1)),
    ]

    print(f"\n>> Static per-phase profile (repeats={args.repeats}, micro_iters={NI}):")
    phases = {}
    for name, fn, a, bi, ti, cfn, cargs in phase_defs:
        _trace_phase(trace_dir, name, fn, a, ti)
        rec = {"us": _bench_us(fn, a, bi, args.repeats),
               "trace": _parse_phase_trace(trace_dir / name),
               "cost": _cost_analysis(cfn, cargs),
               "mem": _mem_stats()}
        phases[name] = rec
        c = rec["cost"]
        print(f"   {name:11s}: {rec['us']:9.1f} µs/call  "
              f"GFLOP={c.get('gflops','?')} FLOP/byte={c.get('arith_intensity','?')} "
              f"({c.get('bound','?')})  → traces/amr/{name}/")

    results["phases"] = phases
    if "full_step" in phases:
        results["per_step_ms"] = phases["full_step"]["us"] / 1e3 / NP

    # ── DYNAMIC: synthetic MOVING BOX + regrid churn ────────────────────────────
    # A fixed-size refined region TRANSLATES across the domain (representative of a
    # moving puncture — it moves rather than spreads).  Refinement is driven
    # synthetically (not by the physics indicator) so the churn is bounded and
    # reproducible: each regrid we REFINE root blocks within `--dynamic-radius` of a
    # moving centre and COARSEN fine blocks left behind.  This isolates the regrid
    # MECHANICS (per-regrid host cost, occupancy variance, slot fragmentation,
    # recompiles) — the evidence base for the compaction (4.6) decision.
    if args.dynamic:
        maxL = max(1, min(args.dynamic_max_level, LEVELS - 1))
        radius = args.dynamic_radius
        print(f"\n>> Dynamic regime: {args.dynamic_steps} steps, regrid every "
              f"{args.regrid_every}; moving box (max_level={maxL}, radius={radius})…")
        gfield = _gaussian_root_interior(nbx, nby, NF, BS)
        dstate, dtopo = build_hierarchy(caps, root_field=gfield, refine=False)
        dt = 0.05 * dx / (2 ** maxL)
        nstep = make_n_level_step(dx, dx, dt, CS, LAMBDA, K1, K2, KO, nbx, nby)
        rc = [0]
        inner = nstep.__wrapped__
        def counted(*a, **k):
            rc[0] += 1
            return inner(*a, **k)
        nstep_c = jax.jit(counted)

        def _near(bbox_cells, cbx, cby):
            return (abs(bbox_cells[0] / BS - cbx) <= radius and
                    abs(bbox_cells[1] / BS - cby) <= radius)

        def _root_ancestor_bbox(L, s):
            while L > 0:
                L, s = dtopo.parent[(L, s)]
            return dtopo.bbox_ijk[(0, s)]

        def moving_box_flags(cbx, cby):
            """REFINE root blocks near (cbx,cby); COARSEN fine blocks left behind.
            Refine bottom-up (root→…) so a new level can be added each regrid."""
            flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
            for L in range(maxL):                     # parents that should have kids
                for s in range(dtopo.caps[L]):
                    if dtopo.active[L, s] and _near(_root_ancestor_bbox(L, s), cbx, cby):
                        flags[L, s] = REFINE
            for L in range(1, maxL + 1):              # fine blocks now outside the box
                for s in range(dtopo.caps[L]):
                    if dtopo.active[L, s] and not _near(_root_ancestor_bbox(L, s), cbx, cby):
                        flags[L, s] = COARSEN
            return flags

        chunk_ms, regrid_ms, timeline = [], [], []
        occ_peak = [0] * LEVELS
        occ_min = [10**9] * LEVELS
        frag_max = [1.0] * LEVELS
        n_growths0 = len(dtopo.grow_history)
        lo, hi = radius, max(radius, (nbx - 1) - radius)   # centre travel range

        steps_done, n_regrid = 0, 0
        while steps_done < args.dynamic_steps:
            tarrays = dtopo.to_jax_arrays()
            t0 = time.perf_counter()                  # timed GPU chunk
            for _ in range(args.regrid_every):
                dstate = nstep_c(dstate, tarrays)
            dstate.blocks[0].block_until_ready()
            chunk_ms.append((time.perf_counter() - t0) / args.regrid_every * 1e3)
            steps_done += args.regrid_every

            frac = steps_done / args.dynamic_steps     # centre translates lo→hi→lo
            tri = 1.0 - abs(2.0 * frac - 1.0)          # 0→1→0 (there and back)
            cbx = lo + tri * (hi - lo)
            cby = lo + tri * (hi - lo)
            flags = moving_box_flags(cbx, cby)
            t0 = time.perf_counter()                   # timed HOST regrid
            try:
                dstate, dtopo = apply_flags(dstate, dtopo, flags, auto_grow=False)
            except RuntimeError as e:
                print(f"   [dynamic] regrid hit budget at step {steps_done}: {e}")
                break
            rms = (time.perf_counter() - t0) * 1e3
            regrid_ms.append(rms)
            n_regrid += 1
            active = [int(np.asarray(dstate.active[L]).sum()) for L in range(LEVELS)]
            frag = [_frag(dstate.active[L]) for L in range(LEVELS)]
            for L in range(LEVELS):
                occ_peak[L] = max(occ_peak[L], active[L])
                occ_min[L] = min(occ_min[L], active[L])
                frag_max[L] = max(frag_max[L], frag[L])
            timeline.append({"regrid": n_regrid, "step": steps_done,
                             "center": [round(cbx, 2), round(cby, 2)],
                             "active": active, "frag": frag,
                             "regrid_ms": round(rms, 3)})

        results["dynamic"] = {
            "steps": steps_done, "regrid_every": args.regrid_every,
            "max_level": maxL, "radius": radius,
            "recompiles": rc[0],
            "cap_growths": len(dtopo.grow_history) - n_growths0,
            "chunk_ms_median": _median(chunk_ms) if chunk_ms else 0.0,
            "chunk_ms_max": max(chunk_ms) if chunk_ms else 0.0,
            "regrid_ms_median": _median(regrid_ms) if regrid_ms else 0.0,
            "regrid_ms_max": max(regrid_ms) if regrid_ms else 0.0,
            "occupancy_peak": occ_peak,
            "occupancy_min": [0 if m == 10**9 else m for m in occ_min],
            "frag_max": frag_max,
            "timeline": timeline,
        }
        d = results["dynamic"]
        print(f"   recompiles={d['recompiles']}  cap_growths={d['cap_growths']}  "
              f"chunk_ms(med)={d['chunk_ms_median']:.3f}  regrid_ms(med)={d['regrid_ms_median']:.3f}")
        print(f"   occupancy peak={d['occupancy_peak']} min={d['occupancy_min']}  "
              f"frag_max={d['frag_max']}")

    # ── Emit JSON + Markdown report ─────────────────────────────────────────────
    out_path = Path(args.out) if args.out else (trace_dir / "profile_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    report_path = Path(args.report) if args.report else out_path.with_suffix(".md")
    _write_report(report_path, results)

    print(f"\n>> Results JSON   : {out_path}")
    print(f">> Markdown report: {report_path}")
    _nvidia_smi("end")
    n_traces = len(sorted(glob.glob(str(trace_dir / "**" / "*.trace.json.gz"), recursive=True)))
    print(f">> Trace files: {n_traces} (per-phase dirs in {trace_dir}); "
          f"open *.trace.json.gz at https://ui.perfetto.dev\n")


if __name__ == "__main__":
    main()
