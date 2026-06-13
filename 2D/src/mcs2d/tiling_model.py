"""
Tiling / temporal-fusion feasibility model — Step 1.3 T0 (analysis, no GPU).

Answers, with arithmetic only, the question that decides the whole tensor-core
plan: **is there an operating point that becomes compute-bound (clears the ridge)
within the H200's shared-memory budget, with acceptable redundant compute** — for
the MCS toy model AND for the real target, BSSN (which is far more compute-heavy,
and 3D, so the regime and the tiling tradeoffs both change).

Roofline model (max(DRAM_time, compute_time)); absolute throughput is order-of-
magnitude (efficiency is swept), but the *regime* (DRAM vs compute bound), the
shared-memory feasibility, and the redundant-halo cost are robust.

Key levers it exposes:
  * dtype bytes/field   — fp64 8 B, BFP48 6 B, Ozaki k-limb k B  (DRAM-bound win)
  * fusion depth L      — sequential stencil-apps/kernel; halo grows as L·NG
  * dimensionality      — 3D halos blow up as (·)³, and 24 fields strain smem
  * compute target      — FP64 vector / FP16-TC / INT8-TC (with Ozaki overhead)
"""

from __future__ import annotations

import argparse
import math

# ── Hardware (H200) ───────────────────────────────────────────────────────────
HBM_GBs        = 4800.0          # HBM3e peak bandwidth, GB/s
SMEM_KB        = 228.0           # shared memory per SM (Hopper, max config)
FP64_GFLOPs    = 34_000.0        # FP64 vector peak
FP16_TC_GFLOPs = 989_000.0       # FP16 tensor-core peak (dense)
INT8_TC_GOPs   = 1_979_000.0     # INT8 tensor-core peak (dense; 2:4 sparse ~2×)

GEMM_WASTE_DENSE  = 4.6          # banded stencil as dense GEMM wastes ~4.6× on zeros
GEMM_WASTE_SPARSE = 2.3          # 2:4 sparse-TC packing (Phase 1.5) recovers ~half

# ── Problems ──────────────────────────────────────────────────────────────────
#  flop = useful physics FLOP / point / step (RK4).  BSSN is a ROUGH estimate
#  (~6k/RHS × 4: ~2k derivatives in 3D + ~4k Christoffel/Ricci/gauge algebra);
#  the group's System-II code has the real count — override with --flop.
PROBLEMS = {
    #  name     NF  NG  dim   flop    G    nstages  xla_mpts  xla_xstate
    "mcs":  dict(NF=10, NG=3, dim=2, flop=2400.0,  G=512, nstages=4, xla_mpts=225.0, xla_x=15.0),
    "bssn": dict(NF=24, NG=3, dim=3, flop=24000.0, G=256, nstages=4, xla_mpts=None,  xla_x=15.0),
}

# ── Scenarios: (label, bytes/field, peak, comp_mult, uses-GEMM, fp64-accurate) ─
#  comp_mult = low-precision ops per useful fp64 FLOP (before GEMM waste).
#  Ozaki stores k INT8 limbs/field (db=k) and does ~comp_mult INT8 MACs/fp64 MAC.
SCENARIOS = {
    "fp64":      (8,  FP64_GFLOPs,     1.0,  False, True),
    "bfp48":     (6,  FP64_GFLOPs,     1.0,  False, True),   # 48-bit block FP, ~fp64-accurate
    "fp16_tc":   (2,  FP16_TC_GFLOPs,  1.0,  True,  False),  # speed probe; NOT fp64-accurate
    "int8_raw":  (1,  INT8_TC_GOPs,    1.0,  True,  False),  # best-case INT8; NOT accurate
    "ozaki_k8":  (8,  INT8_TC_GOPs,   16.0,  True,  True),   # fp64-accurate; 8 limbs
    "ozaki_k4":  (4,  INT8_TC_GOPs,    8.0,  True,  True),   # compact Ozaki; 4 limbs
}


def model(prob, lbl, BS, L, eff=0.4, scratch=2.0, gemm_waste=GEMM_WASTE_SPARSE):
    NF, NG, dim, flop = prob["NF"], prob["NG"], prob["dim"], prob["flop"]
    nstages = prob["nstages"]
    db, peak, cm, gemm, _p = SCENARIOS[lbl]

    side = BS + 2 * L * NG
    redundant = (side / BS) ** dim                 # halo blowup (cubic in 3D!)
    state_pts = prob["G"] ** dim
    state_bytes = state_pts * NF * db
    ops_per_flop = cm * (gemm_waste if gemm else 1.0)

    smem_KB = (side ** dim) * NF * db * scratch / 1024.0
    fits = smem_KB <= SMEM_KB

    # DRAM/step: each kernel covers L/nstages steps; load redundant×tile, write once.
    dram_xstate = nstages * (redundant + 1.0) / L
    dram_bytes_step = dram_xstate * state_bytes
    dram_time = dram_bytes_step / (HBM_GBs * 1e9)

    flops_step = flop * state_pts
    comp_time = (flops_step * redundant * ops_per_flop) / (peak * 1e9 * eff)

    step_time = max(dram_time, comp_time)
    ia = flops_step / dram_bytes_step              # useful FLOP / DRAM byte
    return dict(side=side, redundant=redundant, smem_KB=smem_KB, fits=fits,
                dram_xstate=dram_xstate, ia=ia, mpts=state_pts / step_time / 1e6,
                bound="DRAM" if dram_time >= comp_time else "compute",
                dram_time=dram_time, comp_time=comp_time)


def max_tile_side(prob, db, scratch):
    return int((SMEM_KB * 1024.0 / (prob["NF"] * db * scratch)) ** (1.0 / prob["dim"]))


def best_feasible(prob, lbl, eff, scratch, gemm_waste):
    db = SCENARIOS[lbl][0]
    best = None
    for L in range(1, 13):
        for BS in range(2, 257, 2):
            m = model(prob, lbl, BS, L, eff, scratch, gemm_waste)
            if m["fits"] and (best is None or m["mpts"] > best["mpts"]):
                best = {**m, "BS": BS, "L": L}
    return best


def ridge(prob, lbl, gemm_waste):
    """Roofline ridge (FLOP/byte) for this scenario's compute path on this dtype."""
    db, peak, cm, gemm, _p = SCENARIOS[lbl]
    ops_per_flop = cm * (gemm_waste if gemm else 1.0)
    # compute_bound when useful-FLOP/byte > (peak/ops_per_flop)/HBM, in fp64-FLOP/byte
    return (peak / ops_per_flop) / HBM_GBs


def report(prob_name, eff, scratch, gemm_waste, fp64_ref):
    prob = PROBLEMS[prob_name]
    NF, NG, dim, flop, G = prob["NF"], prob["NG"], prob["dim"], prob["flop"], prob["G"]
    print(f"\n{'='*78}\nPROBLEM = {prob_name.upper()}  "
          f"(NF={NF}, NG={NG}, {dim}D, ~{flop:.0f} FLOP/pt/step, G={G})")
    # naive (untiled, ~2×state ideal) intensity, dtype-independent in fp64
    naive_ia = flop / (2.0 * NF * 8)
    print(f"  ideal untiled intensity (2×state, fp64): {naive_ia:.1f} FLOP/byte   "
          f"[FP64 ridge 7.1, INT8-TC ridge ~410]")

    print(f"\n  Shared-memory feasibility (smem {SMEM_KB:.0f} KB, scratch×{scratch}):")
    for lbl, (db, *_r) in SCENARIOS.items():
        s = max_tile_side(prob, db, scratch)
        bs = s - 2 * NG                              # interior at L=1
        print(f"    {lbl:9s} {db} B/field  max tile side {s:>3}  -> max interior BS "
              f"(L=1) = {bs if bs > 0 else 'NONE — tile too big'}")

    print(f"\n  Best feasible config (eff={eff:.0%}, {('2:4-sparse' if gemm_waste==GEMM_WASTE_SPARSE else f'GEMM×{gemm_waste}')}):")
    print(f"    {'scenario':9s} {'BS':>4} {'L':>3} {'redund':>7} {'smem':>6} "
          f"{'DRAM×st':>8} {'bound':>8} {'Mpts/s':>10} {'vs fp64':>8} {'accurate':>9}")
    fp64_mpts = None
    rows = {}
    for lbl in SCENARIOS:
        b = best_feasible(prob, lbl, eff, scratch, gemm_waste)
        rows[lbl] = b
        if lbl == "fp64" and b:
            fp64_mpts = b["mpts"]
    ref = fp64_ref if fp64_ref else fp64_mpts
    for lbl, b in rows.items():
        acc = "yes" if SCENARIOS[lbl][4] else "NO"
        if not b:
            print(f"    {lbl:9s}  — no shared-memory-feasible tiling")
            continue
        vs = f"{b['mpts']/ref:.1f}×" if ref else "—"
        print(f"    {lbl:9s} {b['BS']:>4} {b['L']:>3} {b['redundant']:>6.1f}× "
              f"{b['smem_KB']:>5.0f}K {b['dram_xstate']:>7.1f} {b['bound']:>8} "
              f"{b['mpts']:>10.0f} {vs:>8} {acc:>9}")

    # Untiled COMPUTE ceiling (no shared-mem tiling, redundant≈1).  When the
    # problem is naturally compute-bound (intensity > ridge), tiling is neither
    # feasible (3D, 24 fields) nor needed — the throughput is set by compute, and
    # tensor cores act on it directly.  This is the realistic 3D-BSSN number.
    state_pts = G ** dim
    print(f"\n  Untiled compute ceiling (no smem tiling — the realistic path if "
          f"compute-bound):")
    fp64_ceiling = FP64_GFLOPs * 1e9 * eff / flop
    for lbl in SCENARIOS:
        db, peak, cm, gemm, acc = SCENARIOS[lbl]
        opf = cm * (gemm_waste if gemm else 1.0)
        ceil_pts = peak * 1e9 * eff / (flop * opf)
        # DRAM floor (ideal 2×state) to flag if memory would actually bind it
        dram_floor_pts = HBM_GBs * 1e9 / (2.0 * NF * db) * 1.0  # pts/s at 2×state
        memcap = dram_floor_pts < ceil_pts
        tag = "  (DRAM-capped!)" if memcap else ""
        print(f"    {lbl:9s} {ceil_pts/1e6:>10.0f} Mpts/s   "
              f"{ceil_pts/fp64_ceiling:>5.1f}× fp64   accurate={'yes' if acc else 'NO ':>3}{tag}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Tiling / fusion feasibility model")
    ap.add_argument("--problem", choices=list(PROBLEMS) + ["both"], default="both")
    ap.add_argument("--eff", type=float, default=0.4)
    ap.add_argument("--scratch", type=float, default=2.0)
    ap.add_argument("--flop", type=float, default=None, help="override BSSN FLOP/pt/step")
    ap.add_argument("--dense", action="store_true", help="dense GEMM waste (4.6×) not 2:4 sparse")
    a = ap.parse_args()
    gw = GEMM_WASTE_DENSE if a.dense else GEMM_WASTE_SPARSE
    if a.flop:
        PROBLEMS["bssn"]["flop"] = a.flop

    print("Tiling / temporal-fusion feasibility model (Step 1.3 T0)")
    print(f"H200: HBM {HBM_GBs/1000:.1f} TB/s, smem {SMEM_KB:.0f} KB/SM; "
          f"FP64 {FP64_GFLOPs/1000:.0f} / FP16-TC {FP16_TC_GFLOPs/1000:.0f} TFLOP/s, "
          f"INT8-TC {INT8_TC_GOPs/1000:.0f} TOP/s")

    names = list(PROBLEMS) if a.problem == "both" else [a.problem]
    for nm in names:
        # use MCS's measured XLA baseline as the fp64 ref where we have it
        report(nm, a.eff, a.scratch, gw, fp64_ref=None)


if __name__ == "__main__":
    main()
