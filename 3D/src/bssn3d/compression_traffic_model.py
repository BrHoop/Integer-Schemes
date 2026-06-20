"""Phase-4 Step 4.1 — spill-TRAFFIC-aware feasibility refinement (the decisive CPU model).

`compression_model.py` scored OCCUPANCY and said BFP compression is ~break-even. But M4 is
spill-*traffic*-bound, not occupancy-bound (--smi MEM 76%, "the memory activity IS spill"), and
the occupancy model admits it under-weights spill. This module models the regime that actually
governs the kernel, calibrated to the REAL v7 A/B anchors (step_3.5), and adds the one cost the
occupancy model omitted entirely: the **pack/unpack instruction overhead**, which is large
because M4 spills ~3600x per thread.

Calibration (two measured anchors, same box / N=128^3, only the v7 toggle differs):
  legacy M4 : 20.47 ms, spill stores 12556 B/thr + loads 16276 B/thr  (28832 B/thr)
  v7        : 21.86 ms, spill stores 15412 B/thr + loads 18408 B/thr  (33820 B/thr)
  => d(time)/d(spill bytes) gives the EFFECTIVE spill cost (an on-chip/L2 bandwidth, ~2x HBM),
     and decomposes M4 time into a spill-free `base` + a `spill` term.

The decisive uncertainty a CPU model cannot resolve: is the spill LATENCY-bound (pack/unpack
overlaps stall cycles -> "free", the user's premise) or THROUGHPUT/issue-bound (pack/unpack adds
to the instruction stream -> the v7 scar)? We report BOTH bounds; the truth is between, and which
end requires the Marylou A/B.

Run:  python -m bssn3d.compression_traffic_model
"""
from __future__ import annotations

from dataclasses import dataclass

from ._codegen import parse, DENDRO_CSE
from .staging import build_dag, min_liveness_order
from .fusion_level_model import liveness_curves

# --- measured anchors (step_3.5 v7 A/B, H100, N=128^3) ---------------------------------------
N_PTS = 128 ** 3
T_LEGACY = 20.47e-3        # s
T_V7 = 21.86e-3
SPILL_LEGACY = 12556 + 16276    # B/thread (stores + loads)
SPILL_V7 = 15412 + 18408
STORE_EVENTS = 12556 / 8        # fp64 spill stores per thread (~1569)
LOAD_EVENTS = 16276 / 8         # fp64 spill loads per thread (~2035)

# pack/unpack op counts (storage-only BFP; compute stays fp64)
PACK_OPS = 3.0     # scale (*2^e) + round + int-convert/store-prep
UNPACK_OPS = 2.0   # int->float convert + multiply (by precomputed 2^-e)


@dataclass
class TrafficCalib:
    cost_per_byte: float   # s per spill byte (total, across all threads)
    base_s: float          # spill-free compute time
    spill_s: float         # spill-traffic time in M4
    bw_eff: float          # effective spill bandwidth (B/s)


def calibrate() -> TrafficCalib:
    dspill = (SPILL_V7 - SPILL_LEGACY) * N_PTS
    dtime = T_V7 - T_LEGACY
    c = dtime / dspill
    spill_s = c * SPILL_LEGACY * N_PTS
    return TrafficCalib(cost_per_byte=c, base_s=T_LEGACY - spill_s,
                        spill_s=spill_s, bw_eff=1.0 / c)


def algebra_ops() -> float:
    stmts, _, _ = parse(DENDRO_CSE)
    dag = build_dag(stmts)
    order = min_liveness_order(dag)
    _, _, _, op_at, _, _ = liveness_curves(stmts, dag, order)
    return float(op_at.sum())


@dataclass
class Bracket:
    kbytes: int
    spill_new_s: float
    pu_s: float
    t_throughput: float    # pack/unpack ADDS to the stream (issue-bound)
    t_latency: float       # pack/unpack OVERLAPS spill stalls (latency-bound)


def bracket(calib: TrafficCalib, alg_ops: float, kbytes_opts=(6, 5, 4, 3, 2)):
    # pack/unpack time: scaled off the base compute rate (alg_ops in base_s).
    pu_ops = STORE_EVENTS * PACK_OPS + LOAD_EVENTS * UNPACK_OPS   # per thread
    pu_s = calib.base_s * (pu_ops / alg_ops)
    rows = []
    for kb in kbytes_opts:
        spill_new = calib.spill_s * (kb / 8.0)   # fewer bytes per spilled value
        t_through = calib.base_s + pu_s + spill_new           # issue-bound: pack/unpack serialize
        t_latency = calib.base_s + spill_new                  # latency-bound: pack/unpack hides in stalls
        rows.append(Bracket(kbytes=kb, spill_new_s=spill_new, pu_s=pu_s,
                            t_throughput=t_through, t_latency=t_latency))
    return rows, pu_ops


def main():
    calib = calibrate()
    alg_ops = algebra_ops()
    rows, pu_ops = bracket(calib, alg_ops)

    print(">> BSSN BFP compression — spill-TRAFFIC feasibility (CPU, calibrated to v7 A/B)")
    print(f"   effective spill bandwidth = {calib.bw_eff/1e12:.2f} TB/s "
          f"(~{calib.bw_eff/3.35e12:.1f}x HBM -> L2-served, NOT bulk-HBM)")
    print(f"   M4 @128^3 decomposed: {T_LEGACY*1e3:.2f} ms = {calib.base_s*1e3:.2f} ms base "
          f"+ {calib.spill_s*1e3:.2f} ms spill  ({calib.spill_s/T_LEGACY*100:.0f}% spill)\n")
    print(f"   algebra = {alg_ops:.0f} ops/pt ; spill events = {STORE_EVENTS:.0f} stores + "
          f"{LOAD_EVENTS:.0f} loads /thread")
    print(f"   pack/unpack = {pu_ops:.0f} ops/pt  =  {pu_ops/alg_ops:.1f}x the entire algebra "
          f"FLOP count  <-- the cost the occupancy model ignored\n")

    print(f"   {'BFP':>6} | {'spill ms':>8} {'pack/unpk ms':>12} | "
          f"{'THROUGHPUT-bound':>16} | {'LATENCY-bound':>14}")
    print(f"   {'':>6} | {'':>8} {'':>12} | {'ms':>6} {'vs M4':>8} | {'ms':>6} {'vs M4':>6}")
    for r in rows:
        sp_through = T_LEGACY / r.t_throughput
        sp_lat = T_LEGACY / r.t_latency
        tag = f"{r.kbytes*8}-bit"
        print(f"   {tag:>6} | {r.spill_new_s*1e3:>8.2f} {r.pu_s*1e3:>12.2f} | "
              f"{r.t_throughput*1e3:>6.2f} {sp_through:>7.2f}x | "
              f"{r.t_latency*1e3:>6.2f} {sp_lat:>5.2f}x")

    print("\n   VERDICT (honest bracket):")
    print("   * THROUGHPUT/issue-bound (pack/unpack serialize): net NEGATIVE — ~0.5-0.6x (SLOWER),")
    print("     because M4 spills ~3600x/thread and even ~2-3 ops/event > the byte savings. This is")
    print("     the v7 scar mechanism (more instructions on a spill-bound kernel = slower).")
    print("   * LATENCY-bound (pack/unpack overlap spill stalls — the 'ALU is free' premise): a")
    print("     modest WIN ~1.1-1.4x, riding the spill traffic removed (compression ratio).")
    print("   * The sign DEPENDS on latency- vs throughput-bound spill — a CPU model cannot tell.")
    print("     Decisive GPU probe: is M4 spill latency- or issue-bound, and does adding pack/unpack")
    print("     ops overlap or serialize? That single Marylou A/B settles Phase 4 feasibility.")


if __name__ == "__main__":
    main()
