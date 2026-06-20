"""Phase-4 Step 4.1 (footprint half) — does BFP compression actually relieve the M4 spill?

The numerics emulator (3D MCS) showed BFP storage is accuracy-cheap. This module answers the
ORTHOGONAL question on CPU, before any Marylou push: if idle live values are parked in BFP-k
while only the *active* frontier stays fp64, does the peak simultaneously-live working set drop
under the register file (1020 B) and/or fit on-chip SMEM?

Method (reuses the Step-3.4 liveness machinery):
  * min-liveness schedule of the real fused DAG (ptxas's own reorder proxy) -> the peak-live
    boundary (~440 values, the seam: ~422 algebra temps + derivs).
  * at the peak, reconstruct the live SET and split it into ACTIVE (read/written by any statement
    within +/-W positions -> must be fp64, 8 B) vs IDLE (live but untouched -> BFP-k, k/8 B).
  * peak working-set bytes = |active|*8 + |idle|*kbytes, compared to:
      - REGISTER target (R): must fit REG_CAP*4 = 1020 B/thread.
      - SMEM target (S): active fits registers; idle (compressed) lives in SMEM -> max
        threads/block = SMEM_BUDGET / (|idle|*kbytes).
  * the v7 scar is parameterized by W: a larger unpacked window = bigger active set = less benefit.

This is a SCREENING model (like fusion_level_model): it bounds the win and says whether the GPU
trip is worth it. It does not predict wall-clock.

Run:  python -m bssn3d.compression_model
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._codegen import parse, DENDRO_CSE
from .staging import build_dag, min_liveness_order
from .fusion_level_model import (
    liveness_curves, _GRAD_RE, REG_CAP, REGS_PER_FP64, REGS_PER_SM, WARP, MAX_WARPS_SM,
)

REG_BYTES = REG_CAP * 4              # 255 regs x 4 B = 1020 B per thread
SMEM_BUDGET = 228 * 1024            # H100/H200 max dynamic SMEM per block (optin), bytes
FP64_B = 8

# --- occupancy / eff (same calibration as fusion_level_model) --------------------------------
W0 = 16.0                # warps to saturate latency hiding (EffModel default)
SPILL_ALPHA = 0.05       # throughput penalty per spilled fp64 value
M4_REGS = 276            # ptxas -v on the fused kernel
M4_WARPS = REGS_PER_SM // (M4_REGS * WARP)            # reg-limited warps/SM for M4 (~7)
M4_SPILL_VALS = max(0, M4_REGS - REG_CAP) / REGS_PER_FP64


def _occ(w):
    return w / (w + W0)


# eff = occupancy * spill-relief; M4 spills, the SMEM-staged variant does not.
M4_EFF = _occ(M4_WARPS) / (1.0 + SPILL_ALPHA * M4_SPILL_VALS)


def _spans(statements, dag, order):
    """(pos, last, deriv_first, deriv_last) — def/last-use boundaries for alg temps and derivs,
    using the SAME conventions as fusion_level_model.liveness_curves."""
    pos = {n: i for i, n in enumerate(order)}
    stmt = dict(statements)
    last = {nm: pos[nm] for nm in order}
    deriv_first, deriv_last = {}, {}
    for nm in order:
        for d in dag.deps[nm]:
            if d in pos:
                last[d] = max(last[d], pos[nm])
        for g in _GRAD_RE.findall(stmt[nm]):
            deriv_first.setdefault(g, pos[nm])
            deriv_last[g] = pos[nm]
    return pos, last, deriv_first, deriv_last


def _live_set(pos, last, deriv_first, deriv_last, b):
    """Names live across boundary b (alg: pos<b<=last ; deriv: first<=b<=last)."""
    alg = {nm for nm, p in pos.items() if p < b <= last[nm]}
    drv = {g for g in deriv_first if deriv_first[g] <= b <= deriv_last[g]}
    return alg, drv


def _touched_in_window(statements, dag, order, center, W):
    """All value names read or written by statements within +/-W of position `center`."""
    stmt = dict(statements)
    lo, hi = max(0, center - W), min(len(order), center + W + 1)
    touched = set()
    for q in range(lo, hi):
        nm = order[q]
        touched.add(nm)                       # output
        touched.update(dag.deps[nm])          # algebra operands
        touched.update(_GRAD_RE.findall(stmt[nm]))  # derivative operands
    return touched


@dataclass
class FeasRow:
    kbytes: int
    window: int
    active: int
    idle: int
    peak_bytes: int
    fits_registers: bool
    smem_max_threads: int   # threads/block that keep the idle compressed set in SMEM


def feasibility(windows=(0, 2, 4, 8), kbytes_opts=(8, 6, 5, 4, 3, 2)):
    statements, grad1, grad2 = parse(DENDRO_CSE)
    dag = build_dag(statements)
    order = min_liveness_order(dag)
    pos, alg_live, deriv_live, *_ = liveness_curves(statements, dag, order)
    spans = _spans(statements, dag, order)

    total = alg_live + deriv_live
    b_star = int(total.argmax())
    peak_vals = int(total[b_star])
    alg_set, drv_set = _live_set(*spans, b_star)
    live_names = alg_set | drv_set
    baseline_bytes = peak_vals * FP64_B   # all-fp64 working set (today)

    rows = []
    for W in windows:
        touched = _touched_in_window(statements, dag, order, b_star - 1, W)
        active = len(live_names & touched)
        idle = peak_vals - active
        for kb in kbytes_opts:
            peak_bytes = active * FP64_B + idle * kb
            smem_threads = SMEM_BUDGET // max(1, idle * kb)
            rows.append(FeasRow(
                kbytes=kb, window=W, active=active, idle=idle,
                peak_bytes=peak_bytes, fits_registers=peak_bytes <= REG_BYTES,
                smem_max_threads=int(smem_threads),
            ))
    return dict(peak_vals=peak_vals, baseline_bytes=baseline_bytes,
                alg=int(alg_live.max()), drv=int(deriv_live[b_star]), b_star=b_star,
                live_names=live_names, rows=rows)


def main():
    r = feasibility()
    print(">> BSSN fused-RHS BFP compression feasibility (CPU screening, H100/H200)")
    print(f"   peak-live boundary @ pos {r['b_star']}: {r['peak_vals']} values "
          f"(alg peak {r['alg']} + derivs live {r['drv']})")
    print(f"   today (all fp64): peak working set = {r['baseline_bytes']} B = "
          f"{r['baseline_bytes']/REG_BYTES:.1f}x the {REG_BYTES} B register file -> spills.\n")
    print(f"   register file = {REG_BYTES} B/thread ; SMEM budget = {SMEM_BUDGET//1024} KB/block\n")

    print(f"   {'BFP':>4} {'W':>2} | {'active':>6} {'idle':>5} | {'peak B':>7} {'vs regfile':>10} "
          f"| {'fits R?':>7} | {'SMEM max thr/block':>18}")
    last_w = None
    for row in r["rows"]:
        if last_w is not None and row.window != last_w:
            print("   " + "-" * 70)
        last_w = row.window
        tag = "8B=fp64" if row.kbytes == 8 else f"{row.kbytes*8}-bit"
        fitsR = "YES" if row.fits_registers else "no"
        smem = f"{row.smem_max_threads} (={row.smem_max_threads//32}w)" if row.smem_max_threads >= 32 else f"{row.smem_max_threads}!"
        print(f"   {tag:>4} {row.window:>2} | {row.active:>6} {row.idle:>5} | {row.peak_bytes:>7} "
              f"{row.peak_bytes/REG_BYTES:>9.1f}x | {fitsR:>7} | {smem:>18}")

    print("\n   reading:")
    print("   * R (register-pack): 'fits R?'=YES means peak working set <= 1020 B -> NO spill in")
    print("     registers. Needs aggressive BFP (low bits) AND a tight unpack window (small W).")
    print("   * S (SMEM-stage): idle compressed set in SMEM; 'SMEM max thr/block' is the largest")
    print("     block that keeps it on-chip. >=64 (2 warps) is healthy; '!' = under one warp (dead).")
    print("   * W is the v7 scar knob: bigger unpack window -> bigger active set -> less benefit.")

    # --- the decisive verdict: SMEM-stage occupancy & eff vs M4 (which spills) -----------------
    print(f"\n>> VERDICT — SMEM-stage occupancy vs M4 (M4: {M4_WARPS} warps/SM, spills "
          f"{M4_SPILL_VALS:.0f} fp64 vals, eff:={M4_EFF:.3f})")
    print("   1 block/SM uses ~all SMEM; warps/SM = min(SMEM-limited, register-limited).")
    print(f"   {'BFP':>6} {'W':>2} | {'reg-frontier B':>14} {'warps/SM':>9} | {'eff (no spill)':>14} "
          f"{'vs M4':>7}")
    for W in (2, 4):
        for row in [r_ for r_ in r["rows"] if r_.window == W and r_.kbytes in (6, 4, 3, 2)]:
            frontier_b = row.active * FP64_B
            w_smem = row.smem_max_threads // WARP
            reg_threads = REGS_PER_SM // max(1, row.active * REGS_PER_FP64)
            w_reg = reg_threads // WARP
            w = max(1, min(w_smem, w_reg, MAX_WARPS_SM))
            frontier_fits = "" if frontier_b <= REG_BYTES else "  (frontier>regfile!)"
            eff = _occ(w)
            tag = f"{row.kbytes*8}-bit"
            print(f"   {tag:>6} {W:>2} | {frontier_b:>12} B {w:>9} | {eff:>14.3f} "
                  f"{eff/M4_EFF:>6.2f}x{frontier_fits}")
    print("   -> R (register-pack) is DEAD at usable precision (only BFP16/W0 fits 1020 B).")
    print("   -> S (SMEM-stage) is the path: the fp64 active frontier fits registers, the idle")
    print("      compressed bulk fits SMEM. But SMEM becomes the occupancy limiter, so the win is")
    print("      modest and rides on BFP width (more compression -> more warps -> more eff), which")
    print("      trades directly against accuracy. Decisive #/spill-relief await a Marylou A/B.")


if __name__ == "__main__":
    main()
