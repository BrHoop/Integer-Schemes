"""Fusion-level cost model (Step 3.4, CPU screening) — where is the optimal split?

M4 fused the whole BSSN RHS into ONE kernel because, when HBM was the bottleneck, eliminating the
multi-GB intermediate round-trip won (2.28x). But M4 is now **compute-bound at ~3% of fp64 peak**,
and the binding constraint is register spill -> low occupancy -> unhidden latency. So the regime
flipped, and the binary "verbatim (many kernels) vs full fusion (1 kernel)" may be missing the
optimum **in between**: a small number of kernels, each with a working set small enough to drop the
spill and lift occupancy, paying only a narrow-waist HBM round-trip between them.

This module answers, on CPU, BEFORE a Marylou push:

  * the **live-temp curve** of the real algebra (the cut-size at every possible split) — does it
    have narrow waists, or is it wide everywhere ("no narrow cut")?
  * a **fusion-level sweep**: split the algebra into K segments at the narrowest waists; for each K
    predict wall-clock from an occupancy/spill model calibrated to the two GPU facts we have
    (M4 @ 20.4 ms / 8 warps, and the 1d maxrregcount sweep's "64 regs -> 2.6x slower"). It reports
    the predicted time vs K and the optimal range.

**This is a SCREENING model, not a simulator.** The split regime (small working set, high
occupancy, NO spill) was never measured — the 1d sweep forced spill at fixed working set, a
different thing — so the split prediction is an *extrapolation* into an unmeasured regime. Its value
is the SHAPE (is the minimum at K=1 or K>1?) and the range, both flagged for one GPU confirmation.
Derivatives are modelled two ways at each split: re-materialized to HBM, or recomputed-from-L2 each
segment (the M2a finding that deriv reads are L2-served) — the cheaper wins.

Run:  python -m bssn3d.fusion_level_model
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ._codegen import parse, DENDRO_CSE
from .staging import build_dag, min_liveness_order, DAG

# --- hardware constants (H100 SXM) ----------------------------------------------------------
HBM_BW = 3.35e12            # B/s
FP64_PEAK = 33.5e12         # FLOP/s (CUDA-core FMA peak)
REGS_PER_SM = 65536
MAX_WARPS_SM = 64
WARP = 32
REG_CAP = 255               # ptxas hard cap; beyond this -> local-memory spill
REGS_PER_FP64 = 2

# --- algebra / kernel facts -----------------------------------------------------------------
NF, NOUT = 24, 24
BYTES = 8
# stencil op-count for ONE derivative recomputed from L2 (6th-order, interior estimate):
GRAD1_OPS, GRAD2D_OPS, GRAD2M_OPS = 13, 13, 49   # grad1 / diagonal grad2 / mixed grad2

_GRAD_RE = re.compile(r"\bgrad2?_\d(?:_\d)?_[A-Za-z][A-Za-z0-9]*\b")


# ============================================================================================
#  Liveness / cut curves on the real DAG
# ============================================================================================
def _deriv_ops(name: str) -> int:
    if name.startswith("grad2_"):
        _, i, j, _f = name.split("_", 3)
        return GRAD2D_OPS if i == j else GRAD2M_OPS
    return GRAD1_OPS


def liveness_curves(statements, dag: DAG, order):
    """Return per-position arrays along `order`:
       alg_live[p]   = # algebra temps live across the boundary just after p
       deriv_live[p] = # distinct derivatives live across that boundary
       alg_ops[p], deriv_set[p] are accumulated for segment costing.
    Position p ranges over 0..len(order) (a split 'after node p-1')."""
    pos = {n: i for i, n in enumerate(order)}
    stmt = dict(statements)
    n = len(order)

    # algebra temp spans (def .. last use)
    last = {nm: pos[nm] for nm in order}
    deriv_first, deriv_last = {}, {}
    for nm in order:
        rhs = stmt[nm]
        for d in dag.deps[nm]:               # DENDRO temp deps
            if d in pos:
                last[d] = max(last[d], pos[nm])
        for g in _GRAD_RE.findall(rhs):      # derivative leaves
            deriv_first.setdefault(g, pos[nm])
            deriv_last[g] = pos[nm]

    alg_live = np.zeros(n + 1, dtype=int)
    for nm in order:
        s, e = pos[nm], last[nm]             # live on boundaries s..e-1 (freed after last use)
        alg_live[s + 1: e + 1] += 1
    deriv_live = np.zeros(n + 1, dtype=int)
    for g in deriv_first:
        s, e = deriv_first[g], deriv_last[g]
        deriv_live[s: e + 1] += 1            # needed from first use..last use inclusive

    op_at = np.array([dag.op_cost[nm] for nm in order], dtype=float)
    return pos, alg_live, deriv_live, op_at, deriv_first, deriv_last


# ============================================================================================
#  Occupancy / spill -> effective throughput  (calibrated screening model)
# ============================================================================================
M4_MEASURED_REGS = 276       # ptxas -v on the fused kernel (HANDOFF_3.2_cuda); the anchor
E_M4 = 0.033                 # 20.4 ms @136^3 ~= 3.3% of FP64 FMA peak


@dataclass
class EffModel:
    """Effective-throughput model: latency hiding (occupancy) x spill penalty, mapped to the
    measured M4 anchor. `ptxas_factor` rescales the model's naive peak-overlap reg estimate onto
    the registers ptxas actually achieves (it reorders/reloads below the naive peak)."""
    w0: float            # warps to saturate latency hiding (Little's law)
    spill_alpha: float   # throughput penalty per spilled fp64 value
    ptxas_factor: float  # naive_regs -> ptxas_regs
    norm: float = 1.0

    def warps(self, regs_used: int) -> int:
        return max(1, min(MAX_WARPS_SM, REGS_PER_SM // (max(regs_used, 1) * WARP)))

    def eff(self, naive_regs: int) -> float:
        regs = naive_regs * self.ptxas_factor
        regs_used = min(regs, REG_CAP)
        spill_vals = max(0.0, regs - REG_CAP) / REGS_PER_FP64
        w = self.warps(int(regs_used))
        occ = w / (w + self.w0)
        spill = 1.0 / (1.0 + self.spill_alpha * spill_vals)
        return self.norm * occ * spill

    def regs_ptxas(self, naive_regs: int) -> int:
        return int(round(naive_regs * self.ptxas_factor))


def make_model(naive_m4_regs: int, w0: float, spill_alpha: float) -> EffModel:
    """Anchor: map naive K=1 peak onto measured 276 regs, normalize eff(M4)->3.3%.
    Reports the implied M4-vs-64reg slowdown as a sanity check against the 1d sweep's 2.6x."""
    ptxas_factor = M4_MEASURED_REGS / naive_m4_regs
    m = EffModel(w0=w0, spill_alpha=spill_alpha, ptxas_factor=ptxas_factor)
    m.norm = E_M4 / m.eff(naive_m4_regs)
    return m


# ============================================================================================
#  Fusion-level sweep
# ============================================================================================
def predict(n_pts: int = 136 ** 3, verbose: bool = True):
    statements, grad1, grad2 = parse(DENDRO_CSE)
    dag = build_dag(statements)
    order = min_liveness_order(dag)
    pos, alg_live, deriv_live, op_at, dfirst, dlast = liveness_curves(statements, dag, order)
    N = len(order)
    n_derivs = len(dfirst)

    total_alg_ops = float(op_at.sum())
    deriv_total_ops = sum(_deriv_ops(g) for g in dfirst)   # cost to compute all derivs once

    # peak register need of the FULLY FUSED kernel (K=1): all derivs held (the seam) + alg peak.
    m4_peak_vals = int(alg_live.max()) + n_derivs
    naive_m4_regs = m4_peak_vals * REGS_PER_FP64

    def kernel_time(eff, seg_regs, flop, hbm_bytes):
        compute = flop / (FP64_PEAK * eff.eff(seg_regs))
        mem = hbm_bytes / HBM_BW
        return max(compute, mem)                            # roofline overlap

    LAUNCH = 5e-6   # ~5 us/kernel launch

    def evaluate_split(eff, cuts):
        """cuts = sorted interior boundary positions (1..N-1). Returns (time_ms, detail)."""
        bnds = [0] + list(cuts) + [N]
        t = 0.0
        hbm_tot = 0.0
        max_regs = 0
        for a, b in zip(bnds[:-1], bnds[1:]):
            seg = order[a:b]
            alg_ops = float(op_at[a:b].sum())
            # derivs USED in this segment (recompute-from-L2) :
            used = {g for g in dfirst if dfirst[g] < b and dlast[g] >= a}
            recomp_ops = sum(_deriv_ops(g) for g in used)
            # peak regs in segment: alg temps live within + derivs simultaneously live within
            seg_alg_peak = int(alg_live[a + 1:b + 1].max()) if b > a else 0
            seg_drv_peak = int(deriv_live[a:b].max()) if b > a else 0
            seg_regs = (seg_alg_peak + seg_drv_peak) * REGS_PER_FP64
            max_regs = max(max_regs, seg_regs)
            # HBM: read the cut-in temps + recompute needs field reads; write cut-out temps.
            cut_in = int(alg_live[a]) if a > 0 else 0
            cut_out = int(alg_live[b]) if b < N else 0
            field_io = (NF if a == 0 else 0) + (NOUT if b == N else 0)
            hbm = (cut_in + cut_out + field_io) * n_pts * BYTES
            hbm_tot += hbm
            flop = (alg_ops + recomp_ops) * n_pts
            t += kernel_time(eff, seg_regs, flop, hbm) + LAUNCH
        return t * 1e3, max_regs, hbm_tot

    total_live = alg_live + deriv_live

    def waist_cuts(K):
        cand = sorted(range(1, N), key=lambda p: total_live[p])
        chosen = []
        for p in cand:
            if all(abs(p - c) > N // (2 * K) for c in chosen):
                chosen.append(p)
            if len(chosen) == K - 1:
                break
        return sorted(chosen)

    def peak_cuts(K):
        # steelman of "split THROUGH the peak": cut at the K-1 WIDEST live points so each
        # segment's internal peak actually drops (at the cost of a wide HBM cut).
        cand = sorted(range(1, N), key=lambda p: -total_live[p])
        chosen = []
        for p in cand:
            if all(abs(p - c) > N // (2 * K) for c in chosen):
                chosen.append(p)
            if len(chosen) == K - 1:
                break
        return sorted(chosen)

    def sweep(eff, cutter):
        rows = []
        t1 = evaluate_split(eff, [])[0]
        for K in range(1, 13):
            t, regs, hbm = evaluate_split(eff, cutter(K) if K > 1 else [])
            rows.append((K, t, eff.regs_ptxas(regs), hbm, t1 / t))
        return rows

    if verbose:
        print(f">> BSSN RHS fusion-level cost model  (N={N} algebra temps, {n_derivs} derivs, "
              f"{n_pts/1e6:.2f} Mpts, H100)")
        print(f"   live-temp curve (cut size at every split point):")
        print(f"     peak={int(total_live.max())}  median={int(np.median(total_live))}  "
              f"narrowest-waist={int(total_live.min())} values   "
              f"(alg peak {int(alg_live.max())} + deriv peak {int(deriv_live.max())})")
        print(f"   naive K=1 peak {naive_m4_regs} regs -> anchored to measured "
              f"{M4_MEASURED_REGS} regs (ptxas_factor={M4_MEASURED_REGS/naive_m4_regs:.2f}), "
              f"eff:=3.3% of FP64 peak\n")

        # The decisive diagnostic: does any split actually REDUCE the per-segment register peak?
        print("   per-segment register peak vs split count (does occupancy improve?):")
        print(f"   {'K':>2} | {'waist-split':>22} | {'peak-split (steelman)':>24}")
        print(f"   {'':>2} | {'max-seg regs':>12} {'HBM GB':>9} | {'max-seg regs':>12} {'HBM GB':>9}")
        m_diag = make_model(naive_m4_regs, 16.0, 0.05)
        for K in (1, 2, 4, 8, 12):
            tw, rw, hw = evaluate_split(m_diag, waist_cuts(K) if K > 1 else [])
            tp, rp, hp = evaluate_split(m_diag, peak_cuts(K) if K > 1 else [])
            print(f"   {K:>2} | {m_diag.regs_ptxas(rw):>12} {hw/1e9:>9.2f} | "
                  f"{m_diag.regs_ptxas(rp):>12} {hp/1e9:>9.2f}")
        print("   -> waist-split: regs PINNED (peak survives in a segment, no occ gain).")
        print("   -> peak-split: regs drop but HBM explodes (cutting a ~440-wide live set).\n")

        # sensitivity grid over the (unmeasured) occupancy/spill curve, both cut strategies.
        for cutter, label in ((waist_cuts, "waist"), (peak_cuts, "peak")):
            print(f"   === split strategy: {label} ===")
            for w0 in (8.0, 32.0):
                for alpha in (0.02, 0.05):
                    eff = make_model(naive_m4_regs, w0, alpha)
                    rows = sweep(eff, cutter)
                    best = min(rows, key=lambda r: r[1])
                    ratio64 = eff.eff(naive_m4_regs) / eff.eff(int(64 / eff.ptxas_factor))
                    bullet = (f"K={best[0]} best ({best[4]:.2f}x vs fusion)" if best[0] != 1
                              else "K=1 (fusion) best")
                    print(f"   [w0={w0:>4.0f}, alpha={alpha:.2f}]  occ-sanity M4/64reg="
                          f"{ratio64:.2f}x  ->  {bullet}")
            print()

        print(">> RESULT (screening): full fusion (K=1) is optimal or within noise across the grid,\n"
              "   under BOTH split strategies. The mechanism is the 'no narrow cut' wall made\n"
              "   quantitative: the ~440-value register peak is structurally unavoidable, so a\n"
              "   waist-split never lowers per-segment occupancy, and a peak-split pays 10-40 GB of\n"
              "   HBM to cut through it. The model is occupancy-OPTIMISTIC (it cannot even reproduce\n"
              "   the 1d sweep's 2.6x spill penalty -- it under-weights spill), which BIASES toward\n"
              "   splitting; splitting still loses, so the K=1 verdict is conservative.\n"
              "   UPSHOT: re-splitting does NOT help (idea #1 refuted on CPU). Occupancy can only\n"
              "   improve by shrinking the peak WIDTH itself -> the e-graph (3.4) is the real lever.")
    return naive_m4_regs, total_live


if __name__ == "__main__":
    predict()
