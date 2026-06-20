"""Fused peak-live cost model (Step 3.2-seam, CPU): does the 2.5D+algebra FUSED kernel fit?

The Step-3.2 CUDA work declared wall B "solved" — the naive direct-CUDA *algebra* (1c) spills
only ~4 KB (255 regs, 0.948 ms). **But that is the STANDALONE algebra: the 138 derivatives were
HBM *inputs* there, so `ptxas` streams them from L1/L2 on demand and they never press the
register file** (`staging.build_dag` mirrors this — it treats `grad_*`/`grad2_*` as leaves, the
"derivative-read reserve", excluded from the live set).

In the FUSED RHS (Step 3.3) the 2.5D stage produces the 138 derivatives **on-chip**, and the
point-wise algebra needs them **jointly** (Christoffels multiply derivatives of *different*
fields, so you cannot stream field-by-field through the algebra). So the derivatives are all
live when the algebra starts and must be *held* until their last use. This module promotes them
from leaves to **held-live nodes** and answers, entirely on CPU (no GPU/2FA):

  1. **The seam (headline).** How far past the 255-register file does the fused working set go
     if the derivatives are held in registers (the naive fusion)?  → does the seam reopen wall B?
  2. **The lever.** Derivatives belong in a *different* on-chip pool — **SMEM tiles** — which do
     not press the register file. But all 138 deriv tiles fit SMEM only at a *small* slab width
     T (`138 · T² · 8 B`). This module sweeps T and reports the feasible (T, residency) frontier:
     SMEM-tile vs recompute-from-window vs register-hold, with the halo-redundancy price of T.
  3. **The verdict.** The largest feasible slab + its register/SMEM/redundancy cost — the budget
     Step 3.2e should be built to (and a check on the handoff's T≈32, which never accounted for
     where the 138 deriv *outputs* live).

Recompute cost only enters the recompute-policy branch (the SMEM/register verdict is independent
of it); the stencil op-counts below are 6th-order-interior estimates, clearly flagged.

Run:  `python -m bssn3d.fused_peak_model`
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ._codegen import parse, DENDRO_CSE
from .staging import build_dag, _peak_overlap, min_liveness_order, DAG
from . import gpu_profiles

# --- hardware / scheme constants -----------------------------------------------------------
# Hardware numbers are SOURCED from the gpu_profiles registry (single source of truth) rather
# than hard-coded here. This module's deep seam analysis is reported for H200 (the default);
# the per-GPU slab frontier lives in `gpu_profiles.recommend` / its `main` table. To retarget,
# pass a different profile's `.smem_per_block` etc., or run `python -m bssn3d.gpu_profiles`.
_GPU = gpu_profiles.PROFILES[gpu_profiles.DEFAULT]   # H200
REG_FILE = _GPU.regs_per_thread          # 255 — per-thread architectural register file
REG_PER_FP64 = 2                         # an fp64 value occupies two 32-bit registers
NG = 4                                   # production ghost width (8th-order KO) -> 2*NG+1 planes
BYTES = 8                                # fp64
SMEM_CAP = _GPU.smem_per_block           # max dynamic SMEM per block (H200: ~227 KB)
REGS_PER_SM = _GPU.regs_per_sm           # 32-bit register file per SM
THREADS_PER_SM = _GPU.threads_per_sm     # hardware thread cap per SM

# Recompute op-cost estimates per derivative evaluation (6th-order centred interior; the 1/dx^n
# folds into the stencil coefficients). ONLY used by the recompute-policy branch — the SMEM vs
# register verdict does not depend on these. Mixed 2nd-derivs are a separable double pass.
STENCIL_OPS = {"g1": 11, "g2_diag": 13, "g2_mixed": 77}

_G1 = re.compile(r"\bgrad_(\d)_([A-Za-z][A-Za-z0-9]*)\b")
_G2 = re.compile(r"\bgrad2_(\d)_(\d)_([A-Za-z][A-Za-z0-9]*)\b")


# --- derivative model ------------------------------------------------------------------------
@dataclass
class Deriv:
    """One distinct `grad_*`/`grad2_*` input, promoted from a leaf to a held-live node.

    ``uses`` are the positions (in a given algebra evaluation order) of the statements that
    reference it. In the fused kernel a held derivative is live from algebra-start (it is produced
    in the preceding 2.5D stage) to its ``last`` use.
    """
    name: str
    fld: str
    family: str            # 'g1' | 'g2_diag' | 'g2_mixed'
    uses: List[int] = field(default_factory=list)

    @property
    def fanout(self) -> int:
        return len(self.uses)

    @property
    def last(self) -> int:
        return max(self.uses)

    @property
    def stencil_ops(self) -> int:
        return STENCIL_OPS[self.family]


def extract_derivs(statements, order: List[str]) -> Dict[str, Deriv]:
    """Map each `grad_*`/`grad2_*` token to a :class:`Deriv` with its use-positions in ``order``."""
    pos = {n: i for i, n in enumerate(order)}
    out: Dict[str, Deriv] = {}
    for lhs, rhs in statements:
        if lhs not in pos:           # outputs/temps not in this order (defensive; all are)
            continue
        p = pos[lhs]
        for a, f in _G1.findall(rhs):
            name = f"grad_{a}_{f}"
            out.setdefault(name, Deriv(name, f, "g1")).uses.append(p)
        for i, j, f in _G2.findall(rhs):
            name = f"grad2_{i}_{j}_{f}"
            fam = "g2_diag" if i == j else "g2_mixed"
            out.setdefault(name, Deriv(name, f, fam)).uses.append(p)
    return out


# --- liveness ---------------------------------------------------------------------------------
def algebra_spans(dag: DAG, order: List[str]) -> List[Tuple[int, int]]:
    """(def, last-use) span of every algebra temp + output under ``order`` (derivatives excluded —
    they are leaves in ``dag.deps``; handled separately as held nodes)."""
    pos = {n: i for i, n in enumerate(order)}
    last = {n: pos[n] for n in order}
    for n in order:
        for d in dag.deps[n]:
            if d in pos:
                last[d] = max(last[d], pos[n])
    return [(pos[n], last[n]) for n in order]


def fused_peak_values(dag: DAG, order: List[str], derivs: Dict[str, Deriv],
                      reg_names) -> int:
    """Peak simultaneously-live *values* (register pool) = algebra temps + register-held derivs.

    Register-held derivs are produced in the 2.5D stage (before the algebra) so each is live from
    algebra position 0 to its last use. SMEM/recompute derivs do not appear (different pool /
    transient)."""
    reg = set(reg_names)
    spans = list(algebra_spans(dag, order))
    spans += [(0, derivs[n].last) for n in reg]
    return _peak_overlap(spans)


# --- resource accounting ----------------------------------------------------------------------
def field_window_bytes(T: int) -> int:
    """One field's 2.5D SMEM window: (T+2NG)² in-plane halo × (2NG+1) z-planes, fp64."""
    return (T + 2 * NG) ** 2 * (2 * NG + 1) * BYTES


def deriv_tile_bytes(n: int, T: int) -> int:
    """``n`` derivative outputs each stored as a T² SMEM tile, fp64."""
    return n * T * T * BYTES


def redundancy(T: int) -> float:
    """In-plane halo recompute factor of a z-marched slab: ((T+2NG)/T)²."""
    return ((T + 2 * NG) / T) ** 2


def wgmma_aligned(T: int) -> bool:
    """Does the slab's T² point-batch fill whole WGMMA M=64 tiles (Ozaki INT8, Phase 4)?

    The Hopper warpgroup MMA wants M=64; the point-batch (the dim we control) is T², so T a
    multiple of 8 → T² a multiple of 64 → no ragged tile. (The derivative *contraction* is
    K=7/9 — small-K, the 4.6× zero-waste that shelved FD-as-GEMM — so tensor cores are the
    Phase-4 secondary lever and do NOT re-rank the slab; this is a tiebreaker only.)"""
    return (T * T) % 64 == 0


def peak_smem_bytes(n_smem: int, T: int, n_recompute_fields: int = 0) -> int:
    """Peak SMEM of the fused kernel.

    phase 1 (derivative stage, field-streamed): accumulated deriv tiles + ONE field window.
    phase 2 (algebra): the deriv tiles + windows of any field whose derivs are *recomputed*
    inline (those windows must persist into the algebra).
    """
    tiles = deriv_tile_bytes(n_smem, T)
    phase1 = tiles + field_window_bytes(T)
    phase2 = tiles + n_recompute_fields * field_window_bytes(T)
    return max(phase1, phase2)


def blocks_per_sm(smem: int) -> int:
    """How many blocks co-reside on one SM given per-block SMEM (SMEM-limited only)."""
    return max(0, SMEM_CAP // max(smem, 1))


# --- policy frontier --------------------------------------------------------------------------
@dataclass
class FusedResult:
    T: int
    policy: str                 # 'all-reg' | 'all-smem' | 'hybrid'
    reg_peak_values: int        # peak live values in the register pool
    reg_peak_fp64: int          # × REG_PER_FP64
    n_smem: int
    smem_bytes: int
    smem_fits: bool
    blocks_sm: int
    redundancy: float
    recompute_flops: int
    fits: bool                  # smem fits AND register peak ≈ standalone (algebra-only spill)


def all_smem(dag, order, derivs, T) -> FusedResult:
    """Store every derivative as a SMEM tile; the algebra stays in registers (its standalone
    ~4 KB spill, unchanged). Feasible only while 138 tiles + a field window fit SMEM."""
    n = len(derivs)
    smem = peak_smem_bytes(n, T)
    reg_peak = fused_peak_values(dag, order, derivs, reg_names=[])   # algebra only
    return FusedResult(
        T=T, policy="all-smem",
        reg_peak_values=reg_peak, reg_peak_fp64=reg_peak * REG_PER_FP64,
        n_smem=n, smem_bytes=smem, smem_fits=smem <= SMEM_CAP,
        blocks_sm=blocks_per_sm(smem), redundancy=redundancy(T),
        recompute_flops=0, fits=smem <= SMEM_CAP,
    )


def all_reg(dag, order, derivs, T) -> FusedResult:
    """The naive fusion: hold every derivative in registers. The seam headline."""
    reg_peak = fused_peak_values(dag, order, derivs, reg_names=list(derivs))
    return FusedResult(
        T=T, policy="all-reg",
        reg_peak_values=reg_peak, reg_peak_fp64=reg_peak * REG_PER_FP64,
        n_smem=0, smem_bytes=field_window_bytes(T), smem_fits=True,
        blocks_sm=blocks_per_sm(field_window_bytes(T)), redundancy=redundancy(T),
        recompute_flops=0, fits=reg_peak * REG_PER_FP64 + 0 <= REG_FILE,
    )


def max_smem_derivs_at_T(T: int) -> int:
    """How many deriv tiles fit SMEM alongside one field window at slab width ``T``."""
    avail = SMEM_CAP - field_window_bytes(T)
    if avail <= 0:
        return 0
    return avail // (T * T * BYTES)


def recompute_is_smem_favorable(T: int, derivs_of_field: int) -> bool:
    """Recomputing one field's derivs (freeing their tiles) costs that field's window resident in
    phase 2. SMEM-favorable iff the freed tiles exceed the window — almost never (a window is a
    (T+2NG)²·(2NG+1) volume; the tiles it frees are derivs_of_field · T²)."""
    freed = derivs_of_field * T * T * BYTES
    cost = field_window_bytes(T)
    return freed > cost


# --- report -----------------------------------------------------------------------------------
def main() -> None:
    statements, grad1, grad2 = parse(DENDRO_CSE)
    dag = build_dag(statements)
    file_order = dag.order
    reorder = min_liveness_order(dag)               # ptxas-achievable reorder
    derivs = extract_derivs(statements, file_order)

    n_g1 = sum(1 for d in derivs.values() if d.family == "g1")
    n_diag = sum(1 for d in derivs.values() if d.family == "g2_diag")
    n_mixed = sum(1 for d in derivs.values() if d.family == "g2_mixed")
    fields_used = sorted({d.fld for d in derivs.values()})

    # standalone algebra peak (no derivs held) — the 1c register working set
    alg_file = _peak_overlap(algebra_spans(dag, file_order))
    alg_reord = _peak_overlap(algebra_spans(dag, reorder))

    print("=" * 90)
    print("BSSN fused peak-live model — does the 2.5D + algebra FUSED kernel fit the register file?")
    print("=" * 90)
    print(f"algebra DAG: {len(file_order)} statements ({len(dag.temps)} temps + "
          f"{sum(dag.is_output.values())} outputs)")
    print(f"derivatives held live by fusion: {len(derivs)}  "
          f"(grad1={n_g1}, grad2_diag={n_diag}, grad2_mixed={n_mixed}); "
          f"{len(fields_used)} distinct fields, {sum(d.fanout for d in derivs.values())} uses")
    print(f"STANDALONE algebra peak-live (derivs = HBM inputs, the 1c regime): "
          f"file {alg_file} val = {alg_file*REG_PER_FP64} regs, "
          f"reorder {alg_reord} val = {alg_reord*REG_PER_FP64} regs")
    print(f"  → already {alg_reord*REG_PER_FP64//REG_FILE}× the {REG_FILE}-reg file; ptxas holds "
          f"~{REG_FILE//REG_PER_FP64} val resident and spills the rest = the measured ~4 KB in 1c.")
    print("  (NB: this model reports register PRESSURE in values/regs — a clean, ptxas-independent\n"
          "   number. It does NOT estimate spill-KB or ms; that translation is ptxas's, deliberately\n"
          "   not faked. The claim here is *relative* pressure, which is rigorous.)\n")

    # --- 1. the seam: hold all derivs in registers --------------------------------------------
    print("-" * 90)
    print("1. THE SEAM — naive fusion holds all 138 derivs in registers (1c assumption broken)")
    print("-" * 90)
    for label, order, base in (("file-order", file_order, alg_file),
                               ("ptxas-reorder", reorder, alg_reord)):
        d = extract_derivs(statements, order)
        peak = fused_peak_values(dag, order, d, reg_names=list(d))
        print(f"  {label:14s}: fused peak-live = {peak} val = {peak*REG_PER_FP64} fp64 regs "
              f"= {peak*REG_PER_FP64/REG_FILE:.1f}× the file (+{peak-base} val of held derivs "
              f"vs standalone)")
    seam_peak = fused_peak_values(dag, reorder, extract_derivs(statements, reorder),
                                  reg_names=list(derivs))
    added = seam_peak - alg_reord
    print(f"  → seam verdict: fusion ADDS ~{added} held-deriv values (≈{added*REG_PER_FP64} fp64 "
          f"regs) of pressure that 1c never had (derivs were HBM inputs there). The fused working\n"
          f"    set jumps {alg_reord}→{seam_peak} val; the algebra alone already overflowed, and "
          f"every added deriv is over-budget → STRICTLY more spill than 1c, in a >4× regime ptxas\n"
          f"    was never measured in. 'Wall B mild' cannot be assumed for the fused kernel. "
          f"WALL B REOPENS.\n")

    # --- 2. the lever: derivs to SMEM, sweep slab width T -------------------------------------
    print("-" * 90)
    print("2. THE LEVER — store all 138 derivs as SMEM tiles (off the register file). Sweep T.")
    print("    (register pool then = the standalone algebra → same ~4 KB spill as 1c)")
    print("-" * 90)
    print(f"  {'T':>3} {'deriv-tiles':>12} {'+window':>9} {'peakSMEM':>10} {'fit?':>5} "
          f"{'blk/SM':>7} {'redund':>7} {'WGMMA':>6}")
    feasible_T = []
    for T in (6, 8, 10, 12, 13, 14, 16, 24, 32, 48, 64):
        r = all_smem(dag, file_order, derivs, T)
        tiles = deriv_tile_bytes(len(derivs), T)
        win = field_window_bytes(T)
        mark = "yes" if r.smem_fits else "NO"
        if r.smem_fits:
            feasible_T.append(T)
        wg = "M=64" if wgmma_aligned(T) else "—"
        print(f"  {T:3d} {tiles/1024:10.1f}KB {win/1024:7.1f}KB {r.smem_bytes/1024:8.1f}KB "
              f"{mark:>5} {r.blocks_sm:7d} {r.redundancy:6.2f}x {wg:>6}")
    feasible_aligned = [T for T in feasible_T if wgmma_aligned(T)]
    Tmax = max(feasible_T) if feasible_T else None
    print(f"  → all-SMEM feasible up to T = {Tmax} (redundancy {redundancy(Tmax):.2f}x).  "
          f"Register peak unchanged at {alg_reord*REG_PER_FP64} regs → same ~4 KB spill as 1c.")
    print(f"  → only WGMMA-aligned feasible slab = T={feasible_aligned} (T² fills M=64 tiles) — "
          f"the Phase-4 tensor-core tiebreaker favoring T=8 over T=13.\n")

    # --- 3. at the handoff's T=32: how much CAN go to SMEM, and is the rest survivable? --------
    print("-" * 90)
    print("3. THE HANDOFF's T=32 SLAB — can it host its own deriv outputs?")
    print("-" * 90)
    n_fit_32 = max_smem_derivs_at_T(32)
    print(f"  at T=32: one field window = {field_window_bytes(32)/1024:.0f} KB; SMEM left for "
          f"tiles = {(SMEM_CAP-field_window_bytes(32))/1024:.0f} KB → only {n_fit_32} of 138 "
          f"derivs fit SMEM.")
    rest = len(derivs) - n_fit_32
    # the other `rest` must be REG (→ spill) or RECOMPUTE (→ needs their fields resident)
    reg_peak_32 = fused_peak_values(dag, reorder, extract_derivs(statements, reorder),
                                    reg_names=list(derivs)[:rest])
    print(f"  the other {rest} held in registers → fused peak {reg_peak_32*REG_PER_FP64} regs "
          f"(≫255).  recompute instead → needs ≤{len(fields_used)} field windows resident = "
          f"{len(fields_used)*field_window_bytes(32)/1024/1024:.1f} MB ≫ {SMEM_CAP/1024:.0f} KB.")
    print(f"  → T=32 is INFEASIBLE for the fused kernel: the 138 deriv outputs fit neither SMEM "
          f"nor registers, and recompute can't hold the windows. The handoff §5.2 slab is too "
          f"wide.\n")

    # --- 4. is recompute ever SMEM-favorable? (kills the recompute option here) ----------------
    print("-" * 90)
    print("4. RECOMPUTE vs SMEM-tile — is recomputing a field's derivs ever SMEM-cheaper?")
    print("-" * 90)
    # max derivs any single field contributes (3 grad1 + up to 6 grad2)
    per_field = {}
    for d in derivs.values():
        per_field[d.fld] = per_field.get(d.fld, 0) + 1
    maxd = max(per_field.values())
    for T in (8, 12, 32):
        fav = recompute_is_smem_favorable(T, maxd)
        print(f"  T={T:2d}: freeing one field's {maxd} tiles = "
              f"{maxd*T*T*BYTES/1024:.1f} KB vs its window {field_window_bytes(T)/1024:.1f} KB "
              f"→ recompute {'WINS' if fav else 'LOSES'} on SMEM")
    print("  → a 2.5D field window is a (T+2NG)²·9 volume; it always exceeds the few T² tiles it\n"
          "    would free. Recompute-from-window is SMEM-unfavorable → STORE derivs in SMEM tiles.\n")

    # --- 5. occupancy: the small slab × the 255-reg algebra -----------------------------------
    print("-" * 90)
    print("5. OCCUPANCY — the SMEM-forced small slab vs the register-bound algebra")
    print("-" * 90)
    print(f"  the fused kernel runs T² threads/block, each carrying the 255-reg algebra. Two caps:")
    print(f"  {'T':>3} {'threads/blk':>11} {'blk/SM(SMEM)':>13} {'blk/SM(reg)':>12} "
          f"{'threads/SM':>11} {'warps/SM':>9} {'occ':>6}")
    for T in [t for t in (8, 10, 12, 13) if t in feasible_T]:
        thr_blk = T * T
        smem = all_smem(dag, file_order, derivs, T).smem_bytes
        blk_smem = blocks_per_sm(smem)
        blk_reg = min(REGS_PER_SM // (thr_blk * REG_FILE) if thr_blk * REG_FILE else 99,
                      THREADS_PER_SM // thr_blk)
        blk = min(blk_smem, blk_reg)
        thr_sm = blk * thr_blk
        warps = thr_sm / 32
        print(f"  {T:3d} {thr_blk:11d} {blk_smem:13d} {blk_reg:12d} {thr_sm:11d} "
              f"{warps:8.1f} {thr_sm/THREADS_PER_SM:5.0%}")
    print(f"  → the 255-reg algebra caps any design at ~{REGS_PER_SM//REG_FILE} threads/SM "
          f"(~{REGS_PER_SM//REG_FILE//32} warps) REGARDLESS of slab — the 1d low-occ regime. "
          f"That was\n    fine for the register-bound algebra; the OPEN RISK is whether the "
          f"latency-bound derivative\n    stage (sharing these threads) can hide HBM field-load "
          f"latency at ~4-5 warps/SM. If not,\n    the fused single-kernel design pays there — the "
          f"regime-transfer caveat, now quantified.\n")

    # --- verdict ------------------------------------------------------------------------------
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"• The seam is REAL: naive fusion = {seam_peak} val = {seam_peak*REG_PER_FP64} regs "
          f"({seam_peak*REG_PER_FP64/REG_FILE:.1f}× the file), +{added} deriv values over 1c. "
          f"1c's 'wall B mild' was the standalone algebra only (derivs = HBM inputs).")
    print(f"• The fix is SMEM-tile residency (different pool than registers), feasible up to "
          f"T={Tmax}. Recompute-from-window is SMEM-unfavorable; register-hold spills.")
    print(f"• Recommended fused slab: small T (T≈{Tmax}: 1 block/SM, "
          f"{redundancy(Tmax):.2f}x halo) OR T=8 (2 blocks/SM, {redundancy(8):.2f}x halo, "
          f"{all_smem(dag,file_order,derivs,8).smem_bytes/1024:.0f} KB) — the occupancy-vs-"
          f"redundancy pick needs the GPU.")
    print(f"• Build Step 3.2e to a T≤{Tmax} slab with all derivs in SMEM; the register pool then "
          f"carries only the algebra (unchanged ~4 KB spill). The handoff's T≈32 is ruled out.")
    print("=" * 90)


if __name__ == "__main__":
    main()
