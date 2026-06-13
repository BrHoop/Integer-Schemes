# algebra.md — BSSN RHS register-pressure & roofline conclusions

**Status: theoretical / CPU-side analysis (2026-06-12). Unvalidated on H200 —
the measurement gate is below.** Written to be cross-checked against the parallel
"algebra" chat, which concluded BSSN is memory-bound (see *Reconciliation*).

All numbers come from a dataflow analysis of the **exact** SSA the production RHS
is generated from (`bssn3d/_codegen.parse` of `bssneqs_SSL_HD_dxsq.cpp`, the
CAHD+SSL variant), so they cannot drift from the emitted `_bssn_rhs_generated.py`.
Method validated: it reproduces the documented **584** peak-live exactly on the
no-CAHD `bssneqs_sympy_cse_wo_derivs.cpp`.

---

## 1. The register spill is STRUCTURAL, not an ordering artifact

Production CAHD+SSL RHS: 826 temps + 24 outputs. Peak simultaneously-live values
(register pressure), via inclusive-interval analysis:

| order | temps-only | with leaf inputs |
|---|---|---|
| SSA file order | 452 | 543 |
| Sethi-Ullman depth-first reschedule | **394** | 438 |

- **Scheduling buys ~13%** (452→394) and is worth taking, but gets nowhere near
  the **~127 fp64-value** register budget (255 regs / 2). Pressure is structural.
- **The floor is a shared first-order tensor trunk.** 128 temps feed ≥12 of the 24
  outputs; the 128-trunk is **100% first-order** (inverse metric, ~27 Christoffels,
  gauge/χ web) — distinct shared tensor components, *not* accumulable sums.

## 2. What does NOT fix it

- **Single-cut kernel fission: dead.** No narrow dataflow waist — a balanced 50/50
  cut round-trips ~362 temps; even 25/75 needs ~204. The only narrow cuts (8–18
  temps) are pathologically lopsided (2%/97% through). The trunk is genuinely wide.
- **Ricci accumulation loop barely helps registers (394 → ~385).** This refutes the
  prior "Ricci needs 36 grad2 co-resident → it's the floor" assumption: only 28/826
  temps are Ricci(grad2-of-metric)-family, the 128-trunk contains **zero** Ricci
  temps, and at the 394 peak only 15 Ricci temps / 4-of-36 metric-2nd-deriv leaves
  are co-live. The CSE/scheduler already serializes that contraction. Ceiling on the
  loop's register benefit ≈ 19 temps.
  - **Still worth doing the Ricci loop — for a different reason:** accumulating over
    (l,m) lets the *derivative* stage stream the 36 grad2-of-metric inputs instead
    of materializing all 36, shrinking the **derivative-bundle** working set. That's
    a separate kernel/bottleneck, not the algebra registers.
- **fp32 on Ricci: declined** (user — catastrophic-cancellation risk in the
  large-term differences; not pursued).

## 3. What DOES fix it: materialize the trunk to SMEM + recompute the bulk

Using `staging.persistent_liveness` / `recompute_ops`:

| materialize | \|M\| | peak live | recompute ops | ×verbatim |
|---|---|---|---|---|
| trunk (feed ≥12 outputs) | 128 | **123** | 31,489 | **6.96×** |
| feed ≥8 | 201 | 180 | 28,982 | 6.40× |
| feed ≥6 | 678 | 416 | 4,991 | 1.10× |

Materialize the ~128 first-order trunk, **recompute** the ~650-temp bulk → peak
live 123 at ~7× recompute. Sharp cliff between trunk (≤201, ≥8 outputs) and bulk
(jumps to 678 at ≥6) — the trunk is a well-defined object.

- **Trunk → SMEM is mandatory:** 128 fp64 ≈ 256 registers > the whole file, and
  fp32 is off the table. 1 KB/point, free in a pointwise kernel (no field halo
  resident). Recompute transients live in registers.
- **Recompute is the memory→compute operator, not a cost.** It multiplies FLOP
  without adding HBM bytes → raises arithmetic intensity. This is the user's
  "do enough redundant compute to balance" instinct, confirmed: there is ample
  duplicate math (688 temps reused by ≥4 outputs).
- The selector should be **output-fanout ≥ K**, not the current
  `staging.select_cut_set` fan-out×cone-cost heuristic (the floor analysis shows
  output-fanout is the correct trunk discriminator).

## 4. Roofline — and why the DERIVATIVE handoff decides the regime

H200 FP64 machine balance ≈ 14 FLOP/B (non-TC; tensor cores ~2×). Derivative stage
≈ 2,070 FLOP/pt (138 stencils × ~15). ops≈FLOP (transcendentals undercounted; rough
but regime-deciding).

| algebra recompute | AI, separate kernels | AI, fused (derivs in SMEM) |
|---|---|---|
| none (materialize all → spills) | 2.4 | 21.5 |
| **7× (trunk strategy)** | **12.1** | **91.7** |

- **Algebra alone, de-spilled + 7×: compute-bound.** ✓
- **Whole RHS, separate deriv+algebra kernels: AI ≈ 12 → marginally MEMORY-bound**,
  even at 7× recompute. The 138 derivatives round-trip through HBM (~2,200 B/pt —
  the largest single HBM item) and drag the aggregate under machine balance.
- **Whole RHS, fused (138 derivatives kept on-chip in SMEM, never HBM): AI ≈ 92 →
  strongly COMPUTE-bound.** The on-chip derivative handoff is the make-or-break.

## 4b. The derivative stage is WRITE-bound — no operator fixes it standalone

Can the derivative stage be made compute-bound *as a separate kernel*, without
fusing it into the algebra? **No, at BBH patch sizes.** A standalone derivative
kernel reads 24 fields once (192 B/pt) and **writes 138 derivative fields
(1104 B/pt)**. The write is the wall — compute-bound needs >18,144 FLOP/pt:

| operator | FLOP/pt | AI | regime |
|---|---|---|---|
| explicit-6 | 2,070 | 1.6 | memory |
| explicit-10 | 3,174 | 2.4 | memory |
| compact/Padé GEMM (patch N=32) | 4,416 | 3.4 | memory |
| compact/Padé GEMM (patch N=64) | 8,832 | 6.8 | memory |

- Compact-as-GEMM only crosses machine balance at **patch side N > 130**; BBH AMR
  patches are ~16–48. Higher order adds FLOP but still writes 138 outputs. Input
  reuse touches only the 192 B read. **Tensor cores are useless on a write-bound
  kernel** (they accelerate compute, not the write).
- ⇒ **"compute-bound derivatives" and "no fusion" are in direct conflict.** The
  only way off the 1104 B write is to consume the 138 derivatives on-chip; since
  the pointwise algebra needs all 138 co-resident, there is no incremental trick.
- **Lightest viable middle = output *contracted* 2nd-deriv tensors** (e.g.
  g̃^lm ∂_l∂_m g̃_ij) from the derivative kernel instead of 66 raw grad2 → write
  drops ~30–50% (needs inverse metric in the deriv kernel = mild fusion). The
  full fix is spatial deriv→algebra fusion within one RK substage (NOT temporal
  across-substage fusion, which is correctly rejected).
- **BBH lens:** dense operators (compact/spectral) are doubly disfavored — small
  patches worsen AI, moving-puncture Gibbs, line-global boundary coupling into the
  unsolved outer-BC strategy. Explicit FD + KO locality is what BBH wants.

## 5. Derivative-stage levers (ranked by impact)

1. **Keep the 138 derivatives on-chip (primary — a MEMORY lever).** Fused tiled
   kernel: compute a tile's derivatives into SMEM, algebra recompute consumes them
   there. This is the AI 12→92 difference. SMEM at BS=8: ~70 KB derivs + ~65 KB
   trunk ≈ 135 KB < 228 KB — fits but tight. Small tiles raise stencil halo
   redundancy, but derivative FLOP is small (~2k of ~33k total) → affordable.
2. **Halo reuse in the stencil (2.5D SMEM streaming)** — already the chosen design.
3. **Tensor-core / Ozaki stencil-as-GEMM (thesis headline, but tertiary).** A
   stencil is low-AI (memory-bound); TC accelerates compute, useless until the
   stage is compute-bound. Post-fusion it's a small slice → Amdahl-capped ~1.4×.
   Banded explicit stencil wastes 4.6× on zeros (bad GEMM target); compact-FD dense
   form is better but carries boundary issues.

**Through-line:** the wins are locality/memory engineering (de-spill + on-chip
derivative handoff); the tensor-core/compute story is a capped topping on a kernel
that *locality* made compute-bound.

## 6. Reconciliation with the parallel "algebra" chat

That chat concluded BSSN is memory-bound, confidently. This is **consistent** if it
characterized the **as-realized** RHS (XLA splits the algebra into ~79 HBM-backed
kernels) or the **separate-kernel** structure (AI ≈ 12). Our claim is narrower and
about the **achievable ceiling**: compute-bound *iff* (a) de-spill via SMEM-trunk +
recompute AND (b) the derivative handoff is fused on-chip. **If the other chat found
a *fundamental* (not realization-specific) memory bound, that overrides this — to be
checked.**

## 6b. fp64 fused-kernel SMEM budget (CPU-derived, 2026-06-12)

The committed 3.2d/H200 gate assumes `BSSN_PALLAS_FP32=1` (fp32 fits the 123-value
trunk in 255 regs). **fp32 is declined (Ricci cancellation) → the fp64 + SMEM-trunk
variant must be probed instead.** Budget for the fused 2.5D kernel (per-point working
set 138 derivs + 128 trunk = 266 fp64 = 532 regs ≫ 255 → SMEM-resident, forced):

| BS | deriv+trunk SMEM | + stream halo | total | blocks/SM | fits 228 KB? |
|---|---|---|---|---|---|
| 4 | 33 KB | 10 KB | 43 KB | 5 | yes |
| 8 | 133 KB | 18 KB | 151 KB | **1** | yes |
| 12 | 299 KB | 28 KB | 327 KB | 0 | **no** |

- **fp64 is feasible (BS ≤ 8) — the no-fp32 path is not blocked.**
- But BS=8 → **1 block/SM = low occupancy**; the compute-bound bet rides on **ILP**,
  not occupancy. This is a NEW H200 question: does SM% climb at 1 block/SM, or do the
  FP units starve for warps? (gate currently checks only spill→0 + regime flip.)
- Pre-contraction (138→108 derivs) buys 151→136 KB at BS=8 — occupancy headroom +
  less halo redundancy, not a bigger tile.
- **Action: re-point `step_3.2_phase2_h200.md` from fp32/BS=32 to fp64/BS=8** — the
  fp32/BS=32 kernel cannot fit and would be the wrong probe target.

## 6c. Implementation landed (2026-06-12) — additive, fp32 path untouched

The fp64 + SMEM-trunk path now has a CPU-validated foundation. **None of the existing
fp32/BS=32 fused work was modified** (`fused_backend.py`, `_bssn_rhs_fused.py`, scheme
`"fused"`, `step_3.2_phase2_h200.md` are all unchanged):

- **`staging.output_fanout()` + `staging.select_trunk_schedule(min_outfanout=12)`** —
  the output-fanout trunk selector (128 temps / 123 peak-live / 6.96× recompute), the
  correct fp64 materialize set (§3), distinct from `rank_candidates`' fan-out×cone.
- **`fused_fp64_backend.py` → `_bssn_rhs_fused_fp64.py`**, scheme **`"fused_fp64"`**
  (additive branch in `rhs.py`, same call path as `"fused"`). fp64-LOCKED, output-fanout
  trunk schedule, fused on-chip derivatives, FD order 8.
- **Increment 1 = whole-grid, CPU-validated:** bit-matches verbatim to **2.45e-13**
  (`test_fused_fp64_matches_verbatim_order8`; 3/3 fused tests, 13/13 schedule+fused green).
  On GPU it WILL spill (128 fp64 trunk temps as kernel locals) until Increment 2.
- **Increment 2 (TODO, H200):** BS=8 halo-tiling + trunk→SMEM scratch — the 266-fp64
  per-point working set must be SMEM, not registers. Intricate Triton-0.9.2 work
  (halo crop, `scratch_shapes`); do with the H200 in the loop, not blind.
- **Contraction (TODO, "if time"):** pre-contract metric 2nd-derivs (emit ~6 L_ij vs 36
  raw grad2_gt) → on-chip working set 138→~108, relieving Increment 2's SMEM. Changes
  the algebra input set → its own focused pass.

> **For the phase tracker:** the `3.2d` row in `docs/phases/README.md` notes the fp32
> CPU prototype; the fp64 sibling above is a parallel artifact the user may want to log
> there. Left for the user to coordinate with the parallel "algebra" chat.

## 7. Open / unmeasured — the H200 gate (tomorrow)

- `spill_probe.py` on the *fused trunk-SMEM + recompute* kernel: does it hold
  on-chip (registers ≤255, spill bytes ≈ 0) or still spill to HBM?
- SM% / MEM% profile (`profile_regime.py --smi`): did the regime actually flip
  (MEM%↓ / SM%↑)?
- **SMEM capacity** at the chosen tile size (trunk + 138 derivs co-resident).
- **SMEM-bandwidth risk:** 7× recompute hammers SMEM reads of the trunk/derivs —
  could become SMEM-BW-bound rather than FP-bound. Must be measured, not assumed.
- Roofline FLOP counts are op-proxy estimates; confirm against a real FLOP count.
