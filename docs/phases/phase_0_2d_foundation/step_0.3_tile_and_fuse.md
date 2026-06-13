# Step 1.3 — Shared-memory tiling + temporal fusion (the gate)

> ⚰️ **RETIRED (2026-06-11) — do not implement.** Superseded by the strategy pivot
> (see `../README.md` ⚑ banner). 2D shared-memory tiling + temporal fusion is a
> **toy-model artifact**: it doesn't transfer to 3D (24 BSSN fields → tiles too big,
> 64× redundant) and isn't needed once the RHS is compute-bound. The *derivative-as-
> GEMM on tensor cores* idea survives only inside the **Ozaki `compute_deriv`
> operator (Phase 4)**, not as a 2D-solver throughput play. The efficiency story
> moved to 3D BSSN (Phases 1–4). This doc is kept as a tombstone for the reasoning.

**Phase 1 · Step 1.3 · Status: ⚰️ RETIRED (was ACTIVE)**

## Purpose
Step 1.2 found the FP64 baseline is **DRAM-bandwidth-bound for one reason**: XLA
implements the 6th-order step as **~15 global-memory `dynamic_slice` passes with
zero shared memory** (0 B), so it streams the state through HBM ~15× per step
with no on-chip reuse. This step writes a **Pallas kernel that loads each block
into shared memory once, computes the RK4 step on-chip, and writes once** —
collapsing those ~15 global passes to ~2 (one read, one write).

That alone should flip the regime: a perfectly-tiled single step moves ~2×state
of DRAM and does the same compute, so its arithmetic intensity jumps from the
observed **~3.2** to **~28 FLOP/byte** — well past the **FP64 ridge (7.1)** →
**compute-bound**. This is *the gate*: tensor cores (1.4) cannot help a
memory-bound kernel; once compute-bound, swapping the FP64 compute for INT8
tensor-core GEMM (1.4 → Phase 2) is where the win lives.

A second knob — **temporal fusion depth `T`** (fuse `T` timesteps per HBM load) —
pushes intensity further, toward the **INT8 ridge (~410)** that the tensor-core
endgame needs. Single-step tiling crosses the FP64 gate; `T` is the dial toward
the INT8 target.

## Scope & decisions
- **FP64 direct stencil, not GEMM yet.** Keep the stencil as ordinary on-chip
  arithmetic in FP64. Casting it as a GEMM and putting it on tensor cores is
  **1.4**; the 2:4-sparse packing that avoids the ~4.6× dense-zero waste is
  **1.5**. 1.3 isolates the *memory* win (tiling + fusion → compute-bound) with a
  correctness-checkable FP64 kernel.
- **This is NOT resurrecting `pallas_fp`.** That retired kernel was a *naive*
  Pallas FP64 RHS — no shared-memory tiling, no temporal fusion, pow2-pad waste,
  one-field-at-a-time loads → 15× slower than XLA. 1.3 is the opposite: the
  shared-memory-tiled, temporally-fused structure that is the actual thesis
  kernel (FP64 first, for correctness + the regime flip).
- **New file** (e.g. `mcs2d/schemes/pallas_smem_fp.py`), reusing the Pallas
  scaffolding from `pallas_ozaki.py` (BlockSpec/patch tiling, `BS`/`NG`, pow2
  padding, the one-hot crop trick) and the banded-stencil structure.
- **Pallas constraints still apply** (`OZAKI_GPU_NOTES.md`): pow2 tile padding,
  no `dynamic_slice`/`gather`/`scatter`, crop via one-hot select-reduce. The
  *no-fp64-dot* rule does **not** bite here — 1.3 is direct stencil arithmetic,
  no MMA (that starts in 1.4).
- **Correctness bar unchanged:** the tiled kernel must pass the Step 1.1
  validation suite (convergence, constraints, oracle) — it is the same physics,
  a different memory schedule.
- **Profiling = `profile_regime.py --smi`** (Nsight banned): the regime flip is
  **MEM%↓ / SM%↑**; no counters needed.

## Background — the arithmetic from 1.2
| | DRAM traffic / step | arithmetic intensity | regime |
|---|---|---|---|
| XLA `fused_floating_point` (0 B smem, ~15 global passes) | ~15×state | **~3.2 FLOP/byte** | DRAM-bound |
| 1.3 single-step tiled (1 read + 1 write) | ~2×state | **~28 FLOP/byte** | **compute-bound** (> FP64 ridge 7.1) |
| 1.3 + temporal fusion depth `T` | ~2×state / `T` (amortized) | rises with `T` → toward **~410** | toward INT8 ridge |

So **shared-memory tiling is the primary lever** (it flips FP64 to compute-bound
on its own); **temporal-fusion depth is the dial** for the much higher intensity
the INT8 tensor cores will need.

## Tasks

### T1 — Shared-memory-tiled FP64 RK4 step (Pallas)
A Pallas kernel that, per block: loads the `(BS+2·NG_T)²` tile (NG_T = 4·NG = 12,
the 4-stage RK4 halo) into on-chip memory **once**, computes all 4 RK4 stages of
the 6th-order MCS RHS on-chip (with KO + constraint damping), and writes the
`BS²` interior **once**. Periodic ghost handled by the existing block-halo sync.
- Correctness: bit-for-bit-ish vs `fused_floating_point` on the birefringent
  oracle (interior), and pass `tests/validation`.
- This is the same temporal halo the XLA path already uses (NG_T=12); the new
  thing is that the tile is **resident in shared memory** across the 4 stages.

### T2 — Confirm the regime flip (the gate)
On the H200, via `profile_regime.py --smi` at 512²:
- **MEM% drops** (DRAM traffic cut from ~15×state to ~2×state) and **SM% stays
  high** → compute-bound. This is the gate criterion.
- Throughput vs the XLA baseline (225 Mpts/s). FP64 tiled may or may not *beat*
  XLA on wall-time (FP64 compute is slow), but the **intensity must rise above
  the FP64 ridge** — that is what makes 1.4's tensor cores worthwhile.
- Cross-check with the benchmark roofline (the kernel's point should move right,
  past 7.1).

### T3 — Temporal fusion depth `T` (the dial toward the INT8 ridge)
Generalize T1 to fuse `T` timesteps per kernel (overlapped/redundant-halo
tiling: halo grows to `T·NG_T`, redundant on-chip compute grows with `T`).
- Sweep `T = 1, 2, 4, …`; measure **arithmetic intensity and MEM% vs `T`**, and
  the redundant-compute cost. Plot the IA-vs-`T` curve.
- Find how far toward the **INT8 ridge (~410)** is reachable before redundant
  halo / shared-memory capacity (H200 ≈ 228 KB/SM; tile = `(BS+2T·NG)²·NF·8 B`)
  dominates. This sets the fusion depth 1.4/Phase 2 will target.
- *Secondary within 1.3* — the FP64 gate (T1–T2) is the must-have; `T>1` is the
  optimization that may carry into 1.4. If overlapped tiling gets unwieldy, note
  the smarter schedules (diamond/trapezoidal tiling) as a 1.4+ refinement.

### T4 — Validation + no-recompile
- Run `tests/validation` against the tiled scheme (convergence order, constraint
  preservation, oracle) — same physics, new schedule.
- If wired into AMR later, keep it **shape-stable** (no recompile on regrid);
  not required for the single-grid gate.

## Risks / gotchas
- **Pallas pow2 pad waste** (Triton): the tile pads to pow2 (e.g. 22→32), wasting
  compute on padding. Quantify; it is a constant tax, not a blocker (1.6 revisits
  alignment). The `pad_waste` is already reported in the benchmark JSON.
- **Shared-memory capacity** caps `T`: a `(BS+2T·NG)²·10·8 B` tile must fit in
  ~228 KB/SM. At BS=32, T=1 → 56²·80 B ≈ 251 KB (already tight!) → may need
  smaller BS or field-streaming. **This is the first thing to size before coding.**
- **FP64 in Pallas is slow** (CUDA-core FP64, ~1/2–1/4 of FP32). 1.3 is a
  *correctness + regime* stepping stone, **not** the speed win — that arrives
  when 1.4 moves the now-compute-bound work onto tensor cores. Don't judge 1.3 on
  raw FP64 wall-time.
- **Compile time:** unrolled multi-stage/`T` kernels can blow up compile (cf.
  the `pallas_ozaki` ~1600 s wall). Roll the stage/`T` loop with `lax.scan` where
  possible; lean on the persistent cache.

## Quantitative targets (ridge points, from 1.2)
- **FP64 ridge = 7.1 FLOP/byte** — single-step tiling (IA ~28) clears it → the gate.
- **INT8 ridge = ~410 FLOP/byte** — needs temporal fusion; `T` sets how close.
- Headline success: the kernel's roofline point moves from ~3.2 (DRAM-bound) to
  the right of 7.1 (compute-bound), with `--smi` MEM%↓ confirming it.

## Deliverables
- `mcs2d/schemes/pallas_smem_fp.py` — shared-memory-tiled (T1), temporally-fused
  (T3) FP64 RK4 step.
- Scheme wired into `main.py` dispatch + the benchmark `SCHEMES`.
- Validation: `tests/validation` green for the tiled scheme.
- Regime evidence: `profile_regime --smi` MEM%↓/SM%↑, benchmark roofline point
  past the FP64 ridge, IA-vs-`T` curve.
- Recorded: the fusion depth `T` that the tensor-core path (1.4) should target.

## Reuse vs build
- **Reuse:** `pallas_ozaki.py` tiling/BlockSpec/crop scaffolding, `_build_D`
  banded-stencil structure, the NG_T=12 temporal-halo logic already in
  `fused_rhs_fp.py`, the validation suite, `profile_regime --smi`, the benchmark
  roofline.
- **Build:** the shared-memory-resident RK4 step kernel, the `T`-deep temporal
  fusion, the shared-memory capacity sizing, and the IA-vs-`T` measurement.

## Exit criteria (the gate)
- ✅ Tiled FP64 kernel correct (passes `tests/validation`; matches oracle interior).
- ✅ DRAM traffic cut (MEM%↓ vs XLA under `--smi`) → **arithmetic intensity above
  the FP64 ridge (7.1)** → compute-bound. *This is the gate for 1.4.*
- ✅ IA-vs-`T` curve measured; the depth needed to approach the INT8 ridge recorded.
- → Hand-off to **1.4**: swap the now-compute-bound FP64 stencil compute for an
  FP16/TF32 tensor-core GEMM (speed probe), then INT8/Ozaki in Phase 2.

## Status / changelog
- 2026-06-10 — Created. Active step after 1.2 (regime = DRAM-bound; XLA uses 0 B
  shared memory). Primary lever: shared-memory tiling; dial: temporal fusion `T`.
