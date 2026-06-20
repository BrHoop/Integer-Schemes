# Plan: Memory-lean AMR — recompute-everything + interiors-only storage

> Repo-tracked copy (survives the session). Status per phase below.
> **Phase 3 (2.5D streaming kernel) is owned by a separate parallel agent — out of
> scope for this thread.** This thread does Phases 1, 2, 4 + the order-8 prereq.

## Context

The Llama multipatch prototype (M0–M4, validated) has a Phase-A block-structured
**Berger–Oliger** AMR (`3D/src/multipatch/amr/`, 25 tests) — fixed-size `BS³` blocks,
ragged per-level `AMRState`, proper-nesting, 2:1 whole-block refinement (confirmed *not*
octree). Block-structured was chosen for the shell-grid / GPU-coalesced / shape-stable
reasons, not memory — so memory must be engineered in.

**Governing constraint: memory footprint is THE roadblock to a fast (HBM-bound) code.**
Principle: **never store what you can recompute** — compute is far cheaper than memory
traffic. Two large avoidable sinks: (1) stored per-slot geometry ≈ 3.6× field storage;
(2) stored per-block halos = `((BS+2NG)/BS)³` = **8× at BS=8, NG=4** (target NG=4 / order 8).

Goal of the immediate effort: **get AMR on the cubesphere working memory-leanly so its
effect on the codebase can be tested** — not a perf showcase.

## Phase 1 — Recompute geometry (eliminate stored `jinv`/`d2coef`)  ✅ DONE (2026-06-15)

Affine cube: `jinv = I/world_scale` and `d2coef = 0` are constant across nodes/blocks/
levels (level only changes FD spacing `dxi`). `geometry.level_geometry` returns those two
constants; `evolve.build_geometry`/`rhs` carry **36 numbers total** instead of per-slot
`(caps[L],3,3,W,W,W)` arrays. `CurvilinearDerivative` unchanged (constants broadcast).
25 AMR tests green (build 78s→23s — no per-slot autodiff); `test_recompute_geometry.py`
locks parity (`<1e-13`) + footprint (36, >1000× reduction). Curvilinear-shell per-node
recompute is the Phase B hook (`level_geometry` raises for non-affine). Cut both persistent
AND peak (geometry was materialized).

## Phase 2 — Interiors-only storage (eliminate stored halos)  ✅ DONE (2026-06-15)

> `AMRState.blocks[L]` now `(caps[L],NF,BS,BS,BS)` (interiors only). `evolve.build_haloed`
> rebuilds the transient haloed working buffers from interiors each substage (L0 root
> stitch → cross-level prolong → same-level face copy → outer BC); `_block_rhs` crops to
> the BS interior; RK4 runs on interiors. `make_root_state`/`sync_within_level_root` adapted.
> 30 AMR tests green (oracle tracking, exact stitch, no-recompile all hold);
> `test_interiors_only.py` locks the shape + persistent-footprint cut = `(W/BS)³`
> (**8× at NG=4/BS=8**). Base-level faces+edges+corners now exact (stitch); fine-fine exact
> edges deferred (per scope). Persistent win only — peak still spikes on the transient
> buffer (that's Phase 3, parallel agent).
>
> _(original Phase-2 spec below)_

- `AMRState.blocks[L]`: **`(caps[L], NF, BS, BS, BS)`** — interiors only; drop the persistent
  `2·NG` halo (`state.py`, `make_root_state`).
- Halo becomes a **transient** working buffer rebuilt inside each RHS substage from stored
  interiors: L0 root stitch (`sync_within_level_root`, now interiors→haloed); fine levels
  embed interior + cross-level prolongation + same-level neighbour copy; outer BC last.
  RHS evaluates on the haloed buffer and **crops to the BS interior**; RK4 runs on interiors.
- **Scope (agreed):** base level (L0) gets faces+edges+corners *exact* from the stitch →
  fixes the base-level mixed-2nd-derivative correctness gap. Fine-level edge/corner ghosts
  stay cross-level (coarse) until general fine-fine adjacency (regridding) exists — exact
  fine-fine edge sync deferred.
- **Honest caveat:** reduces *persistent* footprint (8×); *peak* during a step still spikes
  on the transient haloed buffers — peak only drops with the 2.5D streaming kernel (Phase 3,
  parallel agent). Phase 2 is the layout prerequisite + base-level correctness fix.
- Files: `state.py`, `sync.py`, `evolve.py`; update `test_sync.py`/`test_cube_amr.py`/
  `test_two_level.py` for the interiors-only shapes.

## Phase 3 — 2.5D streaming derivative  ⏸ PARALLEL AGENT (out of scope here)

The rolling `2·NG+1`-plane z-window kernel that retires the transient halo / kills peak
memory (extends `bssn3d/tiled_deriv.py`, Pallas/GPU). Handled by a separate agent.

## Phase 4 — BS sweep + footprint report  ⬜

With geometry and stored halos gone, sweep `BS` at NG=4 to find the halo-amortization vs
over-refinement optimum; correct the `BS=8` "small (memory)" comment with measured numbers.

## Prereq — re-validate single-level multipatch at order 8 / NG=4  ⬜

The prototype was validated at order 6; confirm the single-level path holds at order 8.

## Phase W — Wavelet analyzer (initial-data grid construction + refinement indicator)  ⬜

Add an **interpolating (Deslauriers–Dubuc) wavelet** analyzer on the node-centered
block hierarchy — the same wavelet family Dendro-GR uses, so it's consistent with the
reference. The key reuse: on a node-centered grid the DD detail coefficient at an
inserted (odd) node is exactly the **prolongation residual** `u − P(R u)`, computed with
the existing `_W_MID` midpoint stencil + injection restriction (`kernels.py`). So the
transform is assembled mostly from parts already built; detail magnitude is the local
truncation-error proxy.

- **Primary use — initial-data grid construction.** Given the (analytic or elliptic-solve)
  initial data, compute level-by-level wavelet details and mark for refinement any block
  whose max |detail| exceeds a tolerance → the objective t=0 AMR hierarchy. For puncture
  data this refines around the punctures automatically; evolution then hands off to the
  geometric **moving-box** refinement (tracked on χ — the eventual evolution strategy).
- **Secondary use — pluggable refinement indicator + resolution monitor.** Drops into the
  same regridder indicator interface as the gradient/Löhner options (Phase A4), and serves
  as a validation tool ("did the boxes resolve everything?").
- **Files:** new `amr/wavelet.py` (DD forward transform / detail coefficients via the
  existing prolong/restrict kernels; per-block detail-max indicator), + an initial-data
  grid-builder hook; tests vs analytic fields (details → 0 at the interpolation order on
  smooth data; flag a sharp feature).
- **Scope/honesty:** start with the interior identity `detail = u − P(R u)` (exact, trivial)
  for initial-data construction where the full field is available; multi-level transforms
  across coarse–fine interfaces / patch seams (ghost handling) are a later refinement.
- **Sequencing:** independent of the memory-lean refactor; lands alongside / before the
  regridding (A4) + moving-box work, since it's the grid-bootstrapping primitive they need.

## Verification

- `cd 3D && python -m pytest tests/multipatch_amr/ -q` — all green through Phases 1–2, plus
  recompute-geometry parity + footprint asserts, and a persistent-footprint report.
- `python -m multipatch.validate` for the order-8 single-level re-validation.

## Boundaries / sequencing

- In-place refactor; keep tests green at each phase.
- Still pending after this thread: convergence/parity-through-regrid test, regridding (A4),
  shells + inter-patch coupling (B), MCS-on-AMR (C), sub-cycling.
