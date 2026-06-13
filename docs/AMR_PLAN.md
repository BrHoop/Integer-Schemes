# AMR Implementation Plan — 2D MCS Prototype → BSSN-Ready

> **Scope note (2026-06-11).** This is the **detail doc for the AMR thread**, not a
> top-level phase plan. Its internal "Phase 1–7" numbering is *local to AMR* and is
> **independent of** the canonical project spine in `phases/README.md`. Status: the
> **2D block-structured AMR is complete** (Phases 1–5 here, bar the Phase-5 GPU
> profile). In the project spine, the **3D AMR port is Phase 5.5** — it folds in
> *after* the BSSN RHS is correct (Phase 2) and efficient (Phases 3–4), alongside the
> grid/frame work. Do not read this doc's phase numbers as the project's.

## Context & Goals

Build a working block-structured AMR implementation in JAX, using the existing 2D Maxwell-Chern-Simons solver as the test bed. The MCS code is the throwaway target — what we're actually building is the **reusable AMR infrastructure** that will carry over to BSSN/BBH.

**Success criteria:**
1. AMR-evolved birefringent wave agrees with single-level run to spatial-discretization precision
2. Zero mid-evolution recompilation (verified by test)
3. Architecture cleanly ports to 3D and to any field count (BSSN has 25 vs MCS's 10)
4. Total development effort: 4-6 weeks of focused work

**Non-goals (defer to later):**
- 3D before 2D works
- BSSN physics (separate project)
- Multi-GPU sharding
- Performance optimization beyond "doesn't recompile" *(through Phase 3 only — Phase 4+ now pursue profiling-driven optimization, since the algorithm is correct and the GPU profile exposed the real bottlenecks)*

---

## Architecture (settled)

### Data structures

```python
# JAX-side: fixed shapes always
@register_pytree_node_class
class AMRState:
    blocks: jnp.ndarray  # (LEVELS, MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)
    active: jnp.ndarray  # (LEVELS, MAX_BLOCKS) bool

# Python-side: bookkeeping (dicts, lists, numpy)
@dataclass
class AMRTopology:
    active: np.ndarray             # mirror of JAX active mask for fast host queries
    children: dict                 # (level, idx) → list[(level+1, child_idx)]
    parent: dict                   # (level, idx) → (level-1, parent_idx) or None
    bbox_ijk: dict                 # (level, idx) → (i0, j0) integer corner in level-L cells
    neighbors: dict                # (level, idx, face_id) → (level, neighbor_idx) or None
    streaks: np.ndarray            # (LEVELS, MAX_BLOCKS) hysteresis counters
```

### Core principles (do not break these)

1. **No JAX function takes shapes that depend on the current refinement structure.** Every shape is a startup constant.
2. **All topology decisions in Python.** JAX never sees `children`, `parent`, or `neighbors`.
3. **Per-block kernels are level-agnostic.** Same compiled binary processes a level-0 block and a level-5 block. Only `dx`, `dt`, and block contents differ.
4. **Inactive slots multiply by zero, never get skipped.** Wasted compute is the price for stable JIT.

### Configuration (set at startup)

```python
# 2D prototype defaults — matches existing fused_floating_point and fused_ozaki
LEVELS     = 4         # max refinement depth (8 for BBH eventually)
MAX_BLOCKS = 64        # per level; sized for the prototype, grows for BBH
BS         = 32        # block size — matches existing kernels in 2D
NG         = 3         # 6th-order halo
NF         = 10        # MCS field count
```

Memory: `4 × 64 × 10 × 38 × 38 × 8` ≈ **30 MB** for blocks tensor. Tiny — production BBH will be ~10 GB.

**3D note:** the existing `fused_ozaki` SMEM budget at BS=32 in 3D would overflow Hopper's
228 KB shared memory. When Phase 6 ports to 3D, the ozaki path drops to BS=8 (residues fit)
while the fp64 path can stay at BS=16. The AMR infrastructure itself is BS-agnostic — only
the per-block kernel cares.

---

## Phase 1 — Static Foundation (Week 1)

**Goal:** Get the data structures, per-block kernels, and ghost-zone sync working with a HAND-CONFIGURED refinement structure. No regridding logic. Pure infrastructure.

### Deliverables

**New files:**
- `mcs2d/amr.py` — `AMRState`, `AMRTopology`, helpers, regrid driver (stub)
- `mcs2d/amr_kernels.py` — jitted per-block kernels (prolongate, restrict, sync, advance)
- `mcs2d/tests/test_amr.py` — Phase-1 correctness tests

### Specific work items

| # | Item | Effort |
|---|---|---|
| 1.1 | Define `AMRState` (pytree-registered) and `AMRTopology` (dataclass). | 0.5d |
| 1.2 | Build `prolongate_2to8(parent_block, child_idx)` — 6th-order Lagrange interpolation parent→one of 4 children in 2D. Jitted. | 1d |
| 1.3 | Build `restrict_8to1(children, parent_slot)` — average 4 children into one parent in 2D (8 → 4 in 2D, 8 in 3D). Jitted. | 0.5d |
| 1.4 | Build `sync_ghosts_within_level(blocks, active, neighbor_map)` — for each block, copy ghost cells from same-level neighbors. Jitted. | 1d |
| 1.5 | Build `sync_ghosts_across_levels(blocks, active, parent_links)` — fill fine-block ghost cells from interpolated coarse parent. Jitted. | 1.5d |
| 1.6 | Build `advance_level(blocks, active, dx, dt)` — vmap the existing scheme's per-block RHS over all blocks at one level, masked by active. Jitted. | 1d |
| 1.7 | Hand-configure a 2-level test: root grid + ONE refined block at known location. Construct `AMRState` and `AMRTopology` manually. | 0.5d |
| 1.8 | Validation test: run 100 steps with `fused_floating_point`, compare to single-level result in the refined region. Must agree to 6th-order precision. | 1d |

### Phase 1 exit criteria

- [ ] Birefringent wave evolves stably for 100+ steps in the 2-level layout
- [ ] L2 error vs single-level reference < 1e-7 in the refined region
- [ ] Prolongation + restriction kernels compile once, never recompile
- [ ] All existing 59 tests still pass

**Phase 1 estimated effort: 5-7 days**

---

## Phase 2 — Regridding (Week 2)

**Goal:** Add the regridding driver. Detect where refinement is needed, allocate new fine blocks, free unneeded ones, update topology.

### Deliverables

**Modified files:**
- `mcs2d/amr.py` — add `regrid()`, `compute_indicators()`, hysteresis logic
- `mcs2d/tests/test_amr.py` — Phase-2 tests

### Specific work items

| # | Item | Effort | Status |
|---|---|---|---|
| 2.1 | Build `compute_indicators(blocks)` — jitted, returns indicator per block. gradient-of-Ez criterion. | 0.5d | ✅ `compute_indicator_gradient` in `amr/kernels.py` |
| 2.2 | Build `enforce_nesting_buffer(flags_np, n_buffer=2)` — pure NumPy. Propagate refinement flags upward and dilate. | 1d | ✅ `enforce_nesting_buffer` in `amr/regrid.py` (buffer dilation + no-orphan coarsening) |
| 2.3 | Implement hysteresis (per-block streak counter, only refine after K=3 consecutive flags above threshold). | 0.5d | ✅ `compute_flags` in `amr/regrid.py` |
| 2.4 | Implement `find_empty_slot(topo, level)`. | 0.5d | ✅ `AMRTopology.find_empty_slot` |
| 2.5 | Implement `create_children(...)` — allocate slots, prolongate, update topology. | 1d | ✅ refine path in `apply_flags` |
| 2.6 | Implement `coarsen(...)` — restrict, deactivate, update topology. | 0.5d | ✅ coarsen path in `apply_flags` |
| 2.7 | Build top-level `regrid(state, topo)` orchestrator. | 1d | ✅ `regrid` in `amr/regrid.py` |
| 2.8 | Wire regrid into the timestepping loop (called every K steps). | 0.5d | ✅ `evolve_with_regrid` in `amr/regrid.py` |
| 2.9 | Test: pulse evolves → refined region adapts. | 1d | ✅ `test_amr_tracking.py` (appears-near-pulse + adapts-over-evolution) |
| 2.10 | Test: zero recompilation across regrid events. | 0.5d | ✅ `test_no_recompile.py::test_n_level_step_survives_regrid` |

**Beyond the original plan (added for the BBH/GR goal):**
- ✅ **N-level support** — `make_n_level_step` evolves an arbitrary-depth hierarchy
  (shared-dt; sub-cycling deferred to Phase 3). Max depth set by `LEVELS`
  (env `MCS_AMR_LEVELS`). Validated with a depth-3 recursive-refine test.
- ✅ **`max_level` depth cap** in the regrid pipeline — bounds the slot budget
  (the resolution-independent |∇Ez| indicator otherwise cascades to the deepest
  level wherever a feature is; a self-limiting truncation-error estimator is a
  future refinement).
- ✅ **`restrict_all_into_parents`** — vmapped multi-slot restriction; opt-in via
  `restrict_at_end` (off by default — naive restriction without flux correction
  degrades constraint conservation at coarse-fine boundaries; flux correction
  is a Phase 3 item).
- ✅ **`AMRTopologyArrays`** — JAX snapshot of host topology (`to_jax_arrays()`),
  passed as runtime args so a single compiled step survives regridding.
- ✅ **Long-term stability tests** — pure-Maxwell N=128 guard, birefringent
  stable-regime accuracy, CFJ-is-physical, depth-3 N-level.

### Phase 2 exit criteria — ALL MET

- [x] Pulse evolution: refined region adapts to the feature — `test_amr_tracking.py`
- [x] Verified zero per-regrid recompilation (JAX cache-hit count) — including
      multi-level refine + coarsen events, and across the full `evolve_with_regrid` loop
- [x] Budget overflow detected gracefully (raises clear error) — `find_empty_slot`
      and `apply_flags` raise `RuntimeError`; `max_level` caps depth to avoid it
- [x] L2 error vs reference doesn't degrade — root-only AMR is bit-identical to
      `fused_floating_point`; birefringent stable-regime tracks analytic to L2<1e-5
      over 3 crossings (was blocked by the KO bug, now RESOLVED — see below)

> **✅ RESOLVED during Phase 2 — the "grid-scale instability" was a KO sign bug.**
> The Kreiss-Oliger stencil was stored NEGATED, making dissipation
> anti-dissipative (it amplified the k≈π/dx mode; stronger σ → faster blow-up).
> Fixed by negating `_CKO`/`CKO`/`cko` to the standard δ⁶ across all 5 schemes.
> A SECOND effect — the birefringent IC blowing up at the old default `Lambda=2`
> — turned out to be the PHYSICAL Carroll-Field-Jackiw tachyon (ω²=k²−m_cs·k < 0
> when k < m_cs), not a bug. The default `Lambda` was lowered to 0.4 (CFJ-stable)
> with a documented threshold in `params.toml`. Full diagnosis in
> `tests/regression/test_long_term_stability.py` and the project memory.
> 8th-order KO (NG=4) is a documented accuracy refinement, deferred — see
> `floating_point.CKO`.

**Phase 2 status: COMPLETE.** Remaining for production-grade AMR (now Phase 3):
conservative cross-level flux correction, within-level sync for non-root levels
with multiple blocks, and Berger-Oliger sub-cycling.

---

## Phase 3 — Sub-cycling in Time (Week 3, the hardest)

**Goal:** Implement Berger-Oliger sub-cycling. Finer levels take smaller, more frequent timesteps. This is the algorithmic core of "proper" AMR.

### Deliverables

**Modified files:**
- `mcs2d/amr.py` — `step_amr()` recursive driver
- `mcs2d/main.py` — top-level integration loop calls `step_amr` instead of `step_rk4`
- `mcs2d/tests/test_amr.py` — Phase-3 tests

### Specific work items

| # | Item | Effort | Status |
|---|---|---|---|
| 3.1 | Recursive sub-cycled `step` — level L runs 2 substeps of dt_L/2, recursing into finer levels. | 2d | ✅ `make_subcycled_n_level_step` (+ 2-level `make_subcycled_two_level_step`), unrolled at trace time → one compile |
| 3.2 | Boundary-interpolation-in-time. | 1.5d | ✅ cubic-Hermite (4th-order) via `_hermite_basis`; (value, derivative) pairs threaded down the recursion so every level's halo is 4th-order in time |
| 3.3 | Restriction after each sub-cycle. | 0.5d | ✅ `restrict_all_into_parents` at the end of each level's sub-cycle |
| 3.4 | Test: sub-cycled vs reference. | 1d | ✅ N-level == 2-level to machine precision; 2-level matches shared-fine-dt to the time-interp error; stable + bounded |
| 3.5 | Convergence test. | 1d | ✅ root RK4 verified clean 4th-order; sub-cycled convergent + bounded (see note) |

### Phase 3 exit criteria

- [x] Sub-cycled run matches reference to its expected precision — N-level ==
      2-level to FP round-off (6e-16); 2-level matches shared-fine-dt to the
      Hermite time-interp error
- [~] Convergence-rate test — root RK4 is clean 4th-order; the sub-cycled
      *fine block* is **spatial-boundary-limited** (2nd-order restriction +
      coarse-quality halo, ~1e-6) which masks the temporal order. The aspirational
      "1e-9 / slope −6" is boundary-limited, NOT integrator-limited (see note).
- [x] No regression in the zero-recompilation tests — sub-cycled 2-level and
      N-level both trace exactly once

> **Note — the AMR accuracy floor is spatial, not temporal.** RK4 + cubic-Hermite
> time interpolation is genuinely 4th-order (proven on root-only self-convergence:
> order 4.00). The sub-cycled fine-block error is dominated by the coarse-fine
> *spatial* boundary — 2nd-order restriction and prolongation of coarse-resolution
> halo data. The next accuracy lever toward the 1e-9 target is **conservative
> higher-order cross-level coupling (flux correction)**, not a better integrator.

**Phase 3 status: COMPLETE** (sub-cycling + Hermite + convergence characterised).
Remaining production-AMR work, now folded forward: conservative flux correction
at coarse-fine boundaries; within-level ghost sync for non-root levels with
multiple blocks per level.

> **✅ Compile-time scaling fixed (rolled sub-cycling) — critical for BBH depth.**
> The original sub-cycled step *unrolled* the Berger-Oliger recursion (two inlined
> `advance(L+1)` calls per level), so the traced graph grew like **2^LEVELS**.
> Measured compile time (CPU, FP, caps=1): **4 → 10 → 25 → 53 → 115 s for LEVELS
> 2→6 (~2.1×/level)** — extrapolating to BBH's LEVELS≈15 that is **~24 hours**, and
> Ozaki (heavier nodes + per-level Triton compiles) makes it worse. **Fix:** roll
> the two substeps into a `lax.scan` so each level's body is traced **once** → graph
> is O(LEVELS) nested `while`s, compile **~linear in depth** (measured **9 → 16 → 23
> → 29 → 34 s**; L=15 ≈ **90 s** vs ≈ 24 h). The rolled step is **BIT-IDENTICAL**
> to the unrolled (`array_equal` over 5 steps), still traces once across regrids,
> and is now the **default** `make_subcycled_n_level_step` (the unrolled kept as
> `make_subcycled_n_level_step_unrolled` for A/B). Ragged storage + static-`L`
> kernels preserved. *Remaining compile lever for Ozaki:* make the per-block kernel
> level-agnostic (pass `dx`/`dt` as data) → one Triton compile instead of LEVELS
> (do before Phase 7).

---

## Phase 4 — GPU Profiling & Ghost-Zone Optimization (2D)

**Goal:** Now that the AMR algorithm is correct (Phases 1–3), make it *fast* on
real hardware before locking the data layout into 3D. This phase is **profiling-
driven** — every step is measured on the supercomputer GPU, and every code change
is **bit-identical / machine-precision** (regression-gated), so we never trade
accuracy for speed. (This supersedes the original "no perf optimization" non-goal,
which applied only through Phase 3.)

### What the GPU profile told us (baseline, `traces/amr/`)

On a depth-4 hierarchy at uniform caps=64 (~9% slot occupancy — BBH-representative
sparsity), one sub-cycled step spent **only ~30% of GPU time on arithmetic** and
**~58% on AMR plumbing**: `concatenate` (prolongation glue) 28%, `select`
(active/halo masking) 15%, block slice/gather/reduce the rest. A single cross-level
ghost sync cost ~2.8× a single RHS. The two levers: (A) stop refining whole blocks
to fill thin halos, and (1) stop computing/masking dead slots. See `OPTIMIZATION.md`
for the full op-family tables.

### Specific work items

| # | Item | Target bucket | Status |
|---|---|---|---|
| 4.0 | Per-phase profiler methodology — each phase in its own `jax.profiler.trace` capture + `profile_results.json` for before/after diffs | — | ✅ `profile_amr.py` |
| 4.1 | Profiler cap control (`--caps`, `--autocaps`, `--cap-margin`) — profile at realistic occupancy, not worst-case | — | ✅ `profile_amr.py` |
| 4.2 | **A1: footprint-only prolongation** — slice parent to the child's coarse footprint before refining (76²→48²) | `concatenate` | ✅ `kernels.py` `_prolong_window`; machine-precision test `TestProlongateFootprintIdentical` |
| 4.3 | Fix profiler `rmtree` footgun — clear only per-phase subdirs, never delete `*.json` | — | ✅ `profile_amr.py` `_trace_phase` |
| 4.4 | **Calibrated caps as production default** — `make_calibrated_root_state` reads the sidecar to pre-size; precedence explicit > sidecar > uniform | `select` + dead `add` | ✅ `regrid.py`; tests in `test_amr_calibration.py` — *biggest measured win (~10.5× slot cut)* |
| 4.5 | **A2: annulus-only prolongation.** Two halves: (a) **fuse the two `where`s** in `sync_ghosts_across_levels` into one (`active & halo_mask`); (b) refine only the halo ring via strips | `select` (a) + `concatenate` (b) | ✅ (a) `kernels.py` + `TestSyncAcrossLevelsFusedSelect`. ⛔ (b) **deprioritized** — see verdict |
| 4.6 | **Compaction to dense prefix** — swap-on-remove in regrid so live blocks fill `[0:n_active]` | `select` + memory locality | ⏸ **deferred to the BBH regime (Phase 5/6)** — see verdict |

### Measured result (full re-profile, `traces/amr/v2_*`)

Combined ≈**13× less GPU work per step** (10.7 → 0.78 ms/step), **dominated by
calibrated caps** (uniform→autocaps alone is ~10.5×). Clean attributions at
matched (256-slot uniform) configs:
- **A1 (4.2):** `concatenate` 28.5% → 16.8% of step (≈2.4× less per step). Working as designed.
- **Fused select (4.5a):** `select` ~1.2× less per step; **invisible at autocaps** (that regime is latency-bound, ~43 µs floor). Bit-identical, kept.
- **Dead-slot proof:** `nbx4-uniform` (16 active root blocks, 8.16 ms) ≈ `nbx8-uniform`
  (**64** active, 8.06 ms). 4× the real work, same cost — compute is **caps-bound,
  not active-bound**. This is *the* lever, and calibrated caps (4.4) captures it.

At throughput scale the op-mix is now **flat** — `add` (physics) back on top at
24.7%, then a balanced cluster of intrinsic AMR plumbing (`select` 17, `concatenate`
17, slices 21, `reduce` 9, `gather` 7). No fat target remains.

### Verdict — strips deprioritized, compaction deferred to BBH (do NOT discard)

- **4.5b strips: deprioritized.** After A1, `concatenate` is only 16.8% and the
  strip refinement saves just the *interior-discard* fraction (~1.8× of that), at
  4× the kernel launches. **Correction to an earlier claim:** strips do **not**
  matter more in 3D — that conflated *halo-storage* overhead `(BS+2NG)ᴰ/BSᴰ` (which
  is worse in 3D) with the *interior-discard* saving `BSᴰ/(BS+2NG)ᴰ` (which is
  *smaller* in 3D at small BS, ~38% interior at BS=16 vs ~71% at BS=32 in 2D). Net:
  low value even for BBH. Revisit only if a deep (6–8 level) 3D profile shows
  prolongation unexpectedly dominant.
- **4.6 compaction: ⛔ RETIRED (updated after the dynamic profile).** Earlier this
  was deferred-to-BBH on the theory that dynamics/scale would expose fragmentation.
  The Phase-5 **dynamic moving-box profile measured it directly: fragmentation
  stayed 1.0 at every regrid** — the lowest-free-slot allocation (`find_empty_slot`)
  *self-compacts* (new blocks fill the holes left by destroyed ones), so there is
  nothing for explicit compaction to recover. Occupancy variance is handled by
  calibrated caps. **Retired** from the active plan; revisit only if a real 3D
  two-puncture merger shows pathological fragmentation.

> **🧭 Post-Phase-4: memory-bound, and the levers that follow.** The cost-model
> re-profile (`traces/amr/p5_*`) found **every phase is memory-bound** (FLOP/byte
> 0.0–0.55 ≪ ~10 roofline), splitting into **bandwidth-bound** (`full_step` 4.9 GB,
> `rhs`, `sync_cross`) and **latency-bound** (`restrict` 13000 tiny kernels,
> `sync_within`).  Consequences: KO is ~free; **Ozaki reframes to a bandwidth/
> footprint win (INT8 8× smaller), not FLOPs**; calibrated caps = 5.6×.  Memory
> levers (see OPTIMIZATION.md): **M0 vectorise restriction ✅ done** (per-slot
> `fori_loop`→1 batched scatter; HLO flat in slot count, 13000→~handful kernels,
> bit-identical); **M1a factor prolongation out of the stage loop ✅ done**
> (prolong brackets once, Hermite-combine per substep — linearity; ~8→4 prolongs/
> advance, machine-precision ~1e-15); **M1b fuse the RK4 stage into one Pallas
> kernel — FUTURE** (the biggest remaining bandwidth lever: collapse the
> ~500–2600 kernels/step sync→set→rhs→combine chain; needs a neighbour-strip-reading
> Pallas kernel); **M2 precision/Ozaki footprint**.

### Phase 4 status: COMPLETE (for the static 2D prototype)

A1 + calibrated-caps-default + fused-select delivered the win; `add` is again the
top bucket and the plumbing is balanced/intrinsic. The two remaining structural
levers (strips, compaction) have poor or unmeasurable returns *on this static toy*
— strips genuinely so, compaction only because the toy can't exercise the dynamics/
scale where it pays. Both are recorded above and carried forward, not lost.

---

## Phase 5 — Multi-block Levels & Moving Features (2D, BBH-shaped)

**Goal:** make the 2D AMR handle what BBH actually needs and the static prototype
never exercised — (1) **multiple adjacent blocks on a fine level** (a correctness
gap today) and (2) **refinement that tracks a moving feature** (the dynamics). This
finishes "2D maximally done before 3D" *and* creates the dynamic workload that the
deferred compaction (4.6) must be validated against.

**Why this is the right next phase (not more micro-opt):** the static toy has told
us essentially all it can — Phase 4 left the op-mix flat and the remaining levers
(strips, compaction) either dead or unmeasurable on a static single-feature grid.
The unmeasured bottlenecks live in *dynamics* (regrid churn, fragmentation,
within-level sync) and *scale* (3D, memory-bound). Phase 5 builds the first;
Phase 6 the second.

**Discipline (unchanged):** every step is validated against a reference
(multi-block == single-block to machine precision; moving feature tracks a uniform
high-res reference) and guarded by the no-recompile test.

### Specific work items — in dependency order

| # | Item | Why / validation | Status |
|---|---|---|---|
| 5.1 | **Multi-block within-level ghost sync (non-root).** `sync_ghosts_within_level` (face-only copy of a neighbour's edge interior into the shared-face halo), driven by `AMRTopology.rebuild_neighbors` / `neighbor_slot`+`neighbor_valid`; wired into `make_n_level_step` and `make_subcycled_n_level_step` after cross-level prolongation. | Verified the MCS RHS (`sx`/`sy`+KO) is fully separable → **corner halos never read** → face strips are provably complete. *Validated:* RHS-equivalence vs a single grid (machine precision, x & y) + a "teeth" test. | ✅ `kernels.py`, `state.py`, `evolve.py` |
| 5.2 | **Multi-block test harness + proper-nesting check.** `AMRTopology.check_proper_nesting()`; `test_amr_multiblock.py` (18 tests). | Exercises the path the profiler never hit. Covers kernel exactness (4 faces / boundary / corners / inactive), RHS-equivalence, neighbour topology (adjacency/reciprocity/to_jax fields), nesting pass+violations, multi-block no-recompile, and end-to-end step wiring (both step variants). | ✅ `test_amr_multiblock.py` (18 pass; full fast suite green) |
| 5.3 | **Moving-feature refinement (the dynamics).** ⚠️ **Re-scoped after reading the refine path.** The plan assumed patch-based "copy overlapping data, prolongate only the newly-exposed edge" — but our AMR is **block-structured (grid-aligned)**: a fine block occupies a fixed level-L grid cell, so `apply_flags` already **retains** existing blocks (`already_exists` check, [regrid.py]) and only prolongates *genuinely-new* grid positions (correct — no fine data existed there). A moving feature is handled by activating/deactivating grid-aligned blocks; there is no partial overlap to copy. 5.1's within-level sync then fixes the new-block↔existing-neighbour shared face at evolve time. So **the efficient transfer is already in place**; 5.3 becomes *validation* of the dynamic pipeline rather than new transfer code. | Tracking + no-recompile already covered by `test_amr_tracking.py`. **Added:** retention test (re-REFINE keeps existing children's data, no re-prolongation) + proper-nesting maintained after every regrid through a moving evolution. | ✅ (validation) `test_amr_tracking.py` `TestMovingFeatureDynamics` |
| 5.4 | **Buffer zones at the coarse-fine interface.** ✅ **Already built in Phase 2** — `enforce_nesting_buffer(..., n_buffer)` dilates REFINE flags to same-level neighbours within `n_buffer` blocks, so the feature stays inside the refined region (interface kept off the feature). Exposed through `regrid`/`evolve_with_regrid`. | Tested in `test_amr_regrid.py` (`test_buffer_dilates_to_neighbors`, `test_buffer_zero_is_noop`, …). Optional future refinement: a deeper `n_buffer` sweep if interface noise appears at BBH amplitudes. | ✅ `enforce_nesting_buffer` |
| 5.5 | **Profile the dynamic / multi-block regime.** Re-profile with adjacent fine blocks + active regrid churn (moving feature). **Needs a GPU run.** | This is the workload where compaction's value (fragmentation, occupancy variance, memory locality) can finally be **measured** → feeds the deferred 4.6 go/no-go. | ⬜ TODO (GPU) |
| 5.6 | Conservative cross-level flux correction | *Deferred / likely N/A* — BSSN is non-conservative FD (`OPTIMIZATION.md` §4). Build only if interface noise reappears at BBH amplitudes. | ⏸ deferred |

### Phase 5 exit criteria

- [ ] Multi-block fine level bit-matches a single larger block over the same region
      (within-level sync correct)
- [ ] A moving feature keeps a tracking refined region with **O(edge)** regrid
      transfer cost and **no per-regrid recompile**
- [ ] A dynamic / multi-block GPU profile exists — the evidence base for the 4.6
      compaction go/no-go (and for whether any new bottleneck appears under churn)

**Note on compaction (4.6):** if 5.5 shows significant fragmentation / dead-slot
churn under moving regrids, build compaction here (it's a regrid-side change,
co-located with 5.3). Otherwise carry it to Phase 6 (3D, memory-bound) where the
locality benefit is largest.

---

## Phase 6 — 3D Port

**Goal:** Port the entire AMR infrastructure from 2D to 3D, **inheriting the Phase 4
optimizations** (ragged + calibrated caps, A1 footprint prolongation, fused select)
and Phase 5 multi-block sync. Validate against the existing 3D MCS solver.

**This is where compaction (4.6) most likely lands.** 3D is memory-bound and large
(~25 fields, ~10 GB), so packing active blocks into a contiguous prefix — for
coalesced bandwidth and cache locality, not just slot-count — is expected to be a
first-order win here, unlike on the cache-resident 2D toy. Build/measure it against
the 3D workload. (Halo *storage* overhead is also worst in 3D — `(BS+2NG)³/BS³`,
~160% at BS=16 — but note this is the ghost-carry cost, **not** something the
deprioritized A2 strip-refinement addresses; see Phase 4 verdict.)

### Deliverables

**New files:**
- `mcs3d/amr.py` — 3D variants of `AMRState`, `AMRTopology`, regrid driver
- `mcs3d/amr_kernels.py` — 3D jitted kernels
- `mcs3d/tests/test_amr.py` — 3D AMR tests

### Specific work items

| # | Item | Effort |
|---|---|---|
| 4.1 | Generalize block shape to 3D `(NF, BS+2NG, BS+2NG, BS+2NG)`. Update `AMRState` dimensions. | 1d |
| 4.2 | 3D prolongation (parent → 8 children, each `BS³`). 6th-order trilinear Lagrange. | 1.5d |
| 4.3 | 3D restriction (8 children → 1 parent, volume-averaged). | 0.5d |
| 4.4 | 3D ghost-zone sync: same-level (6 face neighbors), across-level (8 corners, 12 edges + 6 faces). | 2d |
| 4.5 | Refactor `mcs3d/main.py` to use AMR infrastructure. | 0.5d |
| 4.6 | Run 3D birefringent wave with AMR vs single-level reference. Match to truncation-error precision. | 1d |
| 4.7 | First run on supercomputer GPU. Profile + benchmark vs single-level 3D. | 0.5d |

### Phase 6 exit criteria

- [ ] 3D AMR matches 3D single-level birefringent reference
- [ ] Benchmark on Hopper: AMR overhead < 20% of single-level cost at same effective resolution
- [ ] Working test suite for 3D AMR (mirror of 2D tests)

**Phase 6 estimated effort: 5-7 days. Mostly mechanical translation from 2D; ghost-zone bookkeeping is the slow part.**

---

## Phase 7 — Ozaki-on-AMR Throughput *(later)*

**Goal:** wire the INT8 Ozaki/Pallas tensor-core kernel into the AMR per-block RHS
(today AMR runs the FP64 `_make_kernel_fn` from `fused_rhs_pallas`).

**Priority note (from the GPU profile):** *deprioritized as a latency fix* — at
calibrated occupancy the RHS `add` bucket is only ~12% of step time, so faster
arithmetic barely moves the needle for small/sparse grids. It remains the right
**throughput** play for large *dense* levels in 3D/BSSN, where per-block RHS FLOPs
dominate. Revisit during/after Phase 6 once 3D block sizes and occupancy are known.
Not bit-identical (precision change) → gated by the convergence + constraint suite,
not an equality test. See `OPTIMIZATION.md` §5.

---

## Testing Strategy

### Invariants tested at every phase

1. **No mid-evolution recompilation.** Use `jax.jit_lib._GLOBAL_CACHE_HITS` (or equivalent) to assert compilation count is bounded across a run. Fixture in `test_amr.py`.

2. **Existing test suite passes.** All 59 current tests must continue to pass — AMR is purely additive.

3. **Conservation properties.** L1 norm of `Ez` over the domain should be preserved (modulo wave propagation off boundary for Sommerfeld BCs).

### Reference comparisons

| Phase | Reference | Tolerance |
|---|---|---|
| Phase 1 | Single-level run at coarse dx | L2 < 1e-7 in refined region |
| Phase 2 | High-res uniform run | L2 < 1e-6 globally |
| Phase 3 | Single-level run at fine dt | L2 < 1e-9 over 1000 steps |
| Phase 6 | 3D single-level | Same as Phase 1 in 3D |

### Performance benchmarks

Track for each phase:
- Per-step time (median over 100 steps)
- Compile time (first run vs cached)
- Memory (peak device memory)
- Recompile count over 1000-step run

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Triton kernel compile times explode | Medium | High | Use `fused_floating_point` for AMR development; switch to `pallas_ozaki` only for benchmarks. |
| 6th-order prolongation at level boundaries introduces spurious oscillations | Medium | High | Test prolongation kernels in isolation against analytical functions (e.g., x^6) before integration. |
| Sub-cycling time-interpolation is subtle (Phase 3) | High | High | Allocate 2-3 days of buffer in Phase 3. Reference: Berger & Colella 1989 paper has the canonical formulation. |
| Active-mask multiplication doesn't fully zero out inactive blocks (NaN propagation) | Low | High | Test: artificially fill inactive blocks with NaN, verify run completes without NaN escaping to active blocks. |
| Memory budget overflow on first BBH-scale run | Medium | Medium | Document the formula `LEVELS × MAX_BLOCKS × NF × (BS+2NG)^D × 8 bytes`; check before scaling up. |
| Multi-GPU not feasible with this architecture | Low | Medium | JAX sharding has demonstrated it works with similar structures (axis 1 of `blocks`). Defer multi-GPU until single-GPU works. |

---

## Decision Points

After each phase, **check in before proceeding**:

| After... | Question | If "No" → |
|---|---|---|
| Phase 1 | Do the static 2-level tests pass cleanly? Is the architecture clean? | Refactor, don't proceed |
| Phase 2 | Does regridding really run zero recompilations? | Audit shapes, fix before adding sub-cycling |
| Phase 3 | Does the sub-cycled convergence rate match theory? | Sub-cycling has a bug; debug before 3D port |
| Phase 4 | Re-profile: is `add` again the top bucket and every optimization machine-precision? | Find the new bottleneck / fix the accuracy regression before proceeding |
| Phase 5 | Do multi-block within-level sync and moving-box regrid work with no recompile? | Fix 2D before porting the bug to 3D |
| Phase 6 | Does the 3D AMR run on Hopper? | Profile, fix, then commit to BSSN |
| Phase 7 | Does Ozaki-on-AMR hold convergence + constraints? | Keep FP64 on AMR; Ozaki is throughput-only |

---

## What this enables next

Upon completion (end of Week 4-6):

- **3D MCS solver with AMR** — fully working
- **Reusable infrastructure** for any 3D evolution PDE with the same field-block-tile pattern
- **Direct ramp to BSSN**: replace MCS RHS with BSSN RHS in `advance_level`, leave AMR infrastructure untouched
- **Established performance baseline** for ozaki + AMR on real workloads

After Week 4-6, the remaining work for BSSN BBH is:

1. **BSSN RHS** (2-3 weeks): replace `_rhs_unfused` in `main.py` with BSSN equations
2. **Gauge conditions** (1 week): 1+log lapse, Γ-driver shift
3. **Moving punctures** (1-2 weeks): puncture-tracking refinement criterion, gauge initialization
4. **Initial data** (1-2 months OR shortcut by importing from existing tool): Bowen-York
5. **Validate Schwarzschild then Kerr then BBH** (2-3 months including bug-hunting)

Total to first BBH inspiral: ~5-8 months of focused work after AMR is settled.

---

## Open questions (resolve before Phase 1)

1. **Block size for AMR.** Match existing `fused_*` tile size (BS=32) or smaller (BS=16)? Smaller blocks = finer refinement granularity but more bookkeeping.
2. **Refinement criterion for MCS.** Gradient of Ez is the obvious one. Plain enough.
3. **Boundary conditions for non-trivial AMR.** Sommerfeld BC on the *coarse* outer boundary; periodic doesn't quite make sense with refined regions. Need to verify our existing BC code handles refined patches at the outer boundary correctly.
4. **Validation oracle.** Do we have an analytic solution for the wave + refined region? The birefringent wave's analytical solution doesn't change — refinement should improve accuracy in the refined region, not change the physical answer.

These are tractable; flagging so we resolve them in the first day of Phase 1 instead of mid-implementation.
