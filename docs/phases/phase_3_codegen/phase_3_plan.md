# Phase 3 — BSSN efficiency I: GPU-optimal codegen (the PRIMARY lever)

**Status: 🔵 ACTIVE.** 3.0 ✅, 3.1 ✅ (controllability answered → **Pallas**; see §3).
Now on 3.2 (the Pallas algebra kernel). The thesis contribution that doesn't exist
anywhere: a **register-bounded, fusion-first BSSN RHS generator** that makes the
high-order GR RHS compute-bound on the GPU. Phase 2 left a *correct but
memory-bound* RHS; Phase 3 makes it *fast* without changing the math. Tensor cores
(Phase 4) are the secondary, Amdahl-capped multiplier on top.

> Strategy spine: Phase 2 (correct, ✅) → **Phase 3 (codegen, PRIMARY)** → Phase 4
> (Ozaki derivative, SECONDARY). See `../README.md` ⚑ and
> `../phase_2_bssn_correctness/bssn_port_plan.md` §"honest framing".

---

## 0. The honest thesis (what Phase 2 proved, what Phase 3 must deliver)

**Measured baseline (Phase 2.2, H200, `bssn3d/spill_probe.py`, N=48):** XLA compiles
the verbatim CSE RHS into **79 fusion kernels**; the pointwise-algebra fusions are
**pegged at the 255-register ceiling**, and 3 **spill to local memory**
(`loop_add_multiply_subtract_fusion` 255 reg / 1168 B, `…_divide…` 255 / 728,
`loop_add_subtract_fusion_2` 255 / 640; total 2536 B), with several more at
255/178/168/128. This is the empirical form of the static-dataflow prediction
(**584 simultaneously-live temps vs 255 registers**; `[[bssn-codegen-staging]]`).

**Interpretation (do not overstate):** the headline is *fragmentation + the
register ceiling*, not catastrophic spill (2.5 KB is modest). XLA's response to the
register pressure is to **split** the RHS into many kernels, so intermediates
round-trip through HBM between them → the "compute-bound (62 FLOP/byte)" RHS is in
practice **memory-bound on its own intermediates**. "62 FLOP/byte" is a
perfect-cache ceiling.

**Phase-3 claim to earn:** *a fusion-first, register-bounded generator collapses
the RHS into a few kernels that keep intermediates on-chip and stay under 255
registers → the regime flips to compute-bound (MEM%↓/SM%↑, spill→0), at unchanged
FP64 accuracy.* The deliverable is the **generator** (automatic cut-point / store-set
selection), not a one-off hand-tuned kernel.

---

## 1. The landing pad (what Phase 2 hands us)

| Asset | State | Phase-3 use |
|---|---|---|
| Verbatim RHS (`_bssn_rhs_generated.py`) | ✅ bit-matches Dendro-GR to **3.1e-16** | the *reference* the staged RHS must reproduce |
| Dendro-GR `g++` bit-compare oracle (`oracle.py`) | ✅ CPU, no 2FA | **regression guard**: staged RHS still matches (looser tol; see risks) |
| Apples tests (constraints ~6th, stability) | ✅ | regression guard the staged RHS must still pass |
| Spill probe (`spill_probe.py`) | ✅ ptxas reg/spill, arch-detect | the **metric** for 3.3 (spill→0) |
| `profile_regime --smi` (MCS) | ✅ (2D/3D MCS) | port to BSSN for the MEM%↓/SM%↑ regime flip |
| Static-liveness analysis | 584-live measured | seeds 3.1 hand-staging + 3.2 automation |
| Derivative bundle (138 arrays) | ✅ | the ~30% derivative share (Phase 4 target; here just keep correct) |

**Dendro front-end we can reuse (not re-derive):** `~/Code/Dendro-GR/CodeGen/`
`bssn.py`/`dendro.py` (the SymPy tensor hierarchy: inverse metric → Christoffel →
Ricci → Lie), `bssn_manually_staged.py` (proves hand-staging works, `MAX_TEMP_VARS=64`),
and the **`bssn_nx.py` / `nxgraph.py` / `bssneqs_nx_cse_wo_derivs.cpp`** path (the
graph-scheduled "nx" variant — Dendro's own attempt at register-aware scheduling,
the closest prior art to automate). Vendor what we use.

---

## 2. The design (decided in CLAUDE.md "3D BSSN GPU kernel architecture")

The RHS splits into two near-separable halves on **two different on-chip pools**:

- **Derivatives (~30% FLOP):** the only neighbor-reading part → all the halo/reuse
  cost; **field-streamed 2.5D SMEM** (stream one field's halo at a time; temporal
  fusion *rejected* for 24 fields). This is the Phase-4 tensor-core target; in
  Phase 3 it just stays correct and fused-adjacent.
- **Pointwise algebra (~70% FLOP):** Christoffel/Ricci/gauge; reads only same-point
  values → **no halo, compute-bound, the register pool**. This is where the
  584-live spill lives and where Phase 3 acts.

**The register pool is the whole game.** Both extremes are dead: recompute-everything
= 623k raw ops (~150×); store-everything (plain CSE) = 584-live spill. The optimum is
**selective materialization** — store the few high-fan-out tensor-hierarchy temps
(inverse metric, Christoffels, Ricci, CalGt — ~69) as **per-point register scalars**,
recompute the ~800 cheap leaves → ~133 live, fits < 255. The register pressure
concentrates at a physics-mandated choke (inverse metric needs all 6 g̃; Ricci sums
all 36 ∂²g̃), so the cut-set is structural, not a reorderable spike.

> **The recompute half is the part XLA cannot do (Step 3.1 finding).** XLA computes
> each forward SSA value once and either keeps it live or barriers it to HBM — it
> never *recomputes* a cheap leaf to shrink a fused kernel's live set (`remat` is an
> autodiff-only lever; the forward RHS has no backward pass). So "recompute the ~800
> leaves" is **only expressible in a hand-authored kernel** → emission target =
> **Pallas** (own the register file), recompute written explicitly from the generator's
> schedule. Pallas is *necessary but not sufficient*: a verbatim 850-stmt dump still
> spills in Triton — the substance is the materialize/recompute schedule (3.2b).

---

## 3. Steps

### Step 3.0 — Precursor: lock the production RHS variant (incl. constraint damping) — ✅ DONE (2026-06-11)
Codegen should stage the variant we will actually run, so decide **before** optimizing:
- **No-CAHD** (current, bit-validated `bssneqs_sympy_cse_wo_derivs.cpp`) — simplest,
  but momentum drifts undamped (seen in 2.3).
- **CAHD + SSL** (`bssneqs_SSL_HD_dxsq.cpp`) — adds Hamiltonian-constraint damping
  (`chi_rhs += C_CAHD·chi·(dx²/dt)·H`, cahd_c~0.06) + spatial slice-locking. A few
  extra terms; re-transliterate through the *same* pipeline + re-validate via the
  oracle + apples. (Momentum stays under-protected even here — only Gt + KO; full
  momentum damping is a Z4c/Phase-6 question, out of scope.)
- **DECIDED (2026-06-11): CAHD+SSL.** Switch the production variant to
  `bssneqs_SSL_HD_dxsq.cpp` now (cheap, and what long runs need), re-run the
  bit-compare + apples as Phase-2-style gates, and stage *that* in 3.1+. Constraint
  damping was otherwise only a ROADMAP F.6/I.2 TODO — this is where it lands.
  **Scope check (audit, done 2026-06-11):** `bssneqs_SSL_HD_dxsq.cpp` (885 lines) is
  CLEAN for our pipeline — **no `agrad_`/`kograd_`** (centered FD suffices), the
  **same 138 derivative inputs** (72 grad1 + 66 grad2 → bundle unchanged), same 24
  outputs. Only deltas: two new functions `sqrt` + `exp` (the SSL time-ramp) → add
  `sqrt(`→`jnp.sqrt(`, `exp(`→`jnp.exp(` to the translator; and new scalar params
  `BSSN_CAHD_C` (~0.06), `dt`, `dx_i` (spacing, for the dx²/dt CAHD term), `h_ssl`,
  `sig_ssl`, time `t`. ⇒ the Phase-2 pipeline (translator → generated module →
  bundle → `g++` oracle → apples) is reusable almost verbatim; the oracle harness
  just declares the new scalars. Low risk.

### Step 3.1 — Controllability probe (XLA barriers vs Pallas) — ✅ DONE (2026-06-12)
Built `bssn3d/staging.py` (fan-out/DAG analysis: 826 temps, 24 outputs, 4527 ops;
69 single-use vs 757 reused — reuse is structurally broad, top-69 store covers only
~59% of `fanout*cone`), emitted `_bssn_rhs_staged.py` (69 `optimization_barrier` cuts
at the top fan-out×cone temps, via a shared `_codegen._emit_module` so the verbatim
oracle module's bytes are unchanged), wired `scheme="staged"` into `BSSNSolver` +
`spill_probe` (`BSSN_SCHEME=staged`), and `test_bssn_staged.py` (staged == verbatim
**bit-identical** — barriers are numerical no-ops).

**Re-baseline (CAHD+SSL verbatim, H200, N=48):** 97 kernels, 2 spilling, **1104 B**,
max 255 regs (vs no-CAHD 79/3/2536 B — more fragmented, less spill; the Hamiltonian
constraint in `chi_rhs` made XLA split rather than spill harder).

**ANSWER — Pallas required.** Staged A/B: **107 kernels, 0 spilling, 0 B, max 254 regs.**
Barriers **eliminate the spill** → `ptxas` *does* respond to `optimization_barrier`
(XLA is register-controllable), **but only by *splitting*** (97→107 kernels — the
barrier lever trades spill for HBM-round-tripping kernels). And **`remat` is not a
second lever:** `jax.checkpoint` only acts during autodiff; the RHS runs forward (RK4,
no grad) → no-op (`prevent_cse=True` degenerates to a barrier). So XLA's only forward
knob is splitting → it cannot give few-kernels-AND-register-bound, and the §2 selective-
**recompute** design is **inexpressible in XLA forward eval**. The register-resident
kernel must be authored where we own the register file: **Pallas.** The staged-XLA RHS
is kept as a 0-spill **regression baseline**, not the optimization target.

### Step 3.2 — The Pallas algebra kernel (the contribution), in three parts
The generator is unchanged in spirit (automatic store/recompute schedule from the DAG)
but the **emission target is now Pallas**, and the schedule must add the **recompute
set** XLA couldn't express.

- **3.2a — BSSN regime profiler + the motivating measurement. ✅ DONE (2026-06-12).**
  Built `bssn3d/profile_regime.py` (`--smi` nvidia-smi dmon SM%/MEM%, `--jax-trace`,
  `--scheme verbatim|staged`; loops `s + eps·rhs(s)`). **H200, N=128³:** staged-XLA =
  **SM 100% / MEM 98% → memory-bound**; verbatim = SM 100% / MEM 97% → memory-bound.
  **Confirmed: the despill did NOT flip the regime** (SM% is "warp resident," not
  throughput; MEM 98% is the DRAM tell). Earns Pallas by measurement.

> **⚠️ TWO memory walls — the roofline correction (3.2a).** Why the staged RHS is still
> memory-bound has *two* causes, and the algebra-only Pallas kernel (3.2c) clears only
> the first:
> 1. **Inter-kernel intermediate round-trips** — the 107 fragments stream the 826 CSE
>    temps through HBM. **3.2c fixes this** (register-resident algebra, ~3× traffic cut).
> 2. **The 138 derivative arrays** — the `_wo_derivs` design materializes every
>    `grad_*`/`grad2_*` to HBM and reads them back (~1100 B/point of *mandatory* read an
>    algebra-only kernel **cannot** remove). This caps the algebra-only intensity at
>    ~3–6 FLOP/byte — at/below the H200 FP64 balance (~7) → **still memory-bound.**
>
> The "62 FLOP/byte → compute-bound" figure was always a *perfect-cache* number that
> assumed derivatives computed **on-chip from the field halo**, never materialized. So
> **the regime flip needs derivative fusion (3.2d), not algebra-only.** 3.2c stays as the
> register-control de-risking step (independently bit-validatable); the flip is 3.3 on
> the *fused* kernel. (Decision 2026-06-12: algebra-only first, then fuse — sequential.)

- **3.2b — Materialize-set + recompute-set generator. ✅ DONE (2026-06-12).** Extended
  `staging.py`: `persistent_liveness(M)` (peak simultaneously-live M-temps under the
  realized schedule), `recompute_ops(M)` (emitted ops with R inlined per use),
  `liveness_cost_curve`, and `select_schedule(budget)`. **Store-vs-recompute Pareto
  (top-K by fan-out×cone):** recompute-all (K=0) = 0 regs / **55×** ops; store-all
  (K=826) = **452** live / 1.0×; **selected (budget 200): |M|=288 → peak 200 persistent
  regs, only 1.43× recompute** (6469 vs 4527 ops). The **recompute lever XLA lacked is
  now expressible** — 1.43× ops buys a fit under 255. `schedule_pylines(dag, M)` emits
  the realization (M-temps stored, R-temps inlined) — the artifact 3.2c's Pallas backend
  consumes; `test_bssn_schedule.py` (11 green) proves the partition equals verbatim to
  round-off for K=0/64/288/826 + the liveness/cost model is sane.
- **3.2c — Emit the algebra-only Pallas kernel** (derivatives-as-inputs — isolate the
  register pool 3.1 localized to the algebra). One kernel, intermediates as register
  scalars, recompute authored explicitly. **CPU gate ✅ DONE (2026-06-12):**
  `bssn3d/pallas_backend.py` emits `_bssn_rhs_pallas.py` — a **pointwise** kernel
  (stack 24 fields + 138 derivs → `(162, Npts)`, tile along points; `schedule_pylines`
  body with M stored / R inlined). Wired `scheme="pallas"` into `BSSNSolver`,
  `spill_probe`, `profile_regime`. `test_bssn_pallas.py` (6 green, **interpret mode**):
  == verbatim to round-off (pointwise <1e-10; gridded gauge-wave mixed atol 1e-11/rtol
  1e-9 — the 1.43× recompute reorders fp summation, near-zero RHS components cancel).
  **GPU gate ⬜ (2FA):** `BSSN_SCHEME=pallas python -m bssn3d.spill_probe` → does the
  Triton kernel compile + what registers/spill? **Expected regime:** despill + ~3×
  traffic cut, but *still memory-bound* on the 138-array derivative read (wall 2) —
  designed-for, 3.2d clears it. ⚠️ Local Pallas is **interpret-mode only** (math, not
  compilation); the Triton compile may be slow (the kernel is large straight-line, not
  looped — likely under the ~1600 s unrolled-`pallas_ozaki` wall, but watch it); the
  PTX dump may need a Triton-specific extraction if `spill_probe`'s XLA dump misses the
  Triton kernel.
- **3.2d — Fuse the derivatives on-chip (clears wall 2).** Field-streamed 2.5D-SMEM
  derivative stage (read each field's halo once into SMEM, compute its `grad_*`/`grad2_*`
  on-chip, feed the algebra) so the 138 derivative arrays are **never materialized to
  HBM**. Couples the halo/SMEM pool with the register pool — the harder kernel, taken
  *after* 3.2c validates the algebra in isolation. (This is the part the old plan
  deferred to "Phase 4"; the 3.2a roofline pulls it into Phase 3 as the regime-flip
  prerequisite. The Ozaki/tensor-core derivative stays Phase 4.)

### Step 3.3 — Confirm the regime flip (on the fused Pallas kernel)
On the H200 (user 2FA): the **fused (3.2d) Pallas kernel** shows `ptxas` **spill→0**,
**few kernels**, and `profile_regime --smi` **MEM%↓ / SM%↑** vs the staged-XLA baseline
(SM 100% / MEM 98%); FP64 accuracy unchanged (oracle + apples). Record the before/after
(verbatim 97 kernels/1104 B/MEM 97% → staged-XLA 107/0 B/MEM 98% → **fused Pallas: few
kernels, compute-bound, MEM%↓**) as the Phase-3 result. (3.2c alone is expected to cut
traffic but stay memory-bound on the derivative read — the flip needs 3.2d.)

---

## 4. Reuse vs build vs audit
- **Reuse:** the verbatim RHS + oracle + apples (as regression harness), `spill_probe`,
  Dendro's SymPy tensor-hierarchy front-end, the `bssn_nx` graph-scheduler prior art,
  `profile_regime`.
- **Build:** ✅ the staged RHS + `staging.py` (3.1), ✅ the BSSN `profile_regime.py`
  port (3.2a), ✅ the staged↔verbatim equivalence test; the **materialize/recompute
  generator** (3.2b), the **algebra-only Pallas kernel** (3.2c), the **fused 2.5D-SMEM
  derivative kernel** (3.2d).
- **Audit:** ✅ **answered (3.1):** `optimization_barrier` de-spills but fragments;
  `remat` is a forward no-op → **Pallas is the emission target** (3.2c).

## 5. Risks / gotchas
1. **XLA controllability — ✅ RESOLVED (3.1): Pallas it is.** `optimization_barrier`
   de-spills but fragments (97→107 kernels); `remat` is a no-op on the forward RHS.
   So 3.2 commits to **Pallas** — inheriting its costs: local **interpret-mode only**
   (math, not compilation), the **~1600 s compile wall** (scan-roll unrolled loops),
   the `pallas_ozaki` constraints (no fp64 `dot` — irrelevant until Phase 4's GEMM).
   The heavy GPU-iteration loop is the new top risk: every real spill check is a 2FA run.
2. **Staged ≠ bit-identical to the verbatim RHS.** Recompute/reorder changes fp
   summation order → expect agreement to ~round-off (~1e-12), not 3e-16. The oracle
   gate for Phase 3 is "matches verbatim to round-off", and the apples tests must
   still pass — *not* bit-equality.
3. **Amdahl honesty.** Codegen targets the ~70% algebra (register pool). Derivatives
   (~30%) are Phase 4; a perfect derivative speedup caps at ~1.4× on the full RHS.
   Report the algebra-share win and the full-RHS win separately.
4. **Measurement honesty.** `spill_probe` ptxas numbers are grid-size-independent
   (pointwise kernels); the regime flip needs the `--smi` MEM%/SM% read (counters
   locked, Nsight banned). Don't claim "compute-bound" without the MEM%↓/SM%↑ flip.
5. **Variant churn.** If 3.0 switches to CAHD+SSL, re-baseline the spill on *that*
   RHS first (its extra dx²/H terms shift the live set) so 3.1 optimizes the real target.
6. **Don't drift into Phase 4.** No Ozaki/tensor-core/INT8 here — Phase 3 is FP64
   fusion only. The derivative stays the Phase-1 centered FD.

## 6. Exit criteria
- ✅ Production RHS variant locked (3.0), re-validated (oracle + apples) if changed.
- ✅ Controllability answered (3.1): staged-XLA de-spills but fragments; `remat` no-op
  → **Pallas emission target**. Staged-XLA RHS kept as 0-spill regression baseline.
- ✅ Staged-XLA confirmed still HBM-bound (3.2a, MEM 98% @ N=128) — earns Pallas; the
  roofline shows **two memory walls** (intermediate round-trips + the 138-array
  derivative read), so the flip needs derivative fusion, not algebra-only.
- ⬜ Materialize/recompute-set generator (3.2b), regen-guarded, peak-live < 255 by
  construction (the **recompute** set is the new lever vs the 3.1 barrier cut-set).
- ⬜ **Algebra-only Pallas kernel (3.2c)** emitted from the schedule, derivs-as-inputs;
  matches verbatim to round-off + apples pass; `ptxas` 0 spill; traffic cut but expected
  *still memory-bound* on the derivative read.
- ⬜ **Fused 2.5D-SMEM derivative kernel (3.2d)** — derivatives on-chip, 138 arrays never
  materialized (clears wall 2).
- ⬜ **Regime flip measured (3.3):** fused Pallas kernel spill→0 + few kernels +
  `profile_regime --smi` **MEM%↓/SM%↑**, FP64 accuracy unchanged. (GPU-gated, user 2FA.)
- ⬜ Before/after recorded (verbatim 97/1104 B/MEM 97% → staged-XLA 107/0 B/MEM 98% →
  fused Pallas few-kernel compute-bound).

## 7. Suggested order of work
3.0 lock variant ✅ → 3.1 controllability probe ✅ (Pallas) → 3.2a BSSN regime profiler
+ confirm staged-XLA HBM-bound ✅ (MEM 98%; two walls found) → **3.2b** materialize/
recompute generator (CPU dev) → **3.2c** algebra-only Pallas kernel, derivs-as-inputs
(CPU interpret-mode + H200 spill checks; despills but stays mem-bound on wall 2) →
**3.2d** fuse derivatives on-chip (2.5D-SMEM, clears wall 2) → **3.3** confirm the regime
flip on the fused Pallas kernel (H200) → Phase 4 (Ozaki derivative) with a compute-bound RHS.

## Status / changelog
- 2026-06-12 — **Step 3.2c CPU gate DONE → register-resident Pallas kernel emitted.**
  `bssn3d/pallas_backend.py` → `_bssn_rhs_pallas.py`: pointwise kernel (24 fields + 138
  derivs stacked to `(162, Npts)`, tiled along points; `schedule_pylines` body, M stored
  / R inlined; budget=200 → |M|=288, 1.43× recompute). `scheme="pallas"` wired into
  `BSSNSolver`/`spill_probe`/`profile_regime`. `test_bssn_pallas.py` 6 green in
  **interpret mode** (== verbatim to round-off; gridded uses mixed atol 1e-11/rtol 1e-9
  for near-zero gauge-wave RHS cancellation under the 1.43× reorder). **Remaining: the
  GPU gate** (`BSSN_SCHEME=pallas spill_probe` — Triton compile + register/spill; PTX
  extraction may need Triton-specific handling) then 3.2d derivative fusion + 3.3 flip.
- 2026-06-12 — **Step 3.2b DONE → materialize/recompute schedule generator.** Extended
  `staging.py` with `persistent_liveness`, `recompute_ops`, `liveness_cost_curve`,
  `select_schedule`, `schedule_pylines`. Store-vs-recompute Pareto: recompute-all 55×
  ops/0 regs → store-all 452 live/1.0× → **selected (budget 200): |M|=288, peak 200
  regs, 1.43× recompute**. The recompute lever XLA couldn't express is now a generator;
  `schedule_pylines` is the emittable realization for the 3.2c Pallas backend.
  `test_bssn_schedule.py` (11 green): partition == verbatim to round-off (K=0/64/288/826)
  + liveness/cost model sane. CPU-only, no GPU. Next: 3.2c (Pallas emission).
- 2026-06-12 — **Step 3.2a DONE → BSSN still memory-bound; TWO walls found.** Built
  `bssn3d/profile_regime.py` (`--smi`/`--jax-trace`/`--scheme`). H200 N=128³: staged-XLA
  **SM 100% / MEM 98%**, verbatim SM 100% / MEM 97% → both **memory-bound** despite the
  staged despill. Roofline: the regime has *two* causes — (1) inter-kernel intermediate
  round-trips (fixed by the 3.2c register-resident algebra kernel, ~3× cut) and (2) the
  **138 derivative arrays materialized to HBM** by the `_wo_derivs` design (~1100 B/pt
  mandatory read → caps algebra-only intensity at ~3–6 FLOP/B ≤ H200 balance ~7). So
  algebra-only stays memory-bound; the regime flip needs **derivative fusion**. Added
  **Step 3.2d** (field-streamed 2.5D-SMEM derivatives on-chip, 138 arrays never
  materialized) before 3.3's flip; **decision (user): algebra-only first (de-risk
  register control), then fuse — sequential.** "62 FLOP/byte" was always a perfect-cache
  figure assuming on-chip derivatives. Detail: `[[bssn-codegen-staging]]`.
- 2026-06-12 — **Step 3.1 DONE → emission target resolved to PALLAS; Phase 3
  restructured.** CAHD+SSL spill re-baseline (H200, N=48): **97 kernels / 2 spilling /
  1104 B / 255 regs** (no-CAHD 2.2 was 79/3/2536 B). Built `bssn3d/staging.py`
  (fan-out/DAG analysis; 826 temps, 69 single-use vs 757 reused → broad structural
  reuse) + `_bssn_rhs_staged.py` (69 `optimization_barrier` cuts via shared
  `_codegen._emit_module`, verbatim oracle bytes unchanged) + `scheme="staged"` in
  `BSSNSolver`/`spill_probe` + `test_bssn_staged.py` (staged == verbatim bit-identical).
  **Staged A/B (H200): 107 kernels / 0 spilling / 0 B / 254 regs** — barriers de-spill
  but only by *splitting*; **`remat` is a no-op on the forward RHS** (autodiff-only) →
  XLA can't give few-kernels+register-bound, and selective-recompute is inexpressible
  in XLA forward. **→ Pallas committed** (3.2c); staged-XLA kept as a 0-spill baseline.
  Restructured 3.2 into 3.2a (BSSN regime profiler + confirm staged-XLA HBM-bound) /
  3.2b (materialize+**recompute**-set generator) / 3.2c (algebra-only Pallas kernel,
  derivs-as-inputs); 3.3 now confirms the flip on the Pallas kernel. Banner + HANDOFF
  §2/§4 updated in lockstep. Detail: `step_3.1_handstage.md`, `[[bssn-codegen-staging]]`.
- 2026-06-11 — **Step 3.0 DONE: production RHS variant locked to CAHD+SSL.** Vendored
  `bssneqs_SSL_HD_dxsq.cpp` (708887 ops, 850 stmts, sha256 `7b5353750d4240dd…`) and
  re-ran the **entire Phase-2 pipeline** through it: `_codegen.py` extended (sqrt/exp
  → `jnp.`, 6 new scalar params `BSSN_CAHD_C/dt/dx_i/h_ssl/sig_ssl/t` in the signature,
  a **scalar-param drift-guard** that fails fast if a refreshed CSE adds/drops a param);
  regenerated `_bssn_rhs_generated.py`; `PhysicsParams` now *consumes* `cahd_c=0.06`/
  `ssl_h=0.6`/`ssl_sigma=20.0` (Dendro `q1.par.toml` defaults); `BSSNSolver`/
  `BSSNEvolution` thread the runtime scalars (`t` per RK4 substage — the RHS is now
  **time-dependent** via the SSL Gaussian — and `dt`/`dx_i` for the CAHD `dx²/dt`
  factor); `oracle.py` declares the new scalars + derivs-as-arrays (the SSL variant
  indexes `grad…[pp]`, the no-CAHD one used them bare). **Gates re-passed:** oracle
  bit-compare vs Dendro-GR C++ = **1.93e-16** (SSL/CAHD terms bit-identical); 29/29
  BSSN tests green incl. the slow apples evolutions (CAHD+SSL active); Minkowski stays
  static (both CAHD-H and SSL `√chi−α` vanish at flat space). **Next: Step 3.1**
  (`step_3.1_handstage.md`) — but first **re-baseline the spill on the CAHD+SSL RHS**
  (the 79-kernel/2536 B map was the *no-CAHD* variant; CAHD moves the live set,
  esp. `chi_rhs`). 3.1 then probes XLA controllability (remat/barrier vs Pallas).
- 2026-06-11 — Created on Phase 2 completion. Grounded in the measured 2.2 spill
  baseline (79 kernels / 255-reg ceiling / 2536 B) and the static 584-live analysis.
  Folds the previously-unscheduled **constraint-damping (CAHD) decision** into 3.0 as
  the production-variant precursor (it was only a reserved `cahd_c` param + ROADMAP
  F.6/I.2 TODOs before).
