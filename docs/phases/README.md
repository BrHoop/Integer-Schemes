# Project Phases — Execution Tracker

The **canonical, ordered execution view** of the MCS → BSSN tensor-core
numerical-relativity project. One linear spine, no backtracking: foundation →
BSSN correctness → BSSN efficiency (codegen, then Ozaki) → grid/frame → BBH.

`../ROADMAP.md` is the *thematic backlog* (organized by area, mapped onto these
phases); `../AMR_PLAN.md` is the *detail doc* for the AMR thread (2D complete; the
3D port is Phase 5 here). Companion docs in `../`: `HANDOFF` (narrative),
`ARCHITECTURE` (the why), `OPTIMIZATION` (measured GPU baseline), `OZAKI_GPU_NOTES`.

**Docs are written just-in-time:** only the *active* step gets a full doc; future
steps live as table rows here until reached (plans change — no stale stubs).

Legend: ✅ done · 🔵 active · ⬜ planned · ⚰️ retired

---

## ⚑ Strategy (2026-06-11) — read first

The arc that shaped this spine (earned through Phases 0–earlier, do not re-litigate):

1. **MCS is the correctness / Ozaki-reproducibility ground, not the efficiency
   story.** It is compute-light (~15 FLOP/byte) → DRAM-bound (measured, Phase 0.2)
   → tensor cores can't help. MCS stays as the cheap oracle for bit-reproducibility.
2. **BSSN in 3D is the efficiency ground** (24 fields, ~24k FLOP/pt/step) — the
   only place the payoff can be *demonstrated*. So the project marches to a 3D
   BSSN RHS and shows the result there.
3. **Codegen is the PRIMARY lever; tensor cores are SECONDARY.** BSSN's RHS is
   ~70% point-wise nonlinear algebra (not GEMM-able) → Amdahl caps a perfect
   derivative speedup at **~1.4×**. And the symbolic-CSE RHS overruns the register
   file (**static dataflow: peak ~584 simultaneously-live temps vs 255 registers**,
   measured on Dendro-GR's `bssneqs_sympy_cse_wo_derivs.cpp`) → it is **memory-bound
   on its own intermediates**. The honest thesis: *high-order GR RHS is memory-bound
   on GPUs because symbolic-CSE codegen overruns the register file; a fusion-first,
   register-bounded generator makes it compute-bound, and tensor cores then
   accelerate the stencil portion.* → **codegen = Phase 3 (primary)**, Ozaki
   derivative = **Phase 4 (secondary, at its deployment)**.
   **Emission target = Pallas (resolved Phase 3.1, 2026-06-12):** XLA
   `optimization_barrier` eliminates the spill but only by *splitting* into more
   kernels, and `remat` is a no-op on the forward (non-differentiated) RHS — so the
   register-resident, recompute-where-cheap kernel must be authored in **Pallas**,
   not coaxed out of XLA. Staged-XLA RHS = 0-spill regression baseline.

**Port source = Dendro-GR (NOT DendroJax).** DendroJax's GR implementation is buggy
→ disqualified as a correctness oracle. Port the BSSN RHS from **`~/Code/Dendro-GR`**:
the verbatim step *transliterates* the generated CSE `bssneqs_sympy_cse_wo_derivs.cpp`
(942 lines, derivatives-as-inputs, flat SSA) → vectorized JAX — correct-by-construction
from trusted output, no SymPy toolchain, no DendroJax. The staged regen (Phase 3)
later drives Dendro-GR's SymPy front-end (`bssn.py`/`dendro.py`/`bssn_nx.py`).
**Oracle = Dendro-GR C++** (identical ID, one RHS eval, compare to round-off) **+
the analytic apples tests** (reference-free).

**The register-spill confirmation rides on the JAX RHS (Phase 2.2), not Dendro-GR.**
The 584-live number is a *dataflow* property (architecture-independent) and already
answers the GPU question. A *hardware* confirmation can only come from **GPU-compiled
code**: Dendro-GR is CPU C++ — its `g++` register behavior says nothing about the
NVIDIA 255-reg/`ptxas` budget, and its hand-staged CUDA is the *already-fixed*
version. So the spill check is one `ptxas -v` compile (`--xla_gpu_asm_extra_flags=-v`)
of *our* JAX RHS, telling us the fork that picks the codegen fix: XLA **spills to
local memory** vs **splits into many kernels with grid-sized HBM intermediates**.

> *Octree rejected as the long-term home* (wave-zone/dual-frame/cubed-sphere need
> shell grids). Use Dendro-GR once as the porting/validation oracle, then walk away.

---

## Doc directory map (directory names now match the phase numbers)

| Directory | Holds | Phase |
|---|---|---|
| `phase_0_2d_foundation/` | `step_0.1`/`0.2` (done) + retired `step_0.3`; `step_0.1_results/` | **Phase 0** ✅ |
| `phase_1_3d_foundation/` | `phase_1_plan.md` + `step_1.1_results/` | **Phase 1** ✅ |
| `phase_2_bssn_correctness/` | `bssn_port_plan.md` + `phase_2_plan.md` | **Phase 2** ✅ |
| `phase_3_codegen/` | `phase_3_plan.md` | **Phase 3** 🔵 (active) |

Renamed 2026-06-11 so the on-disk layout matches this canonical tracker (the old
thematic names `phase_1_stencil_core`/`phase_3_bssn_physics` + empty stubs are
gone). Code output paths were updated in lockstep: `step_0.1_results/` is wired
into `validate.py`/`msrk_analysis.py`/`sync.sh`. Phases 3–6 get directories
just-in-time when they go active (no stub dirs).

---

## Phase 0 — 2D MCS foundation & regime — ✅ DONE
The correctness/reproducibility ground. Established that MCS is DRAM-bound, so the
efficiency story must move to 3D BSSN.

| Step | Title | Status | Doc |
|---|---|---|---|
| 0.1 | Baseline validation + profiling harness | ✅ | `phase_0_2d_foundation/step_0.1_baseline_validation.md` (+ `step_0.1_results/`) |
| 0.2 | Complete the roofline + GPU regime → **DRAM-bound** | ✅ | `phase_0_2d_foundation/step_0.2_complete_roofline.md` |
| — | ⚰️ **Retired:** 2D shared-memory tiling + temporal fusion (`step_0.3`), and the 2D-solver stencil-as-GEMM / 2:4 / order / precision framing (old 1.4–1.8) | ⚰️ | `phase_0_2d_foundation/step_0.3_tile_and_fuse.md` (tombstone) |

> **Why retired:** 2D tiling/temporal-fusion is a toy-model artifact — it doesn't
> transfer to 3D (24 fields → tiles too big) and isn't needed once compute-bound.
> The *derivative-as-GEMM on tensor cores* idea survives, but as the **Ozaki
> `compute_deriv` operator (Phase 4)**, not as a 2D-solver throughput play.

## Phase 1 — 3D foundation (MCS in 3D) — ✅ DONE
CPU-developable; the infrastructure every later phase lands on. **Plan:
`phase_1_3d_foundation/phase_1_plan.md`** (see §7 for the validation results).
Audit (2026-06-11) found the 3D solver **functionally complete but
unvalidated/untested** — so Phase 1 was *validation + machinery-port + correctness
audit*, not physics-building. Both correctness questions the audit raised resolved
positively (symbol↔Jacobian 2.4e-8; the 3D oracle is an exact solution → 6th-order).

| Step | Title | Status |
|---|---|---|
| 1.1 | 3D state + 6th-order 3D FD (1st, diagonal-2nd, mixed-2nd, KO) — the exact operators the BSSN RHS consumes | ✅ audited (bit-identical to 2D at order 6; RHS reduces to 2D on z-invariant data ≤1e-12) |
| 1.2 | 3D BCs: periodic + Sommerfeld outer | ✅ periodic energy-conservative to round-off; Sommerfeld passive + absorbing |
| 1.3 | Port `validate.py` → 3D; validate vs the **3D birefringent oracle** (convergence order, constraint damping, semi-discrete spectrum) | ✅ stencil 6th, KO-limited 5th, RK4 floor 4th; −K/2 damping; CFJ 1/0; strong-hyperbolic (cond 4.81) |
| 1.4 | Port benchmark/roofline to 3D | ✅ ported; **H200 run done (2026-06-11): memory-bound, ~1% of FP64 peak** — matches the 2D regime |

Exit: ✅ a validated 3D MCS with the full machinery (`validate.py`, `benchmark.py`,
41-test suite, `run.sh`) **and the H200 regime measured (memory-bound)** — **BSSN
(Phase 2) is now an RHS swap onto it.** Phase 1 is fully closed.

## Phase 2 — BSSN correctness (port from Dendro-GR) — ✅ DONE (2026-06-11)
Full plan: `phase_2_bssn_correctness/phase_2_plan.md` (execution plan) +
`phase_2_bssn_correctness/bssn_port_plan.md` (the strategy/port narrative).

| Step | Title | Status |
|---|---|---|
| 2.1 | BSSN 24-var state + `PhysicsParams` (eta/CAHD/SSL/λ gauge knobs; chi/alpha floors) | ✅ `bssn3d` pkg; `SpatialDerivative` → `common` |
| 2.2 | **Verbatim RHS:** transliterate Dendro-GR `bssneqs_sympy_cse_wo_derivs.cpp` → JAX (fed by Phase-1 derivs); validate vs **Dendro-GR C++**. **+ the XLA `ptxas -v` spill measurement** | ✅ bit-compare **3.1e-16**; spill: **79 kernels, 255-reg ceiling, 2536 B spill** |
| 2.3 | Physics validation: gauge wave, robust stability, Hamiltonian/momentum constraint convergence (reference-free) | ✅ Ham/Mom converge **~6th**; stable; conformal algebra preserved |

Exit: a 3D BSSN RHS that **bit-matches Dendro-GR C++** (3.1e-16, CPU `g++` oracle) and
passes the reference-free apples tests, with the `ptxas` spill number in hand. The
vendored Dendro sources (`bssn3d/vendor/`) make the port self-contained. **→ Phase 3.**

## Phase 3 — BSSN efficiency I: GPU-optimal codegen (the PRIMARY lever) — 🔵 ACTIVE
The contribution that doesn't exist anywhere: a register-bounded, fused RHS.
Full plan: `phase_3_codegen/phase_3_plan.md`. Baseline (CAHD+SSL re-baseline 3.1,
2026-06-12): **97 fusion kernels, 255-reg ceiling, 1104 B spill** (no-CAHD 2.2 was
79/2536 B). **3.1 resolved the emission target to Pallas** (XLA barriers de-spill but
fragment; `remat` is a forward no-op) — the staged-XLA RHS is a 0-spill baseline, the
Pallas algebra kernel (3.2c) is the compute-bound target.

| Step | Title | Status |
|---|---|---|
| 3.0 | **Precursor:** lock the production RHS variant — the **CAHD+SSL** variant (`bssneqs_SSL_HD_dxsq.cpp`); re-validate via oracle+apples. Stage the variant we actually run | ✅ DONE — variant locked, oracle **1.93e-16**, 29/29 tests green; drift-guarded codegen; runtime scalars (`t`/`dt`/`dx_i`) threaded |
| 3.1 | Hand-stage probe: `optimization_barrier` cut-set on the ~69 high-fan-out temps. **Answer: can XLA be forced register-bounded via remat/barrier, or is Pallas required?** | ✅ **ANSWERED (2026-06-12) → Pallas.** Barriers de-spill (1104 B→0, regs 255→254) but only by *splitting* (97→107 kernels); `remat` is a no-op on the forward RHS → XLA can't give few-kernels+register-bound, and selective-recompute is inexpressible in XLA forward. Staged-XLA kept as 0-spill baseline. `staging.py` + `_bssn_rhs_staged.py` + `scheme="staged"`. Plan: `step_3.1_handstage.md` |
| 3.2a | `bssn3d/profile_regime.py` (`--smi`); measure staged-XLA regime | ✅ **(2026-06-12)** N=128³ staged-XLA **SM 100% / MEM 98% → still memory-bound** despite 0 spill (verbatim MEM 97%). Roofline → **two walls**: intermediate round-trips (3.2c fixes) + the **138 derivative arrays read from HBM** (`_wo_derivs`, ~1100 B/pt) → caps algebra-only at ~3–6 FLOP/B ≤ H200 balance. Flip needs derivative fusion |
| 3.2b | `staging.py` **materialize/recompute schedule** generator (the lever XLA lacks) | ✅ **(2026-06-12)** persistent-liveness + recompute-cost model + Pareto curve. Recompute-all = 55× ops/0 regs; store-all = 452 live/1.0×; **selected (budget 200): \|M\|=288 → 200 regs, 1.43× recompute**. `schedule_pylines` emits the realization for 3.2c; `test_bssn_schedule.py` 11 green (== verbatim to round-off) |
| 3.2c | Emit the **algebra-only Pallas kernel** (derivs-as-inputs, `pallas_backend.py` → `_bssn_rhs_pallas.py`) | ✅ CPU gate (2026-06-12); `scheme="pallas"`, `test_bssn_pallas.py` 6 green. ✅ **GPU gate (H200, 2026-06-13):** **fp32 = 255 reg / 1936 B spill** (register-resident), **regime SM 100% / MEM 83% → still HBM-bound on wall 2** (down from fp64 ~96%; fp32 killed the spill, the 138-array deriv read remains). fp32 **necessary, not sufficient**. fp32 accuracy benign (CPU strong-field measure) |
| 3.2d | **Fuse derivatives on-chip** (tiled SMEM-halo; 138 arrays never materialized) — clears wall 2. The make-or-break build. Plan: `step_3.2d_fused_tiled.md` | 🔵 **Increment 1 DONE (2026-06-13): tiled on-chip derivative core LOWERS on H200** (`tiled_deriv.py`; power-of-2 HP=16/BS=8 haloed tiles, broadcast-reduce stencil+crop — no dot/transpose/slice). CPU fp64 1.46e-13, fp32 5e-5; `test_bssn_tiled_deriv.py`. Derivs in fp64 (Triton-safe, accurate) → fp32 algebra. Whole-grid prototypes (`fused`/`fused_fp64`) correctly DON'T lower (not pow2). ⬜ **Next: fuse the fp32 algebra onto this core → spill+regime push** |
| 3.3 | Confirm the regime flip **on the fused Pallas kernel** — `profile_regime --smi` MEM%↓/SM%↑ + spill→0 + few kernels → genuinely compute-bound | ⬜ |

## Phase 4 — BSSN efficiency II: Ozaki tensor-core derivative (SECONDARY, Amdahl-capped) — ⬜
Built/validated on cheap MCS, deployed in BSSN. Orthogonal multiplier on the ~30%
derivative share (~1.4× ceiling).

> **Parked threads that land here (both assessed/probed 2026-06-10/11, code on disk):**
> - **MSRK integrators** (`mcs2d/msrk.py`) — deploy at 4.3 as a further orthogonal
>   multiplier (production choice: a 2-step/3-stage RK4-2; re-measure ECF on the BSSN
>   gamma-driver spectrum). 9 tests green, CPU-validated.
> - **Compact (Padé/Lele) FD** (`mcs2d/schemes/compact.py`) — a **dense-GEMM Ozaki
>   target** (no 4.6× banded-zero waste; per-axis circulant aligns with E6 1D-operator
>   splits), so a candidate *derivative operator* for the Ozaki path. **Gated** on the
>   unresolved **non-periodic boundary** problem (currently periodic/FFT only; collides
>   with the SBP-averse outer-BC strategy) and on the fact that it **diverges from the
>   Dendro-GR explicit-FD oracle** — so NOT used in Phases 1–2. A Phase-4 research lever.

| Step | Title |
|---|---|
| 4.1 | Build the bit-reproducible Ozaki-II INT8 `compute_deriv` (stencil→GEMM→2:4 sparse TC). Fix the `pallas_ozaki` ~1600 s compile blocker (scan-roll the modulus loop) |
| 4.2 | Validate bit-reproducibility (CRT determinism) + FP64-equivalence on 2D/3D MCS; moduli/limb tuning for the BSSN field range; BFP48 compact-data |
| 4.3 | Deploy in the 3D BSSN RHS; measure the Amdahl-capped efficiency gain + BSSN bit-reproducibility vs FP64. Adopt a 2-step/3-stage MSRK integrator (re-measure ECF on the gamma-driver spectrum) |

## Phase 5 — Grid & frame architecture — ⬜
| Step | Title |
|---|---|
| 5.1 | Llama multipatch (central Cartesian + cubed-sphere wave-extraction shells) |
| 5.2 | Co-rotating frame (inner) + rotational shift terms |
| 5.3 | Spatial dual-frame (inner co-rotating + outer inertial transition) |
| 5.4 | Inter-patch coupling (penalty/SAT or interpolation) |
| 5.5 | **3D AMR port** — carry the complete 2D block-structured AMR (`AMR_PLAN.md`) to 3D; inherit ragged/calibrated caps, footprint prolongation, multi-block sync |
| 5.6 | Temporal hybrid (static inspiral graph → dynamic merger AMR) |
| 5.7 | Variable per-point FD metric weights (warped patches) · BFP48 canonical storage |

## Phase 6 — BBH physics, diagnostics & production scaling — ⬜
| Step | Title |
|---|---|
| 6.1 | Moving-puncture handling (χ/W, 1+log lapse, Γ-driver shift) |
| 6.2 | Bowen–York / Brill–Lindquist initial data + elliptic Hamiltonian-constraint solve (MPIR) |
| 6.3 | Puncture tracking + BBH-grade AMR |
| 6.4 | Apparent-horizon finder |
| 6.5 | Ψ₄ wave extraction + spin-weighted SH decomposition at the shells |
| 6.6 | Constraint monitors + ADM/horizon mass-spin; constraint-preserving outer BCs |
| 6.7 | Waveform validation vs the SXS catalog |
| 6.8 | Multi-GPU / multi-node JAX sharding (Marylou H200) + checkpoint/restart |

---

## Critical-path ordering
1. **Phase 1 (3D foundation)** — CPU-developable, gates everything; the BSSN RHS
   has nowhere to land without it.
2. **Phase 2 (BSSN correct)** — verbatim Dendro-GR port + validate; the 2.2 spill
   measurement motivates Phase 3.
3. **Phase 3 (codegen, PRIMARY)** — make the RHS compute-bound (the larger win).
4. **Phase 4 (Ozaki derivative, SECONDARY)** — Amdahl-capped efficiency +
   bit-reproducibility on real physics; the thesis result.
5. **Phases 5–6** — grid/frame, then BBH physics + science output, as far as
   validation holds.

> **If you change strategy, update the ⚑ banner here AND `HANDOFF.md` §2–4
> together** — the tracker and the narrative must never drift.
