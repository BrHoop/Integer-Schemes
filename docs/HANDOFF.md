# HANDOFF — context a future agent needs that isn't in the other docs

Read-me-first orientation: the *narrative, strategic, and operational* context the
technical docs don't carry. It cross-references rather than repeats them.

### Read in this order
1. **This file** — orientation, strategy, the load-bearing findings, constraints.
2. `phases/README.md` — the ⚑ **Strategy update** banner + the live tracker.
3. `phases/phase_2_bssn_correctness/phase_2_plan.md` + `bssn_port_plan.md` — the active direction (BSSN port).
4. `phases/phase_1_3d_foundation/phase_1_plan.md` — the last completed phase (3D foundation, validated + benchmarked).
5. `ROADMAP.md`, `ARCHITECTURE.md`, `OPTIMIZATION.md`, `OZAKI_GPU_NOTES.md`, `AMR_PLAN.md`.
6. Auto-memory (`MEMORY.md` index) — user, preferences, known bugs, project goal.

---

## 1. What this project is
A **senior thesis** (advisor: Dr. David Neilsen, BYU) building a GPU/tensor-core
numerical-relativity pipeline, MCS toy model → **BSSN binary-black-hole**, aimed
at extreme mass ratios. Neilsen co-authors **Dendro-GR** (octree + wavelet AMR).
This project is the **block-structured / tensor-core alternative**, with its own
grid endgame (cubed-sphere/Llama multipatch + spatial dual-frame). Hold a
numerical-relativist's bar — Neilsen reviews it. Production GPU = **Marylou H200**.

## 2. Strategic narrative — earned through this work, do not re-litigate
The arc up to mid-2026 (Ozaki-for-FD shelved → tensor-core stencils via Ozaki-II
on sparse TC) is in `ROADMAP.md`/memory. **What changed in June 2026 (this
arc) is the important part:**

1. **Step 1.1 (done):** built the baseline validation + characterization harness
   for 2D MCS (`mcs2d/validate.py` + `tests/validation/`). Physics is correct,
   well-posed, stable, convergent. Found+removed an unstable `directional` BC.
2. **Step 1.2 (done):** measured the regime on the H200. **The FD RHS is
   DRAM-bound (~3.2 FLOP/byte).** XLA's "fused" FP64 path is *not* a Pallas kernel
   (renamed `fused_rhs_pallas.py`→`fused_rhs_fp.py` to stop the confusion); it
   streams the stencil through HBM as ~15 element-wise `dynamic_slice` kernels with
   **0 bytes shared memory** — running at ~5–6% of peak HBM (inefficient, not
   saturated). `pallas_fp` (hand FP64 Pallas) was **15× slower than XLA → removed.**
3. **The intensity model (`mcs2d/tiling_model.py`, Step 1.3-T0)** then reframed the
   whole thesis. Key results:
   - **MCS is too compute-light to ever show the payoff.** It's DRAM-bound; tensor
     cores can't help. → MCS is now the **cheap correctness / Ozaki-bit-
     reproducibility ground**, NOT the efficiency story.
   - **BSSN is the efficiency ground** (24 fields, ~24k FLOP/pt/step ≈ 62 FLOP/byte
     → compute-bound). The payoff can only be demonstrated there, in 3D.
   - **2D shared-memory tiling is a toy-model artifact** — doesn't transfer to 3D
     (24 fields → tiles too big, 64× redundant) and isn't needed (compute-bound).
     **Do NOT build out the 2D tile-and-fuse kernel.**
   - **The deliverable is the Ozaki tensor-core `compute_deriv` operator**, built
     and validated on MCS, deployed in BSSN.
4. **The codegen finding — the current crux.** BSSN's RHS is ~70% point-wise
   nonlinear algebra (Christoffel/Ricci/gauge — *not* GEMM-able) and ~30%
   derivatives (stencil). So Amdahl caps the tensor-core derivative win at ~1.4×.
   **And static dataflow analysis of Dendro-GR's CSE-generated RHS
   (`bssneqs_sympy_cse_wo_derivs.cpp`) found a peak ~584 (median 378) simultaneously-
   live temporaries vs the 255-register budget → forced spilling → the compute-
   heavy RHS is very likely MEMORY-bound on its own intermediates**, partly
   invalidating the model's "BSSN is compute-bound" optimism. So **the primary
   lever is GPU-optimal fused codegen** (a register-bounded, recompute-balanced
   generator that keeps intermediates on-chip), and **tensor cores are secondary**.
   The honest thesis is now: *"high-order GR RHS is memory-bound on GPUs because
   symbolic-CSE codegen overruns the register file; a fusion-first generator makes
   it compute-bound, and tensor cores then accelerate the stencil portion."*
   **→ Phase 3.1 (2026-06-12) resolved the emission target to Pallas, see §4.**

**Net direction:** 3D-MCS foundation → port BSSN into Integer-Schemes (3D) and
get it *correct* → make it compute-bound via better codegen (**primary**) → add
the Ozaki tensor-core derivative (**secondary, at its deployment**). Linear spine
in `phases/README.md` (the ⚑ banner + Phase 0–6 tracker) and `bssn_port_plan.md`.

## 3. The decision: port the BSSN RHS from Dendro-GR (NOT DendroJax)
**Updated 2026-06-11.** The earlier plan used `~/Code/dendrojax` (a JAX BSSN) as a
bit-for-bit porting oracle. **DendroJax's GR implementation turned out to be buggy
→ disqualified as a correctness oracle** (validating a port against a buggy
reference is worse than no oracle). New decision (user-driven, sound):
- **Port the BSSN *RHS* from `~/Code/Dendro-GR`** — the group's published, validated
  code. The *verbatim* step **transliterates** the generated CSE
  `CodeGen/bssneqs_sympy_cse_wo_derivs.cpp` (942 lines, derivatives-as-inputs, flat
  `const double DENDRO_N = …;` SSA) → vectorized JAX. Correct-by-construction from
  trusted output; **no SymPy toolchain, no DendroJax.** The *staged regen* (the
  codegen contribution) later drives Dendro-GR's SymPy front-end
  (`bssn.py`/`dendro.py`/`bssn_nx.py`) — replacing only the C++ back-end emitter.
- **Oracle = Dendro-GR C++** (identical ID, one RHS eval, compare to round-off) **+
  the analytic apples tests** (gauge wave, robust stability, constraint convergence
  — reference-free). DendroJax is dropped entirely; octree/wavelets/ghost-exchange
  left behind (octree is a dead end for the shell-grid endgame).
- **Spine:** Phase 1 **3D-MCS foundation first** → Phase 2 BSSN state/gauge +
  verbatim RHS + validate (the spill measurement rides here) → Phase 3 GPU-optimal
  codegen (**primary**) → Phase 4 Ozaki derivative (**secondary**). Full plan in
  `bssn_port_plan.md`.

> **The register-spill confirmation rides on the JAX RHS (Phase 2.2), not Dendro-GR.**
> Register allocation is architecture-specific: Dendro-GR is **CPU C++** — `g++`'s
> register behavior says nothing about the NVIDIA 255-reg/`ptxas` budget, and its
> hand-staged CUDA is the *already-fixed* version. The 584-live number is a
> *dataflow* property (architecture-independent) and already answers the GPU
> question; the *hardware* confirm is one `ptxas -v` compile
> (`--xla_gpu_asm_extra_flags=-v`) of **our** JAX RHS, telling us the fork that
> picks the codegen fix (XLA **spills to local memory** vs **splits into many
> kernels with grid-sized HBM intermediates**).

## 4. Where we are RIGHT NOW (the live edge)
- **Phases 0–2 ✅, Phase 3 ACTIVE. Step 3.1 DONE (2026-06-12) → emission target
  resolved to PALLAS.** The controllability probe (`_bssn_rhs_staged.py`, 69
  `optimization_barrier` cuts; H200 A/B vs the CAHD+SSL re-baseline 97 kernels/1104 B):
  barriers **eliminate the spill** (1104 B → 0, regs 255→254) — XLA *is* register-
  controllable — **but only by *splitting*** (97 → 107 kernels; the barrier lever
  trades spill for HBM-round-tripping kernels). **Crucially, `remat` is NOT a second
  lever here:** `jax.checkpoint` only acts during autodiff, and the RHS runs forward
  (RK4, no grad) → no-op (`prevent_cse=True` degenerates to a barrier). So XLA's *only*
  forward knob is splitting, which cannot give few-kernels-AND-register-bound — and the
  §-below selective-**recompute** design (shrink the live set by rebuilding cheap
  leaves) is **inexpressible in XLA forward eval**. → The register-resident, recompute-
  where-cheap kernel must be authored where we own the register file: **Pallas.** The
  staged-XLA RHS stays as a 0-spill **regression baseline**, not the optimization
  target. Restructured Phase 3 (Pallas-committed): `phases/phase_3_codegen/phase_3_plan.md`
  §3 (3.2a regime-profiler + confirm staged-XLA still HBM-bound → 3.2b materialize/
  **recompute**-set generator → 3.2c **algebra-only** Pallas kernel, derivs-as-inputs →
  3.3 regime flip on the Pallas kernel). Heavier loop ahead: local Pallas is
  interpret-mode only (math, not compilation) + the ~1600 s compile wall → every real
  spill check is a Marylou 2FA run. See `[[bssn-codegen-staging]]`.
- **Step 3.2a DONE (2026-06-12) — BSSN still memory-bound; the roofline found TWO
  walls.** Built `bssn3d/profile_regime.py` (`--smi`); H200 N=128³: staged-XLA **SM
  100% / MEM 98%**, verbatim MEM 97% → both **memory-bound despite the staged despill**
  (SM% = warp-resident, not throughput; MEM% is the DRAM tell). Roofline: (1) inter-
  kernel intermediate round-trips (the 107 fragments) — the algebra-only Pallas kernel
  (3.2c) fixes this, ~3× cut; (2) the **138 derivative arrays the `_wo_derivs` design
  materializes to HBM** (~1100 B/pt mandatory read) — an algebra-only kernel CANNOT
  remove it → caps intensity at ~3–6 FLOP/B ≤ H200 FP64 balance (~7) → still mem-bound.
  **"62 FLOP/byte" was always a perfect-cache figure assuming on-chip derivatives.** So
  the regime flip needs **derivative fusion** (new Step 3.2d: field-streamed 2.5D-SMEM,
  138 arrays never materialized), not algebra-only. Decision (user): algebra-only first
  to de-risk register control, then fuse — sequential. The Ozaki/tensor-core derivative
  stays Phase 4; 3.2d is plain on-chip FD fusion.
- Phase 1.1 + 1.2 **done**; docs + tracker restructured.
- **Dendro CodeGen study DONE (2026-06-11)** — `~/Code/Dendro-GR/CodeGen` fully read.
  Detailed conclusions live in **CLAUDE.md** ("3D BSSN GPU kernel architecture" entry);
  the narrative:
  - **Three generators, none is the thing we want.** `generate_cpu` (one giant SymPy
    `cse` over all outputs) is what **ships in `rhs.cpp`** and causes the spill — its
    CSE temps are **never freed**. `generate_separate` bounds pressure but **recomputes**
    all shared tensors per field. `generate_code_nx` (+ `visit_node`/`store_node`/
    `evict_node`) is the *right idea* — DAG + liveness-based slot reuse — but takes
    **one arbitrary topological sort** (schedule-selection sketched, never implemented)
    and emits **C++ scalars**. `bssn_manually_staged.py` is the tell: Dendro hit the
    register wall and solved it **by hand-staging** (cut at tensor boundaries,
    `MAX_TEMP_VARS=64`). The clean SymPy front-end (`dendro.py`: `set_metric →
    Christoffel → Ricci → lie`) is **reusable; only the back-end emitter is C++-bound**.
  - **MEASURED the register pressure** (static liveness on
    `bssneqs_sympy_cse_wo_derivs.cpp`, derivatives-as-inputs): the CSE algebra **peaks
    at 584 simultaneously-live temps, median 378, >255 across 69% of the kernel**, plus
    **138 distinct derivative inputs** → ~700+ live values vs the 255-register file.
    Pervasive spill → **the 70% "compute-bound" algebra is memory-bound on its own
    intermediates** (sharper than the old "~480"; confirms the regime). A naive
    min-liveness **reschedule didn't help (639)** → the width is **structural** (broad
    plateau; Ricci sums all 36 `grad2(gt)`), not a reorderable spike.
  - **CONCLUSION — staging is the design** (user agreed, done deliberating). Store the
    **~69 high-fan-out tensor-hierarchy temps** (inverse metric, Christoffels, Ricci,
    CalGt) as **per-point register scalars**, recompute/inline the ~800 cheap leaves →
    **~133 live → fits under 255 → genuinely compute-bound**. Both extremes dead:
    recompute-everything = 623k raw ops (~150×); store-everything (CSE) = 584-live
    spill. Thesis codegen contribution = **automate the cut-point/store-set selection**
    (DAG min-cut / articulation) and **emit the staged schedule** — the unbuilt middle
    between Dendro's generators. **Emission target = Pallas (resolved Phase 3.1, see
    the live-edge entry below); NOT XLA `remat`.**
  - **Kernel architecture (separable, two on-chip pools):** derivatives (~30% FLOP,
    the only halo/reuse cost; tensor-core target) vs pointwise algebra (~70%, the
    register/staging ground). **Field-streamed 2.5D explicit SMEM** (stream fields one
    at a time → SMEM holds one field's halo → big BS → low halo redundancy);
    **temporal fusion rejected** for 24-field 3D. "BSSN compute-bound" is a
    perfect-cache ceiling — **naive 3D is memory-bound** (7-plane × 24-field z-reuse
    window > 50 MB L2 past N≈190). See CLAUDE.md for the full entry.
- **Immediate move (agreed): the 3D-MCS foundation (Phase 1)** — carry the machinery
  to 3D (CPU-developable; the BSSN RHS has nowhere to land without it). Do NOT start
  the BSSN RHS port before the 3D foundation exists.
- **The spill confirmation is folded into Phase 2.2**, not a standalone probe: once
  the verbatim Dendro-GR RHS is transliterated to JAX, one `ptxas -v` compile of *it*
  on the H200 gives the register/spill number (does XLA spill vs split into HBM-backed
  kernels). The earlier idea of probing DendroJax (or compiling Dendro-GR's CPU C++)
  is dead — DendroJax is dropped, and CPU `g++` register behavior is irrelevant to the
  GPU `ptxas` budget.

## 5. Operational constraints — these have burned us
- **I (the agent) cannot run on a GPU.** Local is CPU only; Pallas/Triton tests run
  `interpret` (math only, skips Triton lowering). GPU bugs surface only on the GPU.
- **GPU = Marylou H200**, via `./sync.sh push` → **interactive Duo 2FA the user
  performs.** `sync.sh` now shares ONE SSH connection (single 2FA per push/pull);
  `pull results` / `pull traces` retrieve deliverables. Workflow: prep → user runs
  → pastes → iterate. Don't promise GPU validation I can't do.
- **Nsight (`ncu`/`nsys`) is BANNED on the cluster** + GPU perf counters locked
  (`ERR_NVGPUCTRPERM`). Permission-free profiling only: **`mcs2d/profile_regime.py`**
  (`--smi` = nvidia-smi SM%/MEM%, `--jax-trace` = Perfetto op breakdown), `nvtop`,
  and a counter-free `nsys` *trace* (parse its SQLite — that's how we got the
  99%-busy / 0-shared-mem regime facts). DRAM-vs-compute shows as MEM%↓/SM%↑.
- **JAX pinned 0.9.2** (local Python 3.14; Marylou Python 3.12). Compile cache
  `~/.jax_cache`. **No fp64/int32 `dot` in Pallas** (hangs Triton); see `OZAKI_GPU_NOTES.md`.
- **Git:** commit/push only when asked; branch first if on `main` (we are on `main`,
  nothing committed this arc). Co-author trailer required.

## 6. Dead ends / corrections (don't repeat / don't regress)
- **Ozaki *compute* for the FD RHS:** shelved (FD is memory-bound; Ozaki is for
  dense compute-bound GEMM). Don't "finish" it for FD.
- **2D shared-memory tiling / temporal fusion as a deliverable:** toy-model
  artifact (model showed it doesn't transfer to 3D and isn't needed). Don't build it.
- **`pallas_fp` removed** (15× slower than XLA; no FP64 Pallas win exists).
  **`directional` BC removed** (unstable, injected energy). **`fused_rhs_pallas` →
  `fused_rhs_fp`** (it was never Pallas — pure XLA).
- **The cost model can't measure DRAM** on a fused kernel (counts on-chip/halo
  traffic); the `hlo_*` benchmark metrics are upper bounds, not DRAM. Real DRAM
  needs counters (banned) → use `profile_regime --smi`.
- **Physics to get right** (Neilsen pounces): punctures MOVE (not frozen);
  co-rotating frame can't reach the wave zone (dual-frame required); puncture
  convergence is smoothness-capped (frame as "high-order in wave zone, wavelet-
  controlled at puncture"); CFJ tachyon at large Λ is physics not a bug; KO sign is
  load-bearing; constraint damping rate is K/2 (underdamped Gundlach). Ozaki = **II
  (CRT/INT8) only**; "FP16 Ozaki" is an error.

## 7. Working with this user
- **Be brutally honest, not encouraging-by-default.** They're an NR domain expert,
  know their group's code, and catch hand-waving. A rigorous negative result beats
  optimism — the whole codegen/regime reframe came from *them* pushing back, twice.
- **Docs are just-in-time** — only the active step gets a doc; no bulk stubs.
- Collaborative/iterative — they refine and course-correct; don't over-commit a
  plan before they react. They handle the GPU runs + the NR physics; the agent
  handles implementation velocity + analysis/tooling.

## 8. Current code state (Integer-Schemes)
- **2D MCS solver** (System IV, NF=10): `2D/src/mcs2d/main.py`. Schemes now:
  `floating_point`, `fused_floating_point` (XLA-fused FP64 baseline — `fused_rhs_fp.py`),
  `ozaki`, `fused_ozaki`, `pallas_ozaki`. (`pallas_fp` removed.)
- **Validation:** `mcs2d/validate.py` (research-figure harness: oracle, semi-discrete
  symbol/Jacobian, convergence, spectrum, stability) + `tests/validation/` (4 files,
  ~31 tests: convergence/order-separation, spectrum/hyperbolicity, boundaries,
  constraint-damping). All green on CPU. Deliverables in
  `docs/phases/phase_0_2d_foundation/step_0.1_results/`.
- **Profiling/analysis tools (new):** `mcs2d/benchmark.py` (roofline + `hlo_*`
  cost-model metrics + `--replot`), `mcs2d/profile_regime.py` (`--smi`/`--jax-trace`,
  Nsight-banned-aware), `mcs2d/tiling_model.py` (the intensity/feasibility model —
  run `--problem both`; the source of the regime/Amdahl/compact-data findings).
- **AMR** (2D block-structured Berger–Oliger) complete + shape-stable; not the focus.
- **3D is bare** (`3D/` has only floating_point + unfused ozaki). Phase 1 builds
  the 3D foundation.
- **Reference repos (read-only):** `~/Code/Dendro-GR` (the porting + validation
  oracle — `CodeGen/bssneqs_sympy_cse_wo_derivs.cpp` to transliterate, `CodeGen/bssn.py`
  /`dendro.py`/`bssn_nx.py` SymPy front-end for the staged regen), `~/Code/cubesphere`
  (their grid work). `~/Code/dendrojax` is **dropped** (buggy GR impl).

## 9. Open questions / pending
- ~~**Read the Dendro CodeGen** staging/scheduling — can we re-emit a register-bounded,
  fused RHS?~~ **DONE (2026-06-11).** Yes, via **staging** (store ~69 tensor-hierarchy
  temps as register scalars + recompute the leaves → ~133 live, fits 255). Measured the
  CSE-everything spill directly: **peak 584 / median 378 live, >255 across 69%** of the
  algebra, + 138 derivative inputs. Both extremes dead (recompute-all 623k ops ~150×;
  store-all 584-spill). Reuse Dendro's SymPy front-end; the contribution is an
  **automatic cut-point/store-set generator emitting JAX/Pallas**. Full write-up in §4
  + CLAUDE.md ("3D BSSN GPU kernel architecture").
- **Confirm the spill** on the H200 — folded into **Phase 2.2**: one `ptxas -v` compile
  (`--xla_gpu_asm_extra_flags=-v`) of the *ported JAX* RHS verifies whether XLA spills to
  local memory or splits into HBM-backed kernels (and whether it realizes the staged
  schedule in Phase 3). NOT a probe of DendroJax or Dendro-GR's CPU C++. (user GPU)
- The Ozaki **moduli/limb count** for FP64-equivalence in the BSSN field range, and
  the **mixed-precision Ozaki** ("more limbs in the middle") — the model says
  compact data + fewer Ozaki products is the lever that moves the net win from ~1.6×
  toward ~5–7×.
- **BFP48** (6-byte, already in `common/.../bfp48.py`) as the compact-data win.
- ~~RK "v5" (group's 3-stage, uses previous-step data) — assess when the user has it.~~
  **DONE (2026-06-10):** assessed the Sanches et al. multistep-RK4 methods
  (arXiv:2603.05763 — the CarpetX/ET group's previous-RHS-reuse integrators) on
  2D MCS. Code: `2D/src/mcs2d/msrk.py` (+ `msrk_analysis.py`, `tests/integration/
  test_msrk.py`, 9 green; figure in `step_0.1_results/msrk_assessment.png`).
  **Result:** all three are clean 4th-order; coefficients verified vs the paper's
  ASR intercepts to 1e-6; linear-CFL limits confirmed by the nonlinear solver.
  **Regime rule = which method wins depends on what limits dt:** far inside
  stability (MCS at CFL=0.05, ~16× margin) cost ∝ stages → RK4-3 (2 stages) is
  cheapest; near the stability limit (production BSSN, CFL pushed for speed)
  **RK4-2(2)/(1) win and RK4-3 is the *worst* — below plain RK4** (its tiny ASR
  forces a small dt that eats the 2-stage saving; matches the paper's BBH). So
  the production choice is a **2-step/3-stage method (RK4-2(2) or RK4-2(1))** —
  also memory-frugal (stores 1 prior-RHS buffer vs RK4-3's 2). Adopt in **Phase
  4.3**, after Phase 3 makes the RHS compute-bound, as an orthogonal multiplier on the
  Amdahl-capped tensor-core derivative win. Re-measure ECF on the BSSN
  (gamma-driver) spectrum once Phases 1–2 exist — that sets the real regime; the
  `max_stable_dt`/symbol machinery transfers directly.

---

*If you change strategy, update §2–4 here AND the ⚑ banner in `phases/README.md`
together — the narrative and the tracker must never drift.*
