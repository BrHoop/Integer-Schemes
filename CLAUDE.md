# CLAUDE.md — Integer-Schemes project reference

## What this project is
A **senior thesis** (advisor: Dr. David Neilsen, BYU) building a GPU/tensor-core
block-structured numerical-relativity pipeline: MCS toy model → BSSN binary-black-hole
at extreme mass ratios. The goal is INT8/Ozaki-II on 2:4 sparse tensor cores for
high-order stencils made compute-bound by temporal fusion — an open research gap.

Read the docs in this order:
1. `docs/HANDOFF.md` — strategic narrative, dead ends, working style
2. `docs/phases/README.md` — live execution tracker (what phase/step is active)
3. `docs/phases/phase_N_*/step_N.N_*.md` — the active step (just-in-time)
4. `docs/ROADMAP.md` — thematic backlog
5. `docs/ARCHITECTURE.md` — why the code is shaped as it is
6. `docs/OPTIMIZATION.md` — measured GPU baseline
7. `docs/OZAKI_GPU_NOTES.md` — Triton 0.9.2 gotchas + compile blocker
8. `docs/AMR_PLAN.md` — AMR phase detail (Phases 1–7, most complete)

---

## Repository layout
```
common/src/mcs_common/   — shared utilities (wave_state, bfp48, jax_config, ioxdmf)
2D/src/mcs2d/            — primary solver + AMR (all real work here)
  schemes/               — floating_point, ozaki, pallas_ozaki, fused_rhs_fp, fused_rhs_ozaki
  amr/                   — state.py, kernels.py, regrid.py, evolve.py
  main.py / benchmark.py / profile_amr.py / profile_ozaki.py / visualize.py
2D/tests/{unit,integration,regression}/
3D/src/mcs3d/            — bare (floating_point + unfused ozaki only; no AMR yet)
docs/                    — all project markdown
```

Install: `pip install -e ./common ./2D ./3D`

---

## Current state (as of 2026-06-11)
- **Phase 0, 1, 2 DONE; Phase 3 (codegen) ACTIVE.** Phase 0 (2D MCS foundation +
  regime), Phase 1 (3D foundation, H200 benchmark: 3D MCS **memory-bound, ~1% FP64
  peak**), Phase 2 (BSSN correctness — verbatim RHS bit-matches Dendro-GR, spill
  measured). Canonical spine: `docs/phases/README.md` (⚑ banner + Phase 0–6 tracker).
- **Phase 3.0 DONE (2026-06-11): production RHS variant locked to CAHD+SSL.**
  Switched the BSSN RHS from the no-CAHD `bssneqs_sympy_cse_wo_derivs.cpp` to
  **`bssneqs_SSL_HD_dxsq.cpp`** (Hamiltonian-constraint damping on chi + SSL
  lapse-locking) — what long runs need — and re-validated through the *same* Phase-2
  pipeline: oracle bit-compare vs Dendro-GR C++ = **1.93e-16**, 29/29 BSSN tests green
  (incl. slow apples evolutions with CAHD+SSL active). `_codegen.py` gained sqrt/exp
  mapping + a **scalar-param drift-guard**; the RHS is now **time-dependent** (SSL
  Gaussian ramp `exp(−t²/2σ²)`) and grid/step-dependent (CAHD `dx²/dt`), so
  `BSSNSolver`/`BSSNEvolution` thread `t` (per RK4 substage), `dt`, `dx_i`;
  `PhysicsParams` now *consumes* `cahd_c=0.06`/`ssl_h=0.6`/`ssl_sigma=20.0`
  (Dendro `q1.par.toml`). **Next: Step 3.1** (`phase_3_codegen/step_3.1_handstage.md`)
  — re-baseline the spill on CAHD+SSL (the 79-kernel/2536 B map was the no-CAHD
  variant), then probe whether XLA can be forced register-bounded (remat/barrier) or
  Pallas is required. See `[[bssn-codegen-staging]]`, `[[bssn-oracle-cpu-bitcompare]]`.
- **Local-machine OOM hazard (2026-06-11):** ~14 GB RAM / 4 GB swap; TWO concurrent
  Claude sessions compiling JAX at once swap-thrash → hard freeze. Stopgap
  `run_locked.sh` (flock + `systemd-run --user --scope -p MemoryMax`) serializes +
  caps heavy runs; user may retire it once the parallel C++ work wraps. See
  `[[feedback-local-machine-fragile]]`.
- **Phase 1 (3D foundation) complete (2026-06-11).** Ported `validate.py` +
  `benchmark.py` to 3D and built the test suite (`3D/src/mcs3d/validate.py`,
  `3D/src/mcs3d/benchmark.py`, `3D/tests/{unit,integration,validation}/`, 41 tests
  green). The audit's two open correctness questions resolved positively: the
  hand-derived 3D semi-discrete symbol matches the AD Jacobian to **2.4e-8**
  (decisive gate), and the 3D birefringent oracle is an exact solution → **6th-order**
  stencil convergence. Certificates: KO-limited 5th production order, RK4 floor 4th,
  −K/2 constraint damping, CFJ tachyon 1/0 (spectral), strong-hyperbolic (cond 4.81),
  divB at round-off. 3D RHS reduces to the validated 2D RHS on z-invariant data
  (≤1e-12). Deliverables in `docs/phases/phase_1_3d_foundation/step_1.1_results/`.
  **H200 benchmark done (2026-06-11):** 3D MCS is **memory-bound** (RHS ~0.25
  FLOP/byte cost-model, achieved ~354 GFLOP/s ≈ **1% of FP64 peak**), matching the
  2D regime — characterization only; the efficiency story is BSSN.
- **STRATEGIC PIVOT (updated 2026-06-11)** — see the ⚑ banner +
  `phase_2_bssn_correctness/bssn_port_plan.md`:
  - **MCS is compute-light → DRAM-bound** (Phase 0.2). It is now the **correctness /
    Ozaki-bit-reproducibility ground**, NOT where the speedup is shown.
  - **BSSN is the efficiency ground** (24 fields, ~24k FLOP/pt/step ≈ 62 FLOP/byte
    → compute-bound). **Decision: port BSSN's RHS from `~/Code/Dendro-GR` into
    Integer-Schemes, in 3D.** Octree rejected as a long-term home
    (wave-zone/dual-frame/cubed-sphere need shell grids).
  - **Port source = Dendro-GR, NOT DendroJax.** DendroJax's GR impl is **buggy** →
    dropped as oracle. Verbatim port = **transliterate** Dendro-GR's generated CSE
    `CodeGen/bssneqs_sympy_cse_wo_derivs.cpp` (flat SSA) → vectorized JAX
    (correct-by-construction; no SymPy toolchain). **Oracle = Dendro-GR C++**
    (bit-compare one RHS eval) **+ analytic apples tests.**
  - **Codegen is the PRIMARY lever; tensor cores SECONDARY.** BSSN RHS is ~70%
    point-wise algebra (not GEMM-able) → Amdahl caps the TC-derivative win at ~1.4×.
    The CSE codegen spills (**static dataflow: peak ~584 live vs 255 regs**, measured
    on Dendro-GR's `.cpp`) → memory-bound on its own intermediates. **GPU-optimal
    fused/staged RHS (Phase 3)** = the main win; **Ozaki derivative = Phase 4**
    (at its BSSN deployment). 2D shared-memory tiling is a retired toy-model artifact.
  - **Spill confirm rides on the JAX RHS (Phase 2.2)**: one `ptxas -v` compile of
    *our* RHS (XLA spills vs splits into HBM kernels). Dendro-GR is CPU C++ — its
    `g++` register behavior is irrelevant to the GPU `ptxas` budget; not a probe target.
  - **3D foundation first (Phase 1):** extend MCS to 3D + port the machinery
    (validate.py, Ozaki, benchmark), validate vs the 3D birefringent oracle, BEFORE
    the BSSN RHS.
- **Profiling:** see GPU workflow — Nsight banned, counters locked; use
  `profile_regime.py --smi` (nvidia-smi MEM%/SM%) + the tiling/intensity model
  `mcs2d/tiling_model.py`.
- **3D is bare:** no AMR, no Pallas kernels (Phase 1 builds the 3D foundation).
- **JAX pinned to 0.9.2** (local + Marylou); 0.8.1 was wrong (it never compiled Pallas on GPU)
- **Multistep-RK4 (MSRK) time integrators assessed (2026-06-10) — DONE.** The
  Sanches et al. previous-RHS-reuse methods (arXiv:2603.05763, the CarpetX/ET
  group's "RK v5" family) are implemented + measured on 2D MCS:
  `mcs2d/msrk.py` (3 methods + RK4-startup steppers + companion-matrix stability),
  `mcs2d/msrk_analysis.py` (driver), `tests/integration/test_msrk.py` (9 green).
  All three are clean 4th-order; coefficients verified vs the paper's ASR
  intercepts to 1e-6; linear-CFL limits confirmed by the nonlinear solver.
  **Which scheme wins depends on what limits dt:** far inside stability (MCS at
  CFL=0.05, ~16× margin) cost ∝ stages → RK4-3 (2 stages) cheapest; **near the
  stability limit (production BSSN) RK4-2(2)/(1) win and RK4-3 is the WORST —
  below plain RK4** (tiny ASR forces a small dt; matches the paper's BBH). So the
  production choice is a **2-step/3-stage method (RK4-2(2) or RK4-2(1))**, also
  memory-frugal (1 prior-RHS buffer vs RK4-3's 2 — matters for the memory-bound
  RHS). **Adopt in Phase 4.3**, after Phase 3 makes the RHS compute-bound, as an
  orthogonal multiplier on the Amdahl-capped (~1.4×) tensor-core derivative win.
  Re-measure ECF on the BSSN gamma-driver spectrum once Phases 1–2 exist (the
  `max_stable_dt`/symbol machinery transfers); the MCS regime is not the BSSN one.
- **Compact (Padé/Lele) FD probed (2026-06-11) — exploratory, interior validated.**
  Code: `mcs2d/schemes/compact.py` (`CompactDerivative(order=6|8, ko_order=)`, wired
  as `scheme="compact"` via params `compact_order`/`compact_ko_order`),
  `mcs2d/compact_stability.py` (derives 8th-order coeffs, symbol/CFL/spectral checks).
  Findings: compact is a *dense* operator `A^{-1}B` → a genuine dense GEMM, a BETTER
  Ozaki/tensor-core target than the banded explicit stencil (no 4.6x zero-waste,
  operator split amortized) — but its efficient form is the FULL dense GEMM, NOT a
  cheap truncated stencil (truncation needs band ~±47 to hold 6th order → dead).
  **compact-8 + 8th-order-KO** is the right pairing: ~75x lower error than
  explicit-6+KO6 in-solver (8th-order KO sign FLIPS: `-delta^8/256`). Stability:
  periodic compact is von-Neumann stable, modest CFL penalty (compact-8 0.751 vs
  explicit-6 0.828, confirmed nonlinearly), and spectrally CLEAN (no spurious +Re
  eigenvalues; only the physical CFJ tachyon, reproduced exactly). **Open = non-
  periodic boundaries** (see boundary note). Detail: memory `compact-fd-probe`.
- **Non-periodic boundary / outer-BC strategy (2026-06-11) — DECIDED, not built.**
  **DECISION: radiative BC for now** (adopt Dendro-GR's per-variable Bayley-Sommerfeld
  recipe ~verbatim + CAHD + KO + algebraic enforcement — proven, SBP-free, enables the
  bit-compare oracle; it's not the thesis contribution). **Hyperboloidal outer shell =
  backburner future accuracy upgrade** (reaches scri+, retires the artificial BC, fits
  the dual-frame/outer-shell endgame); CCE = bolt-on extraction upgrade. Default
  derivative = explicit FD (inherits boundary simplicity); compact = interior upgrade
  pending probes.
  User constraint: **avoid SBP/SAT at all costs** (order loss → more points). The
  SBP-free path = overlap (Llama-style) + high-order interpolation + KO at patch
  interfaces, + a closure-free **sponge** at the outer boundary (toward asymptotic
  reference, not zero). Sponge concerns largely deflate on coarse radially-stretched
  outer shells (cheap cells; extract inside; smooth thick ramp kills reflection).
  Constraint violation is the real one — and **Dendro/DendroJax BSSN already does
  Hamiltonian-constraint damping via CAHD** (`chi_rhs += C_CAHD*chi*(dxsq/dt)*H`,
  cahd_c~0.06; auto-stronger on coarse cells) so NO Z4c needed; the MOMENTUM
  constraint is the under-protected channel (only Gt + KO) → reference-matching
  matters most there. Compact boundaries are UNTESTED; the decisive cheap probe
  (finite-domain compact + naive closure + sponge → does dissipation kill +Re
  eigenvalues) is offered, not yet run. Grid choice (octree vs multipatch/dual-frame)
  GATES whether compact-boundary work is needed. Detail: memory
  `outer-bc-and-boundary-strategy`. NOTE: cubesphere (`~/Code/cubesphere`) is the
  ADVISOR's project; Integer-Schemes is the user's (their call).
- **3D BSSN GPU kernel architecture (2026-06-11) — design conclusions, not built.**
  The RHS splits into two near-separable halves with DIFFERENT bottlenecks on
  DIFFERENT on-chip pools:
  - **Derivatives (~30% FLOP):** the ONLY part reading neighbors → ALL the
    ghost-zone/halo/reuse cost is here; also the stencil-as-GEMM / tensor-core
    target. Each derivative is of a SINGLE field (cross-field coupling lives only
    in the algebra) → **streamable field-by-field**.
  - **Pointwise algebra (~70% FLOP):** Christoffel/Ricci/gauge; reads only
    same-point values (**138 distinct `grad_/grad2_` derivative inputs** + fields,
    `bssneqs_sympy_cse_wo_derivs.cpp`) → NO halo, compute-bound; the register /
    store-vs-recompute ground.
  - **"BSSN is compute-bound (62 FLOP/byte)" is a PERFECT-CACHE ceiling.** Naively
    realized in 3D it is MEMORY-bound: full z-reuse needs 2·NG+1=7 planes resident;
    7 × ~24 fp64 fields × N²·8B > 50 MB H200 L2 at **N≈190** → z-halo reads spill to
    HBM. Compute-bound REQUIRES explicit reuse capture; one-point-per-thread+L2 does
    NOT deliver it at realistic 3D resolution. (User suspected this; it holds.)
  - **Chosen design = field-streamed 2.5D explicit SMEM.** Stream fields ONE at a
    time through the derivative stage → SMEM holds one field's halo (BS=16→85 KB) →
    BS can be large → inter-tile halo redundancy `(BS+2NG)³/BS³` falls toward ~1.8×.
    **Temporal fusion REJECTED for 24-field 3D** (redundant-halo compute
    `(BS+2T·NG)³/BS³` explodes: BS=8,T=2 → 15.6×).
  - **Algebra intermediates CANNOT be SMEM tiles** — the Ricci stage's co-resident
    set (~80–100 values) × BS³ > 228 KB even at BS=8 → intermediates must live as
    **per-point REGISTER scalars** (accumulated as fields stream), a DIFFERENT pool
    than the loaded field halo (so they do NOT shrink the loadable tile).
  - **Two orthogonal pools:** SMEM/halo (controlled by field-streaming) vs registers
    (store/recompute/spill). **CORRECTED (2026-06-13, `docs/algebra.md` §2): the
    register-pressure floor is a ~128-temp FIRST-ORDER tensor trunk** (inverse metric +
    ~27 Christoffels + gauge/χ web — temps feeding ≥12 of the 24 outputs), **NOT Ricci**
    — only 28/826 temps are Ricci-family and the trunk contains ZERO of them (the earlier
    "Ricci sums all 36 grad2(gt) → that's the floor" claim is refuted; the CSE/scheduler
    already serializes the Ricci contraction). Recompute-everything is dead (~623k–709k raw
    ops, ~150× compute); store-everything spills (CSE algebra = 887 temps). Optimum =
    **selective materialization** of the first-order trunk (→ SMEM in fp64, or registers in
    fp32) + recompute the ~650-temp bulk, per output-fanout (not the old fan-out×cone).

> **Exploratory code artifacts (2026-06-10/11), all UNCOMMITTED + CPU-validated,
> easy to delete:** `mcs2d/msrk.py`, `mcs2d/msrk_analysis.py`,
> `tests/integration/test_msrk.py` (9 green), `mcs2d/schemes/compact.py`,
> `mcs2d/compact_stability.py`, a `scheme="compact"` branch in `main.py`, and
> `step_0.1_results/msrk_assessment.png`. Run: `python -m mcs2d.msrk_analysis`,
> `python -m mcs2d.compact_stability`. These probe time-integration (MSRK) and
> spatial-discretization (compact FD) alternatives for the eventual 3D/BSSN path;
> none are on the critical path yet. Full context in the three memory files above.

---

## Critical constraints — do not violate

1. **Shape-stable JIT:** array shapes never change mid-run. Regrid flips `active` bits only.
   `AMRTopologyArrays` (runtime args) carry topology changes without recompile.
2. **No fp64 or int32 `dot` in Pallas kernels** — hangs the Triton compiler.
   Only int8/fp16/fp32 `dot` is safe. See `OZAKI_GPU_NOTES.md §2`.
3. **No `dynamic_slice`, `gather`, `scatter`, `dynamic_update_slice` in Pallas 0.9.2.**
   Crop via one-hot select-reduce (`_mid`/`_onehot` in `pallas_ozaki.py`).
4. **KO dissipation sign is load-bearing.** The negated form is anti-dissipative and destroys runs.
5. **Punctures move** (moving-puncture method). Never say "frozen" or "zeroed."
6. **Co-rotating frame can't reach the wave zone** (superluminal shift beyond r≈c/Ω).
   Spatial dual-frame required.
7. **Convergence claims at the puncture are smoothness-capped**, not 6th-order.
   Frame as "high-order in wave zone, wavelet-controlled at puncture."
8. **Ozaki variant is Ozaki-II (CRT/INT8) only.** "FP16 Ozaki" is an error.

---

## GPU/compute workflow
- **I cannot run GPU code locally.** Local runs CPU only; Pallas/Triton tests are
  `interpret` mode (math only — does NOT test compilation).
- **GPU = Marylou (H200, Hopper).** Push via `./sync.sh push` (now one shared SSH
  connection → a single Duo 2FA per push/pull). Pattern: prep → user runs → pastes → iterate.
- Local JAX 0.9.2 (Python 3.14). Marylou: Python 3.12, JAX 0.9.2. Persistent compile
  cache at `~/.jax_cache`.
- **Pallas compile wall:** `pallas_ozaki` with `trunc=2` takes ~1600 s on GPU (unrolled).
  Fix: roll modulus loop with `lax.scan`. Not yet done. Use persistent cache as mitigation.
- **Nsight is BANNED on the cluster** (policy) + GPU perf counters locked (`ERR_NVGPUCTRPERM`),
  so `ncu`/`nsys --gpu-metrics` give nothing. Allowed/permission-free profiling only:
  `mcs2d/profile_regime.py` (`--smi` = nvidia-smi SM%/MEM%, `--jax-trace` = Perfetto op
  breakdown), `nvtop` (live), and a counter-free `nsys` *trace* (kernel/graph timeline → parse
  its SQLite). The DRAM-vs-compute *flip* shows as MEM%↓/SM%↑ — no counters needed.
- **FP64 baseline regime (Step 1.2): DRAM-bandwidth-bound.** XLA's `fused_floating_point` step is
  99% GPU-busy, 100% occupancy, ~4% of FP64 peak, and uses **0 B shared memory** — it does the
  6th-order stencil as global-memory `dynamic_slice` passes (~15 element-wise kernels, no on-chip
  reuse). ⇒ Step 1.3 = tile the stencil into shared memory + temporally fuse the RK stages.

---

## Shelved / dead ends
- **Ozaki INT8 compute for the FD RHS (sparse/memory-bound path):** SHELVED.
  FD is memory-bound; Ozaki accelerates dense GEMM. Casting banded stencil as dense
  GEMM wastes ~4.6× on zeros. DO NOT try to "finish" this for the FD path.
- **Spectral / SpEC-style:** rejected (too large a pivot; SpEC accuracy is 20+ years
  of excision/dual-frame/control apparatus, not reproducible from papers).
- **JAX ≥ 0.9 for pallas_ozaki (was 0.8.1):** 0.8.1 never actually compiled on GPU.
  0.9.2 is correct. The break was in 0.9 tightening Triton lowering.

---

## Testing strategy (2D)
- `unit/` — per-kernel correctness, fixed shape, < 5 s each
- `integration/` — multi-step evolution vs analytic oracle or reference
- `regression/` — no-recompile guard, bit-identical AMR, long-term stability, convergence
- `validation/` — Step 1.1 baseline guards (CPU-safe, ~5 min): spatial convergence,
  semi-discrete + RK4 spectrum, dispersion, KO kernel, CFJ, strong hyperbolicity,
  boundary energy-passivity, constraint-damping. Backed by `mcs2d/validate.py`.

Run: `pytest 2D/tests/` or `./2D/tests/run.sh [unit|integration|regression|validation|fast|amr]`

Key test invariants:
- **No mid-evolution recompilation** (`test_no_recompile.py`) — machine-checked
- **Self-convergence must use exact integer step counts** (non-integer T/dt manufactures
  spurious error that swamps truncation error)
- **Stability targets in light-crossing times**, not fixed step counts

### Step 1.1 validation harness (`mcs2d/validate.py`) — findings to remember
- **One symbol, many views.** All spectral diagnostics post-process the linearized
  semi-discrete symbol `_symbol(kx,ky)` (10×10), cross-checked to the exact AD Jacobian
  (`_build_jacobian`) to ~1e-8. `python -m mcs2d.validate` writes research-meeting figures.
- **Linearize at Pi=Pi₀, not u=0.** MCS is quadratic (Pi·B, Ez·∂ξ); at u=0 the CS coupling
  vanishes (no birefringence, no CFJ). The Pi₀ background makes `cs·2L·Pi₀·B` a linear mass
  term with coefficient exactly m_cs — required to see the real physics.
- **Constraint damping rate is K/2** (underdamped Gundlach pair), not K. Tests assert −K/2.
- **Convergence is a three-line story** (measure at FIXED physical time, not fixed step
  count — fixed steps make T∝dx and inflate the apparent order by ~1; the old "6.5" was this
  artifact). Stencil (KO off) = **6th**; production (default 6th-order KO σ=0.05, CFL=0.05) =
  **~5th** because σ/dx·6th-difference KO is an **O(h⁵)** dissipation (standard 6th-order KO
  is "rated" for 4th-order FD; 2N+2 rule → 6th-order FD wants 8th-order KO, deferred to 1.3);
  RK4 temporal floor = **4th** but only binds at CFL≳0.4 (never at the operating 0.05).
  `TestOrderSeparation` certifies stencil 6 + temporal 4 in isolation.
- **x↔y symmetry flips Ez, Bz** (out-of-plane), swaps in-plane (Ex↔Ey, Bx↔By); scalars fixed.
  Derived from the RHS — the equal-kx/ky oracle would otherwise mask an axis bug.
- **`directional` BC was REMOVED** (Step 1.1) after the validation suite found it unstable
  for general data (injected energy, blew up ~1e9–1e15 regardless of v or KO). Only two BCs
  remain: periodic (exact, conservative) and sommerfeld (passive, ~71% absorbing).
- **Profiling is GPU-only** (`profile_baseline` refuses on CPU); everything else runs on CPU.

---

## AMR status (block-structured Berger–Oliger, 2D)
- Phases 1–5 complete: static foundation, regridding, sub-cycling (Hermite), GPU opt,
  multi-block within-level sync, moving-feature validation
- **Rolled sub-cycling** with `lax.scan`: compile O(2^L)→O(L); ~9 s for L=2, ~34 s for L=6
- **Calibrated caps** (`make_calibrated_root_state`): ~10.5× throughput win; now default
- AMR accuracy floor: **spatial**, not temporal (2nd-order restriction at coarse-fine
  boundary; fix = higher-order restriction, NOT a better integrator)
- Compaction (4.6): **retired** — dynamic profile showed fragmentation=1.0 (self-compacts)
- **Phase 5 GPU profile still TODO** (item 5.5 in AMR_PLAN.md)
- 3D port: Phase 6, not started

---

## Key physics
- MCS fields: Ex Ey Ez Bx By Bz ξ Πξ Ψ Φ (NF=10)
- Analytic test: birefringent circularly-polarized plane wave `Ez = E0 sin(kx·x + ky·y − ωt)`
- **CFJ tachyon (physics, not a bug):** ω²=k²−m_cs·k < 0 when k < m_cs=2Λ; unstable for large Λ
  Default Λ=0.4 (stable regime, k≈0.89 > m_cs=0.8). Λ=2 blows up intentionally.
- Constraint damping: K1/K2 (κ₁,κ₂) damp divE and divB violations

---

## Working style preferences
- **Brutally honest assessments.** User is a domain expert; hand-waving gets caught.
- **Docs are just-in-time:** do NOT bulk-create stub docs for future steps.
  Only write docs for the active step. User has explicitly corrected this behavior.
- **Hold a numerical-relativist's bar** (Neilsen reviews the thesis).
- Commit/push only when asked. Branch first if on `main`. Co-author trailer required.
- GPU validation requires user action (2FA push); plan around it explicitly.

---

## Key references
- *Do We Need Tensor Cores for Stencil Computations?* arXiv:2603.00477 (2026)
- SPIDER/SPTCStencil: 2:4 sparse-TC stencils via strided swap. arXiv:2506.22035
- Ozaki Scheme II / GEMMul8: INT8 fp64-GEMM emulation, bit-reproducible. arXiv:2504.08009
- Fernando, Neilsen et al — Dendro-GR BBH with wavelet AMR. PRD 107, 064035 (2023)
- GR-Athena++ — vertex-centred oct-tree puncture evolutions. ApJS (2021)
