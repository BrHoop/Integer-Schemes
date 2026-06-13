# ROADMAP — Integer/Tensor-Core Numerical Relativity (MCS → BSSN BBH)

Master implementation checklist. The end goal is an extreme-mass-ratio BBH
solver on GPU tensor cores, built up from the Maxwell–Chern–Simons (MCS) toy
model. Tags:

- **[DONE]** in code · **[EXISTS]** partial/scaffold · **[TODO]** to build ·
  **[SHELVED]** decided against · **[GATED]** only if a prerequisite is taken

Companion docs: `ARCHITECTURE.md` (the *why*), `OPTIMIZATION.md` (measured GPU
baseline + efficiency analysis), `OZAKI_GPU_NOTES.md` (Triton 0.9.2 / compile
record), `AMR_PLAN.md` (AMR phase detail).

---

## A. Current state (baseline)
- **[DONE]** JAX pinned 0.9.2; both Pallas kernels rewritten for 0.9.2 Triton
  (ref-indexing, one-hot select-reduce crop, per-channel stores; no
  slice/split/scatter); int8 hardcoded; BFP48 mantissa input; int32 limb
  residues. Math validated in interpret; 10/10 pallas tests.
- **[DONE]** Tooling: `profile_ozaki.py` (`--diagnose/--quick/--no-ko/--mods-ext`,
  `profile_trunc` hook); `profile_amr.py`.
- **[EXISTS]** Block-structured Berger–Oliger AMR: shape-stable JIT, ragged
  per-level caps + calibrated auto-grow, rolled sub-cycling (O(L) compile),
  proper nesting + buffer dilation, gradient indicator + hysteresis, moving-box
  scaffold, pluggable RHS (`_make_kernel_fn`), persistent compile cache,
  temporal-fusion scaffold (`make_fused_ozaki_step`).

## B. Shelved / gated (scope honesty)
- **[SHELVED]** Ozaki INT8 RNS *compute* for the **finite-difference** RHS —
  wrong tool (sparse + memory-bound). Keep as a research probe; do NOT invest in
  the compile fix *for the FD path*.
- **[GATED]** Modulus scan-roll compile fix — moot unless the tensor-core stencil
  path (§E) is taken, where stencil-as-GEMM likely fixes it anyway.

## C. Memory / bandwidth (integer-free, portable — do regardless)
1. **[TODO]** Temporal RK4 fusion (4 stages → 1 HBM pass), ≈4× bandwidth — the
   biggest memory win; scaffold exists, generalize.
2. **[TODO]** Fuse RHS + RK-update + BC into one pass (cut intermediate HBM
   round-trips).
3. **[TODO]** BFP48 (general reduced-precision) as CANONICAL stored state ("Y3"):
   per-block exponent metadata + halo gather + bit-shift rescale on read; ~25%
   transfer. Lossy (48 vs 52-bit + block-exponent at high dynamic range) —
   validate accuracy. BFP32 = 50%, riskier.

## D. AMR upgrades
1. **[TODO]** **Wavelet refinement criterion** (replace gradient indicator) —
   error-controlled, self-terminating; this is Dendro's puncture-handling
   mechanism. HIGHEST-VALUE, kernel-agnostic, drop-in for
   `compute_indicator_gradient`.
2. **[TODO]** Puncture / moving-box tracking refinement (depth set by the small
   BH; box follows it) — promote the `profile_amr` scaffold to production regrid.
3. **[TODO]** Level-dependent block size (smaller `BS` at deep levels → DOF
   efficiency / precision at extreme Q).
4. **[TODO]** Minimum-tile-size constraint for the tensor-core path (don't let
   deep blocks shrink below efficient MMA size; or batch small blocks).
5. **[TODO]** High-order prolongation/restriction (inter-level transfer) — low
   order here silently caps global convergence; must match the stencil order.
6. **[KEEP]** Pluggable RHS backend.

## E. Tensor-core stencil path (the thesis core)
1. **[TODO]** **Stencil-as-GEMM** (ConvStencil `stencil2row`) — maps to tensor
   cores AND likely dissolves the compile-time blowup.
2. **[TODO]** **Temporal fusion → compute-bound** stencils (2026 result;
   prerequisite for tensor-core payoff; overlaps C1). Tune fusion depth
   (α-redundancy tradeoff — keep it low).
3. **[TODO]** **Sparse tensor cores (2:4)** via SPIDER strided-swap for the
   banded stencil → kills the zero-waste.
4. **[TODO]** **INT8 / Ozaki-II on sparse tensor cores** — the open research gap.
   Stage: FP16/TF32 first (proven) → INT8/Ozaki-II (novel, bit-reproducible).
5. **[TODO]** Quantify the 6th-order padding alignment (`2r+1=7 ≈ m=8`) as the
   motivating result.
6. **[TODO]** Decompose the 3D stencil into 1D operators to map onto Ozaki splits.
7. **[TODO]** Amdahl-split design: linear derivatives → tensor cores; nonlinear
   BSSN algebra → FP64. Measure the fraction to bound the achievable speedup.
8. **[TODO]** MPIR (mixed-precision iterative refinement) for elliptic/linear
   solves (e.g. initial-data constraint solve) — proven 4–5× on tensor cores.

## E2. BSSN RHS algebra op-count reduction (compute-side, POST-fusion)
The 70% non-GEMM algebra is **FMA-bound** (~4500 ops/pt: 2486 mul/add vs ~12
transcendentals + 55 div). Tensor cores/Ozaki can't touch it → the only lever is
**fewer multiply-adds**. **⚠ NONE of these fix the memory wall** — they are compute-side
and only cash out *after* derivative fusion (3.2d) makes the kernel compute-bound;
until then a cheaper algebra just lowers arithmetic intensity. Full idea bank + user
dispositions: memory `algebra-speedup-ideas`.
1. **[TODO]** Hoist grid-constant scalars out of the per-thread Pallas kernel (the SSL
   `exp(-t²/2σ²)`, CAHD `dx²/dt`); `pow(x,±2/±3)` → multiplies (verify `integer_pow`
   already lowers). *Free; accepted.*
2. **[TODO]** **E-graph / equality-saturation op-minimization** of the RHS DAG
   (egg/egglog) — beats SymPy CSE's syntactic-only reuse; ~10–30%. *Highest-value
   experiment; pure CPU.*
3. **[TODO]** Tensor-symmetry-aware codegen (γ̃/Ricci symmetric — scalar CSE flattens it).
   *Gated: confirm tensors are symmetric in practice.*
4. **[GATED]** Structural physics `det(γ̃)=1` (kills inverse-metric division) + `tr(Ã)=0`.
   *Parked: det≠1 in practice → needs constraint enforcement.*
5. **[GATED]** Leaner conformal variable (W/χ/φ division+pow count). *Parked: choice is
   intentional design.*
6. **[KEEP]** Fewer EVALS not cheaper evals = MSRK (E/§ Phase 4.3); warp-cooperative for
   fp64 zero-spill (parked).

## F. Physics / formulation (MCS toy → BSSN)
1. **[TODO]** Switch MCS toy **System IV → System II** (BSSN analog: Γ aux
   variable, heavy ∂², Baumgarte damping). Reformulate to `A, Γ, ψ`.
2. **[TODO]** Co-rotating Gaussian-smeared charge/current sources (flat-space
   stand-in for the binary).
3. **[TODO]** BSSN(OK) 3D RHS (~24 fields) — the real target.
4. **[TODO]** Moving-puncture handling: finite conformal variable (χ/W), 1+log
   lapse, Γ-driver shift. (Singularity is *avoided by slicing* (trumpet), not
   zeroed; punctures move.)
5. **[TODO]** Bowen–York / Brill–Lindquist initial data — incl. the elliptic
   Hamiltonian-constraint solve (ties to E8/MPIR).
6. **[TODO]** KO dissipation tuned at the puncture; constraint damping.

## G. Grid / frame architecture (Phase 2)
1. **[TODO]** Llama multipatch: central Cartesian + cubed-sphere wave-extraction
   shells; static contiguous arrays.
2. **[TODO]** Co-rotating frame (inner patch): Coriolis/centrifugal shift
   `β + Ω×r`; pre-compute rotational terms.
3. **[TODO]** **Spatial dual-frame (REQUIRED)**: inner co-rotating + outer
   **inertial** wave zone with transition — the wave zone can't co-rotate
   (superluminal shift beyond `r ≈ c/Ω`).
4. **[TODO]** **Temporal hybrid**: static co-rotating graph for quasi-circular
   inspiral → one-time remap → dynamic block-AMR for plunge/merger (two compiled
   graphs).
5. **[TODO]** Inter-patch coupling (penalty/SAT or interpolation) across patch
   boundaries.
6. **[TODO]** Variable per-point FD metric weights in the warped cubed-sphere
   patches.

## H. Diagnostics & science output
1. **[TODO]** Apparent-horizon finder (tracking + diagnostics; feeds moving-box).
2. **[TODO]** Ψ₄ (Weyl) wave extraction + spin-weighted spherical-harmonic
   decomposition at the cubed-sphere shells.
3. **[TODO]** Constraint monitors (Hamiltonian + momentum) over the evolution.
4. **[TODO]** ADM mass/momentum; horizon mass/spin.
5. **[TODO]** Constraint-preserving / radiation outer boundary conditions for
   BSSN (distinct from the MCS Sommerfeld/periodic BCs).
6. **[TODO]** Waveform validation against the SXS catalog.

## I. Validation & review (for Dr. Neilsen)
1. **[TODO]** Convergence vs an FP64 CUDA baseline — framed as **high-order in
   the wave zone, wavelet-threshold-controlled error at the puncture** (NOT "6th
   order everywhere"; order at the non-smooth puncture is smoothness-capped).
2. **[TODO]** Hamiltonian/momentum constraint preservation over thousands of RK
   steps, INT8/Ozaki-II vs FP64 — the crux precision validation.
3. **[TODO]** Nsight Compute: SpTC compute-bound + TMA async-fetch latency hiding
   + WGMMA throughput.
4. **[TODO]** Bit-reproducibility demonstration (Ozaki-II/CRT) — the determinism
   contribution for long high-Q runs.

## J. Production / scaling
1. **[TODO]** Multi-GPU / multi-node JAX sharding (`shard_map`/mesh) for Marylou
   H200 nodes.
2. **[TODO]** Memory-footprint management (3D BSSN × Ozaki splits × multipatch vs
   141 GB HBM3e).
3. **[TODO]** Checkpoint / restart for long runs.

---

## Critical-path ordering

> This file is the **thematic backlog**. The **canonical ordered execution view is
> `phases/README.md`** (the Phase 0–6 spine). The thematic areas map onto it as below.
> *Superseded framings to ignore:* "System IV→II dress-rehearsal" (F1 — we port BSSN
> directly from Dendro-GR, skipping System II); "Stencil-as-GEMM = Phase-1 deliverable"
> (the GEMM survives only inside the Ozaki derivative, now Phase 4); DendroJax as
> oracle (dropped — buggy GR impl; oracle = Dendro-GR C++ + analytic tests).

1. **Phase 1 — 3D foundation** (MCS in 3D): 3D FD (E6 decompose), 3D state/BC,
   port validate/benchmark. CPU-developable; gates everything.
2. **Phase 2 — BSSN correctness** (F3/F4): verbatim Dendro-GR RHS → JAX + validate
   vs Dendro-GR C++ + apples tests (I1); the `ptxas` spill measurement.
3. **Phase 3 — GPU-optimal codegen (PRIMARY)** (E2/E7): register-bounded staged RHS
   from Dendro-GR's SymPy DAG → compute-bound.
4. **Phase 4 — Ozaki tensor-core derivative (SECONDARY)** (E1/E3/E4/C3): stencil→GEMM
   →2:4 sparse INT8/Ozaki-II + bit-reproducibility/constraint validation (I2/I4).
5. **Phase 5 — grid/frame** (G, D1 wavelet criterion, 3D AMR port) →
   **Phase 6 — BBH physics + diagnostics + scaling** (F5/F6, H, J) — as far as
   validation holds.

## Key references
- *Do We Need Tensor Cores for Stencil Computations?* (2026) — fusion → compute-bound; high-order wins. arXiv:2603.00477
- SPIDER/SPTCStencil (2025) — 2:4 sparse-TC stencils via strided swap. arXiv:2506.22035
- Ozaki Scheme II / GEMMul8 (2025) — INT8 fp64-GEMM emulation, bit-reproducible. arXiv:2504.08009
- Fernando, Neilsen, Zlochower, Hirschmann, Sundar — Dendro-GR BBH with wavelet AMR. PRD 107, 064035 (2023)
- GR-Athena++ — vertex-centered oct-tree puncture evolutions. ApJS (2021)
