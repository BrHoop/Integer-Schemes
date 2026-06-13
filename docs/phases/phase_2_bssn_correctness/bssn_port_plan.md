# BSSN Port Plan — Dendro-GR → Integer-Schemes (3D)

**Status: PLANNED (Phases 1–4 of the spine).** Supersedes the old DendroJax-oracle
sketch. Written after the intensity model showed MCS is too compute-light to
demonstrate the tensor-core payoff (BSSN is the compute-bound efficiency ground),
and **after DendroJax's GR implementation was found buggy** → the port now sources
from **Dendro-GR**, the group's published/validated code.

> This is **Phase 2 (BSSN correctness)** in the canonical spine (`../README.md`):
> Phase 1 (3D foundation, ✅) → **Phase 2 (BSSN correctness)** → Phase 3 (codegen)
> → Phase 4 (Ozaki derivative). The execution checklist lives next to this file in
> `phase_2_plan.md`; this doc is the strategy/port narrative.

## Why this, why here

The tensor-core/Ozaki thesis can only be *demonstrated* on a compute-bound
problem. MCS (10 fields, ~2400 FLOP/pt/step, ~15 FLOP/byte) is DRAM-bound — the
wrong proxy. BSSN (24 fields, **~24k FLOP/pt/step ≈ 62 FLOP/byte**) is compute-bound
(modulo the register-spill caveat below) and is the actual physics target. So BSSN
comes in as the **efficiency ground**; MCS stays the **cheap correctness /
Ozaki-bit-reproducibility ground**.

**Port BSSN *into* Integer-Schemes** (not the reverse, not "work in someone's
octree repo), because:
- **Octree is a dead end for the endgame.** The goal is wave-zone extraction at
  large radius, a spatial **dual-frame** (co-rotating inner + inertial outer), and
  **cubed-sphere/Llama multipatch** — all wanting spherical-shell grids, not
  Cartesian boxes. An octree structurally can't become that.
- Integer-Schemes is where the **Ozaki machinery, validation suite, roofline
  tooling, and the planned grid/frame work** live.

## Port source: Dendro-GR (DendroJax dropped)

**DendroJax's GR implementation is buggy → disqualified as a correctness oracle**
(validating a port against a buggy reference is worse than no oracle). The port
sources from **`~/Code/Dendro-GR`** instead:

- **Verbatim (Phase 2.2):** *transliterate* the generated CSE
  `CodeGen/bssneqs_sympy_cse_wo_derivs.cpp` (942 lines, derivatives-as-inputs, flat
  `const double DENDRO_N = <expr over earlier DENDRO_*, field[pp], grad_*>;` SSA)
  → vectorized JAX. Each `field[pp]` becomes a JAX array, each `grad_*` a precomputed
  derivative array (from the Phase-1 3D FD machinery), each `DENDRO_N` a `jnp`
  expression. **Correct-by-construction from trusted output — no SymPy toolchain, no
  DendroJax.**
- **Staged regen (Phase 3):** drive Dendro-GR's SymPy front-end
  (`CodeGen/bssn.py`/`dendro.py`/`bssn_nx.py`) to emit a register-bounded, staged
  JAX/Pallas RHS — replacing only the C++ back-end emitter (the front-end is the
  reusable asset).
- **Oracle:** **Dendro-GR C++** (identical ID, one RHS eval, compare to round-off)
  **+ the analytic apples tests** (reference-free). Use it to validate, then walk away.

## The honest framing (codegen primary, tensor cores secondary)

1. **The win is fusion/codegen.** BSSN's RHS is ~70% point-wise nonlinear algebra
   (Christoffel/Ricci/gauge — *not* GEMM-able) and only ~30% derivatives (stencil —
   tensor-core-able). Amdahl caps a perfect derivative speedup at **~1.4×** on the
   full RHS. The larger lever is a **GPU-optimal fused RHS** that keeps intermediates
   on-chip. **EVIDENCE:** static dataflow on Dendro-GR's CSE RHS found **peak ~584
   (median 378) simultaneously-live temps, >255 across 69% of the kernel**, + 138
   distinct derivative inputs → ~700+ live vs the 255-register file → **the
   compute-heavy RHS is very likely memory-bound on its own intermediates.** Honest
   thesis: *high-order GR RHS is memory-bound on GPUs because symbolic-CSE codegen
   overruns the register file; a fusion-first, register-bounded generator makes it
   compute-bound, and tensor cores then accelerate the stencil portion.*
   - **Hardware confirm (Phase 2.2):** one `ptxas -v` compile
     (`--xla_gpu_asm_extra_flags=-v`) of the *ported JAX* RHS — does XLA spill to
     local memory, or split into HBM-backed kernels? (Dendro-GR's CPU C++ can't
     answer this; register allocation is architecture-specific.)
   - **Staging is the design** (deliberated, done): store the ~69 high-fan-out
     tensor-hierarchy temps (inverse metric, Christoffels, Ricci, CalGt) as per-point
     register scalars, recompute/inline the ~800 cheap leaves → ~133 live → fits 255.
     Both extremes dead: recompute-all = 623k raw ops (~150×); store-all (CSE) =
     584-live spill. The contribution = **automate the cut-point/store-set selection**
     (DAG min-cut / articulation) and **emit JAX/Pallas** staged.
2. **3D kills shared-memory tiling** (24 fp64 fields → tiles too big, 64× redundant)
   — but BSSN doesn't need it (compute-bound untiled). The 2D tile-and-fuse work is a
   toy-model artifact; **do not port it forward.**

## Phase 1 — 3D foundation (MCS in 3D)

Integer-Schemes is 2D; BSSN is irreducibly 3D. Get the **infrastructure** right and
validated on simple physics *before* the heavy RHS:
- 3D state (NF fields), 3D 6th-order FD (1st, 2nd diagonal, 2nd mixed), 3D
  ghost/periodic + Sommerfeld outer BC, RK4 (have).
- **Port the 2D machinery to 3D:** `validate.py`, benchmark/roofline, the schemes.
- **Validate** vs the 3D birefringent oracle (MCS extends to 3D) + the ported
  validation suite (convergence order, constraints, spectrum).
Exit: a validated 3D MCS with the full machinery. BSSN is then an RHS swap onto it.

## Phase 2 — BSSN correctness (the port)

**2.1 State + gauge.** 24 vars (`alpha, chi, K, gt0-5, beta0-2, At0-5, Gt0-2, B0-2`),
`PhysicsParams` (eta/CAHD/SSL/lambda gauge knobs), asymptotic values, chi/alpha floors.

**2.2 Verbatim RHS + validate + spill measurement.** Transliterate
`bssneqs_sympy_cse_wo_derivs.cpp` → JAX (fed by Phase-1 derivs). **Bit-compare vs
Dendro-GR C++** (identical ID, one RHS eval, round-off). **+ the `ptxas -v` spill
measurement** of the compiled JAX RHS (the number that motivates Phase 3). Port the
RHS only; leave octree/wavelets/ghost-exchange behind.

**2.3 Physics validation.** Standard apples tests: gauge wave, robust-stability
(random noise stays bounded), Hamiltonian + momentum constraint convergence —
correctness independent of any reference.

## Phase 3 — GPU-optimal codegen (the contribution that doesn't exist anywhere)

Re-balance the codegen for GPU: less CSE / more recompute to fit registers, full
point-wise fusion so intermediates never hit HBM, via the **staging** design above.
**3.1** hand-stage the worst offenders (guided by 2.2). **3.2** automate the
cut-point/store-set selection from Dendro-GR's SymPy DAG (`bssn_nx.py`) → emit
staged JAX/Pallas (pin XLA via `remat`/`optimization_barrier`). **3.3** measure the
regime flip (`profile_regime --smi` MEM%↓/SM%↑, `ptxas` spill→0). *The larger win.*

## Phase 4 — Ozaki tensor-core derivative + efficiency demonstration

Build+validate the bit-reproducible Ozaki-II INT8 `compute_deriv` on cheap MCS (fix
the `pallas_ozaki` ~1600 s compile blocker), then deploy into the 3D BSSN RHS;
measure the compute-bound efficiency gain (Amdahl-capped at the ~30% derivative
share) and **bit-reproducibility** vs the FP64 derivative. The thesis efficiency
result on real physics. (MSRK 2-step/3-stage integrator deploys here as an orthogonal
multiplier — re-measure ECF on the gamma-driver spectrum.)

## Deferred — Phase 6 (BBH proper)

Moving-puncture handling (χ/W, 1+log, Γ-driver), Bowen–York / Brill–Lindquist
initial data + elliptic solve, puncture tracking, BBH-grade AMR. Not needed for the
efficiency demonstration.

## Scope / risk notes

- **3D infrastructure is the real cost** — the RHS algebra is code-gen (exists in
  Dendro-GR); the 3D state/derivs/BC/halo build is genuine work. Budget for Phase 1.
- **Transliteration fidelity:** the `.cpp` → JAX transform is mechanical but must be
  exact (operator precedence, `1.0/x`, `pow`, integer vs float literals). The
  Dendro-GR C++ bit-compare (2.2) is the guard.
- **BSSN stability needs babysitting** even when correct (CAHD/dt stiffness,
  conformal-det renormalization); the analytic tests catch regressions.
- **Keep MCS** as the fast Ozaki-bit-reproducibility oracle throughout.

## Status / changelog
- 2026-06-10 — Created (DendroJax-oracle version).
- 2026-06-11 — **Retargeted to Dendro-GR.** DendroJax dropped (buggy GR impl);
  verbatim port = transliterate Dendro-GR's generated CSE `.cpp` → JAX; oracle =
  Dendro-GR C++ + analytic tests; spill confirm rides on the JAX RHS (CPU Dendro-GR
  can't measure GPU registers). Renumbered to the linear spine (Phases 1–4); Ozaki
  derivative moved to Phase 4 (secondary, at its deployment).
