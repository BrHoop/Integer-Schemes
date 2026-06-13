# Step 1.1 — Baseline Validation + Profiling Harness (2D MCS, single-level)

**Phase 1 · Step 1.1 · Status: PLANNED (next to implement)**

## Purpose
Establish that the standard 2D Maxwell–Chern–Simons (MCS) finite-difference
solver is **correct, well-posed, and stable**, and capture a **performance
baseline** — *before* any tensor-core work. Everything in Phase 1 and beyond
assumes this foundation, so it must be airtight. The profiling here also
directly informs later Phase-1 steps (regime, fusion) and future phases.

## Scope & decisions
- **AMR OFF.** Single uniform periodic grid. Rationale: isolate the core FD
  scheme — AMR adds inter-level interpolation / regridding-noise / prolongation-
  order confounders that would make a scheme bug indistinguishable from an AMR
  artifact. (AMR-vs-single-level agreement is a *separate, existing* test, run
  **after** 1.1, not part of it.)
- **System IV** (the existing code) — scheme/hardware validation is formulation-
  agnostic. The **System IV → System II** reformulation (the faithful BSSN
  analog) is **Phase 3, Step 3.1**, where it makes this same validation stress
  the ∂² stencils the way BSSN will.
- **Eigenvalue focus (confirmed with user):** the **semi-discrete operator
  spectrum** is the PRIMARY test. The continuum characteristic-speed and
  symmetrizer/energy tests are included as supporting well-posedness checks.

## Eigenvalue terminology (resolved — three distinct concepts)
| Test | Object | "Good" means |
|---|---|---|
| **Semi-discrete spectrum** (PRIMARY) | eigenvalues of the discrete RHS operator | **non-positive real parts** (a *positive* real part = exponentially growing mode = instability); `λ·dt` inside the RK4 stability region |
| Characteristic speeds | eigenvalues of the continuum principal symbol | **real** ⇒ hyperbolic / well-posed |
| Symmetrizer / energy | the symmetrizer matrix | **positive-definite** ⇒ symmetric hyperbolic |

> Note: for an evolution operator "positive eigenvalues" are *bad* (growth). The
> place "positive = good" applies is the symmetrizer (energy positivity).

---

## Test Suite A — Correctness & convergence
1. **Analytic-oracle convergence (gold standard).** Use the birefringent
   plane-wave solution (already in `benchmark._make_oracle`): the CS coupling
   splits the two polarizations into distinct dispersions `ω±(k)`. Run at
   `N, 2N, 4N`; report absolute L2 error + convergence order **for every field**
   (Ex…Phi), not just Ez.
2. **Separate spatial vs temporal order (critical gotcha).** RK4 is 4th-order in
   time, the stencil 6th in space. With `dt ∝ dx` you measure `min(4,6)=4` and
   wrongly conclude "4th order." To isolate **6th-order spatial**: hold `dt` tiny
   (time error negligible) or scale `dt ∝ dx^{3/2}` so `dt⁴ ∝ dx⁶`. To confirm
   **4th-order temporal**: hold `dx` tiny, vary `dt`. Report both separately.
3. **KO-dissipation order check.** Run with/without KO and across σ; confirm the
   6th-order KO does NOT degrade the spatial order (it shouldn't — higher order —
   but a mis-scaled σ silently kills convergence; this is the bug class behind
   the earlier KO-sign issue).
4. **Linear & nonlinear regimes.** `λ=0` (pure Maxwell, cleanest exact solutions)
   AND `λ≠0` (actual nonlinear MCS). Convergence must hold in both.
5. **Self-convergence (Richardson)** as a backup where no exact solution exists
   (three resolutions, Q-factor → order).

## Test Suite B — Hyperbolicity & eigenvalue spectrum
1. **Semi-discrete operator spectrum (PRIMARY).** MCS is linear, so the spatial
   RHS *is* a matrix on a small periodic grid — build it explicitly (or via
   JVP probing), compute eigenvalues. Check: (a) **no large positive-real-part
   modes** (well-posed discretization), (b) `spectrum·dt` fits inside the **RK4
   stability region** ⇒ empirical **CFL limit**, (c) KO adds the expected
   negative-real-part damping to high-`k` modes.
2. **Principal-symbol eigenvalues (continuum).** Characteristic matrix for a
   generic `k̂` → eigenvalues = characteristic speeds. Confirm **real**
   (hyperbolic); expect `±1` (light), `0` (gauge/constraint), and the
   **birefringent split** from λ. **Sweep λ → locate where eigenvalues go
   complex (loss of hyperbolicity / CFJ tachyon).** This maps the well-posed λ
   range and confirms the known `Λ=2`-unstable regime.
3. **Constraint-damping eigenvalues.** Confirm the Ψ/Φ (κ₁,κ₂) sector gives
   constraint-violating modes **negative** real parts (decay), tracking κ. Direct
   analog of the BSSN Hamiltonian-constraint damping validated later.
4. **Symmetrizer / energy positivity (optional).** Check a positive-definite
   energy norm exists (symmetric-hyperbolicity).
5. **Dynamical cross-check.** Evolve a polarized wave packet, **measure numerical
   propagation speeds**, confirm they match the analytic birefringent `ω±(k)`.
   Ties the abstract spectrum to a real evolution.

## Test Suite C — Stability & conservation (long-run)
1. **Long-term stability.** Thousands of steps; no blow-up — small-scale analog
   of the "thousands of RK steps" BSSN criterion.
2. **Constraint preservation over time** (THE most BSSN-relevant test). Monitor
   `∇·E − 4πρ − 2λ(…)` and `∇·B` vs time; with damping they stay bounded/decay.
3. **Energy monitor.** Track system energy (or its expected KO/damping decay).
4. **Boundary conditions.** Exercise periodic (convergence) + Sommerfeld/
   directional (outflow, no spurious reflection); confirm each behaves.

## Test Suite D — Profiling baseline (each item feeds a later step)
1. **Roofline / arithmetic intensity** (`cost_analysis` + Nsight): FLOP/byte of
   the RHS → confirm memory-bound now (~0.5). → *feeds 1.3 (regime) & 1.4
   (fusion decision): how far from compute-bound.*
2. **Memory traffic** (bytes read/written per step). → *baseline for BFP48 /
   temporal-fusion bandwidth wins (ROADMAP C1–C3).*
3. **Per-stage timing** (RHS vs derivative kernels vs BC vs RK4) + **throughput**
   (Mpts/s, per-step vs grid, scan-based). → *where time goes; baseline for every
   later speedup claim.*
4. **Stencil-order scaling** (4th/6th/8th): accuracy-per-cost + FLOP/byte vs
   order. → *quantifies "high order pushes toward compute-bound" — supports the
   whole tensor-core path and the 6th-order-alignment result (1.6 / ROADMAP E5).*
5. **Compile-time baseline** per scheme. → *reference for the stencil-as-GEMM
   compile fix (ROADMAP E1) and the 1600 s problem.*

---

## Deliverables
A reproducible harness producing: convergence tables (spatial + temporal),
per-field error plots, **eigenvalue-spectrum plots** (discrete + continuum) with
the **λ-hyperbolicity map**, constraint-vs-time plots, and a profiling/roofline
report — bundled into one **validation report** suitable to hand Dr. Neilsen as
"the base scheme is correct, well-posed, stable, and here is its performance
profile."

## Reuse vs build
- **Reuse:** birefringent oracle (`benchmark._make_oracle`), schemes + RK4
  stepper, `benchmark.py`'s scan-timing + `_rhs_cost`, `calc_constraints`.
- **Build:** convergence driver (spatial/temporal separation), eigenvalue/
  spectrum analyzer (discrete operator + continuum symbol), λ-sweep hyperbolicity
  map, long-run constraint monitor, report assembler.

## Implementation
- `2D/src/mcs2d/validate.py` — the characterization harness (runs the suites,
  emits plots + a Markdown/JSON report). Exploratory/report-style.
- `2D/tests/validation/` — a small pytest set asserting the **key invariants**
  for CI: spatial order ≥ ~5.5 (smooth, isolated), no discrete eigenvalue with
  real part > tol, constraints bounded over N steps. Pass/fail regression guard.

## Exit criteria
- ✅ Spatial convergence ≈ 6th order (isolated) & temporal ≈ 4th; both verified.
- ✅ Discrete spectrum has no spurious growing modes; CFL limit measured; KO
  damping present. Continuum speeds real over the intended λ range (and the
  unstable λ regime identified).
- ✅ Constraints bounded/decaying over thousands of steps.
- ✅ Profiling baseline captured (roofline, traffic, throughput, order-scaling,
  compile time) — ready to inform 1.3/1.4.

## Concrete parameters (initial — adjust as needed)
- **Convergence grids:** N ∈ {64, 128, 256} periodic; report `p = log₂(e(N)/e(2N))` per field.
- **Spatial-order isolation:** `dt ∝ dx^{3/2}` (or a fixed tiny `dt`) so `dt⁴ ≪ dx⁶`.
  **Temporal-order:** fix N=256, vary `dt`.
- **Eigenvalue spectrum:** small grid N ∈ {16, 32} (operator size `NF·N²`; 16² → 2560,
  dense `eig` is fine). Periodic.
- **λ hyperbolicity sweep:** λ ∈ {0, 0.1, 0.5, 1, 2, 5} — expect real speeds at small λ,
  complex eigenvalues (loss of hyperbolicity) beyond threshold; confirm the Λ=2 regime.
- **KO:** σ ∈ {0, 0.02, 0.1}. **Long-run:** ≥ 5000 RK4 steps.

## Methods
- **Discrete operator (linear MCS):** build the RHS matrix `J` by applying `rhs` to unit
  basis vectors on a small periodic grid (or `jax.jacfwd` of the flattened RHS); eigenvalues
  via `np.linalg.eig`.
- **RK4 stability:** for each eigenvalue λ, `z = λ·dt`; require `|R(z)| ≤ 1`,
  `R(z) = 1 + z + z²/2 + z³/6 + z⁴/24`. CFL = largest `dt` keeping all `z` in the region
  (pure-imaginary axis stable to `|z| ≈ 2.83`).
- **Continuum symbol:** plane wave `e^{i k·x}` into the linearized RHS → characteristic
  matrix `M(k̂)`; eigenvalues = speeds.

## Expected ground truth
- **Speeds:** ±1 (light), 0 (gauge/constraint); λ-birefringence splits the two transverse
  polarizations into distinct `ω±`. Real (hyperbolic) for small λ.
- **Orders:** spatial ≈ 6 (isolated), temporal ≈ 4 (RK4).
- **Constraints:** bounded/decaying at the rate set by κ₁,κ₂; converge to 0 with resolution
  at the scheme order.

## Status / changelog
| date | what |
|---|---|
| 2026-06-09 | Step planned & documented (suites A–D + params/methods/expected); ready to implement. |
