# Phase 2 — BSSN correctness (port from Dendro-GR): implementation plan

**Status: ⬜ NEXT.** The strategy/port narrative is in `bssn_port_plan.md`; this is
the execution checklist (mirrors `../phase_1_3d_foundation/phase_1_plan.md`). Phase 2
puts a **correct** 3D BSSN RHS onto the validated Phase-1 machinery. It is a
*transliteration + validation* phase — **not** the codegen/efficiency work (that is
Phase 3) and **not** new physics derivation (Dendro-GR already generated the RHS).

Exit = a 3D BSSN RHS that (a) **bit-matches Dendro-GR C++** on one eval and (b)
passes the reference-free apples tests, plus the `ptxas -v` spill number that
motivates Phase 3.

---

## 1. The landing pad (what Phase 1 hands us) + what is genuinely new

Phase 1 left a validated 3D foundation that BSSN drops onto with minimal new
infrastructure:

| Need | Phase-1 status | Phase-2 work |
|---|---|---|
| 3D state container | `WaveState` (NF=10 pytree) | **new** `BSSNState` (24 vars) — same pytree pattern |
| 6th-order 3D FD: 1st (`grad_`), 2nd diagonal+mixed (`grad2_`) | `SpatialDerivative` (orders 4/6/8, validated bit-vs-2D) | **reuse**; add a batched "derivative bundle" builder (72 `grad_` + 66 `grad2_`) |
| RK4 stepper | `step_rk4` | reuse |
| periodic + Sommerfeld BC, ghost zones | have | reuse (radiative outer BC is Phase-2.1 detail) |
| validation harness pattern (`validate.py`: one-object, AD cross-check, apples tests) | have | **mirror** for BSSN (constraints, gauge wave, robust stability) |
| benchmark/roofline + `ptxas`/`profile_regime` plumbing | have (3D) | reuse for the spill measurement |

**Source-of-truth facts (audited 2026-06-11 in `~/Code/Dendro-GR/CodeGen`):**
- **24 evolved vars:** `alpha, chi, K, gt0–5, beta0–2, At0–5, Gt0–2, B0–2`
  (outputs in the `.cpp`: `a_rhs, chi_rhs, K_rhs, gt_rhs[0–5], At_rhs[0–5],
  Gt_rhs[0–2], b_rhs[0–2], B_rhs[0–2]`).
- **`bssneqs_sympy_cse_wo_derivs.cpp`:** 942 lines, **623 308 original ops**, flat
  `const double DENDRO_N = <expr over earlier DENDRO_*, field[pp], grad_*>;` SSA.
- **138 derivative inputs = 72 first-centered (`grad_i_f`) + 66 second-centered
  (`grad2_i_j_f`, diagonal *and* mixed). NO advective/upwind (`agrad_`) in this
  variant** → the Phase-1 centered FD provides *exactly* these inputs; **no new
  derivative operator is needed for the verbatim port** (confirm via the bit-compare,
  risk §4.2).
- Gauge params referenced: `eta`, `lambda_f[0..1]` (+ the standard moving-puncture
  knobs). The CSE consumes derivatives **as inputs** — so the algebra can be
  bit-checked in isolation from the FD (see 2.2).

**Genuinely new in Phase 2:** the 24-var state + gauge params, the derivative-bundle
builder, the **transliterated algebra** (the bulk), the Dendro-GR bit-compare oracle
harness, and the BSSN validation suite. No new physics, no new FD math.

### Package layout (decide at 2.1)
BSSN is 3D and reuses the 3D FD operator. Put it in a sibling package
`3D/src/bssn3d/` that imports the derivative operator — and **promote
`SpatialDerivative` to `common/`** rather than make a third copy (it is already
duplicated 2D/3D; BSSN would be the third). Keep MCS untouched (it stays the cheap
Ozaki-bit-reproducibility oracle).

---

## 2. Steps

### Step 2.1 — BSSN state + gauge params (+ outer BC choice)
- `BSSNState` pytree: 24 fields, same `register_pytree_node_class` pattern as
  `WaveState`; named accessors matching the Dendro var order.
- `PhysicsParams`: `eta` (Γ-driver damping), `lambda`/`lambda_f` (1+log + shift
  gauge), CAHD coefficient (`cahd_c ~0.06`), SSL, `chi`/`alpha` floors, asymptotic
  values (α→1, gt→δ, chi→1, rest→0).
- Initial data for *testing only* (not BBH): gauge-wave ID + Minkowski-with-noise
  (robust-stability ID). Bowen–York/puncture ID is deferred to Phase 6.
- **Outer BC:** adopt the decided radiative (Bayley–Sommerfeld) recipe ~verbatim +
  CAHD + KO + algebraic enforcement (det gt=1, tr A=0). Periodic suffices for the
  gauge-wave and robust-stability apples tests, so the radiative BC can land
  incrementally. (See the outer-BC memory; SBP/SAT avoided by decision.)

### Step 2.2 — Verbatim RHS + bit-compare + spill measurement (the core)
- **Derivative bundle:** one function `state → {grad_i_f, grad2_i_j_f}` building all
  138 arrays from the Phase-1 `SpatialDerivative` (first via `compute_d1`; diagonal
  second via `compute_d2`; **mixed** second via composed `compute_d1∘compute_d1`).
  Batch with the `_d1_batched` vmap pattern (the shape the Ozaki/staged path wants).
- **Transliterate `bssneqs_sympy_cse_wo_derivs.cpp` → JAX:** each `field[pp]`→a JAX
  array, each `grad*_…`→a bundle array, each `const double DENDRO_N = …`→a `jnp`
  expression (one straight SSA block; XLA handles the graph). Mechanical but must be
  **exact** — operator precedence, `1.0/x`, `pow`, integer-vs-float literals.
- **DECISIVE GATE — bit-compare vs Dendro-GR C++** (the Phase-2 analogue of Phase-1's
  symbol↔Jacobian check): capture **one RHS eval's inputs+outputs** from Dendro-GR
  (fields + the 138 derivative arrays + the 24 `*_rhs`) on identical data; feed the
  *same inputs* to the JAX RHS; diff to round-off. Because the `.cpp` takes
  derivatives as inputs, this tests the **algebra transliteration in isolation** (no
  FD confound) — the cleanest possible fidelity gate.
- **Spill measurement (motivates Phase 3):** one `ptxas -v` compile of the compiled
  JAX RHS (`--xla_gpu_asm_extra_flags=-v`) on the H200 — does XLA **spill to local
  memory** or **split into HBM-backed kernels**? Record the register/spill number.
  (Dendro-GR's CPU C++ cannot answer this; GPU-only. User 2FA.)

### Step 2.3 — Physics validation (reference-free apples tests)
Mirror the `validate.py` harness for BSSN; each view gets a guard test:
- **Gauge wave:** 1D harmonic gauge wave at known amplitude → convergence at FD
  order; no spurious growth over many crossing times.
- **Robust stability:** Minkowski + low-amplitude random noise → constraints and
  fields stay **bounded** (no exponential blow-up) — the standard NR robustness test.
- **Constraint convergence:** Hamiltonian + **momentum** constraints converge at the
  expected order under refinement (momentum is the under-protected channel — only
  Gt + KO guard it; CAHD damps Hamiltonian). Reference-free.
- **Conformal algebra:** det(gt)=1 and tr(A)=0 enforced/preserved; chi>0 floor holds.
- Keep MCS as the fast Ozaki-bit-reproducibility oracle alongside (unchanged).

---

## 3. Reuse vs build vs audit

- **Reuse:** `SpatialDerivative` (promote to `common/`), `step_rk4`, BC/ghost
  machinery, the `validate.py` harness *pattern*, benchmark/`ptxas`/`profile_regime`
  plumbing, the pytree-state pattern.
- **Build:** `BSSNState` + `PhysicsParams`, the 138-array derivative bundle, the
  transliterated RHS, the Dendro-GR bit-compare harness, the BSSN apples-test suite.
- **Audit (verify, then lock):** that the Phase-1 centered FD reproduces Dendro's
  `grad_`/`grad2_` conventions (factor, stencil width, mixed-derivative ordering) —
  the bit-compare is the audit.

---

## 4. Risks / gotchas

1. **Transliteration fidelity is everything.** 942 SSA lines, 623k ops: one wrong
   precedence/`1.0/x`/literal-type silently corrupts the RHS. *Mitigation:* the
   Dendro-GR bit-compare (2.2) on the isolated algebra is the guard — do it FIRST,
   before any evolution. Consider auto-translating the `.cpp` with a small parser to
   remove hand-typo risk, then bit-checking.
2. **Derivative convention match.** Confirm Dendro's `grad_i_f` is the same centred
   6th-order operator (same normalization) and whether the production driver upwinds
   *any* `grad_` for β·∂ advection (this CSE variant has no `agrad_`, but the live
   solver may). The bit-compare on real Dendro inputs settles it; if upwinding is
   used, add an advective operator (Phase-1 FD is centred-only).
3. **Building the oracle.** Getting one trusted RHS eval out of Dendro-GR (C++/MPI
   build, or a minimal harness around the generated kernel) is real setup work. The
   `_wo_derivs` design helps: dump fields+derivatives+outputs for one ID, compare
   offline — no need to run JAX against live C++.
4. **BSSN needs babysitting even when correct** — CAHD/dt stiffness, conformal-det
   renormalization, gauge transients. The apples tests catch regressions; budget for
   tuning, not just porting.
5. **State growth:** 24 fp64 fields × `(N+2·NG)³` is ~2.4× MCS per point; keep
   validation grids modest (N≈32–64) so CPU dev stays quick. The spill/regime numbers
   need the H200.
6. **Don't drift into Phase 3.** The verbatim RHS *will* spill (that's the point — it
   motivates Phase 3). Do **not** start hand-staging/fusing here; just measure it.

---

## 5. Exit criteria

- ✅ `BSSNState` (24 vars) + `PhysicsParams`; gauge-wave + robust-stability ID.
- ✅ Derivative bundle (72 `grad_` + 66 `grad2_`) from the Phase-1 FD.
- ✅ **Verbatim RHS bit-matches Dendro-GR C++ to round-off on one eval** (the decisive
  transliteration gate) on the isolated algebra. **DONE 2026-06-11: max relative diff
  3.1e-16 across 5 seeds** (CPU-only `g++` oracle, `bssn3d/oracle.py`).
- ✅ Apples tests green (2026-06-11): **Ham + Mom constraints converge at ~6th order**
  (5.97/5.96 on the analytic gauge-wave ID — validates the constraint transliteration);
  robust stability bounded; gauge-wave dynamically stable (bounded — note: this CSE
  variant has no CAHD/Z4c, so undamped momentum drifts secularly, expected);
  conformal-algebra (det g̃=1, tr Ã=0) preserved under evolution to round-off.
- ✅ BSSN test suite (unit: bundle/algebra; integration: short evolutions) + `run.sh`.
- ✅ `ptxas -v` spill number recorded for the JAX RHS (the Phase-3 motivation).
  **DONE 2026-06-11 (H200, `bssn3d/spill_probe.py`, N=48):** XLA hits BOTH failure
  modes — **fragments the RHS into 79 fusion kernels** (HBM-coupled intermediates)
  AND the big pointwise-algebra fusions are **register-bound at the 255 ceiling with
  spills** (3 spill: 1168/728/640 B = 2536 B total; worst kernels at 255 regs).
  Empirical confirmation that the verbatim CSE RHS is memory-bound on its own
  intermediates → motivates Phase 3.

---

## 6. Suggested order of work
2.1 state+gauge+ID (fast, unblocks everything) → 2.2 derivative bundle → 2.2
transliterate + **Dendro-GR bit-compare on the isolated algebra** (the decisive gate,
do early — like Phase-1's Jacobian check) → 2.3 apples tests + suite → 2.2 the
`ptxas -v` spill measurement (batches with other GPU work) → hand off to Phase 3
(codegen) with the spill number in hand.

## Status / changelog
- 2026-06-11 — **Step 2.3 COMPLETE (apples tests) → Phase 2 DONE.** Built evolution
  (`evolve.py`: RK4 + KO + algebraic enforcement det g̃=1/tr Ã=0/floors + periodic BC)
  and the constraint operator (`constraints.py` + transliterated `physcon.cpp` →
  `_constraints_generated.py`, **DCE'd from ham/mom0-2** so the psi4/coord/`sqrt`
  subtree drops → 672 stmts, pow-only). Tests: `test_bssn_constraints.py` (Ham+Mom
  converge **5.97/5.96 ≈ 6th order** on the analytic ID — validates the constraint
  transliteration reference-free; Minkowski constraints vanish; regen guard) and
  `test_bssn_apples.py` (conformal algebra preserved to round-off; robust stability
  bounded; gauge wave dynamically stable — `slow`-marked). Honest caveat baked into
  the test: no CAHD/Z4c in this CSE variant → undamped momentum drifts at coarse N
  (expected); the rigorous constraint check is the 6th-order ID convergence.
  **Vendored `physcon.cpp`.** Also fixed a local-dev hazard: the 10k-char generated
  lines + the XLA dump crashed Pylance → added `.vscode/settings.json` +
  `pyrightconfig.json` excludes, git-ignored `bssn_xla_dump/`. **Phase 2 complete;
  next: Phase 3 (GPU-optimal codegen).**
- 2026-06-11 — **Step 2.2 COMPLETE: spill number measured on H200.** `bssn3d/
  spill_probe.py` (turnkey: dumps XLA PTX to a persistent in-repo dir, runs the
  CUDA-wheel `ptxas -v` at sm_90a, parses registers/spill). Verbatim RHS at N=48:
  **79 fusion kernels**, **3 spilling** (1168/728/640 B, total 2536 B), worst
  fusions **pegged at 255 registers**. Both predicted forks fire — register-spill
  AND HBM-kernel-fragmentation → the verbatim CSE RHS is memory-bound on its own
  intermediates (Phase-3 motivation, now empirical not just static-584). Probe
  gotchas fixed en route: lazy `bssn3d/__init__` so `XLA_FLAGS` is set before jax
  imports; dump dir off `/tmp` (cluster-unreliable); cache disabled to force a
  fresh compile; prefer the wheel `ptxas` (PTX ISA match). **Step 2.2 fully done;
  next: 2.3 apples tests.**
- 2026-06-11 — **Step 2.2 DECISIVE GATE PASSED + project made self-contained.**
  `bssn3d/oracle.py`: auto-generates a standalone C++ harness (162 inputs baked as
  hex-float literals → `#include` the CSE → print the 24 outputs), compiles with
  `g++ -std=c++17` (hex floats need C++17), and diffs vs the JAX algebra on identical
  single-point inputs. **Max relative diff 3.1e-16 across 5 seeds** (many outputs
  bit-identical) — the transliteration is validated, ~10⁴× inside the round-off bar.
  Test `3D/tests/integration/test_bssn_oracle.py` (CPU, skips without g++).
  **Self-containment:** vendored the one Dendro file into `bssn3d/vendor/`
  (+ provenance README); `_codegen`/`oracle` now read the in-repo copy — no runtime
  reference to `~/Code/Dendro-GR`. Remaining 2.2: only the `ptxas -v` spill number
  (GPU, user 2FA). **Next: 2.3 apples tests.**
- 2026-06-11 — **Step 2.2 core landed (transliteration + bundle + structural gates;
  CPU).** Parser-based translator `bssn3d/_codegen.py` (textual, not a full parser —
  after comment-strip the CSE is pure arithmetic with `pow` as the only function and
  `lambda`→`lmbda` the only keyword fix) emits the committed
  `bssn3d/_bssn_rhs_generated.py` (911 SSA statements, 72 grad1 + 66 grad2 inputs,
  24 outputs in state order). `derivative_bundle.py` builds the 138 arrays from the
  Phase-1 FD (mixed 2nd = composed `compute_d1`); `rhs.py` (`BSSNSolver`) wires
  bundle + algebra → `BSSNState` (no KO/BC/renorm yet — kept separable for the
  isolated bit-compare). `tests/unit/test_bssn_rhs.py` (7 green): **Minkowski-static
  gate** (full RHS = 0 to 3e-14 — exercises all 911 stmts / 138 derivs / 24 outputs /
  SSA order / `pow`), bundle key/shape, finiteness, determinism, jit, and a
  **regen-idempotence guard** (committed artifact == fresh generation).
  **ORACLE DE-RISKED:** because the CSE takes derivatives as inputs, its only external
  symbol is `pow`, so `g++ -std=c++14` compiles the `.cpp` body *standalone* (no
  octree/MPI) — confirmed compiling+running. ⇒ the decisive bit-compare is a
  ~200-line auto-generated harness (declare inputs → `#include` the CSE → dump
  outputs), **fully CPU-side, no 2FA** (revises risk §4.3 down). Remaining 2.2:
  (a) auto-generate the harness + hex-float value exchange + offline diff to round-off
  (the decisive gate); (b) the `ptxas -v` spill number (GPU, 2FA).
- 2026-06-11 — **Step 2.1 landed (state + params + test ID + package).** Promoted
  `SpatialDerivative` to `mcs_common.derivatives` (single source; `mcs3d.schemes.
  floating_point` now re-exports it verbatim — full 48-test 3D suite still green, no
  behavior change; the order-6-locked 2D copy left untouched for now). New sibling
  package `3D/src/bssn3d/`: `state.py` (`BSSNState` 24-var pytree in Dendro output
  order + `PhysicsParams` — eta/lambda[0..3]/lambda_f[0..1] consumed by the verbatim
  RHS, plus reserved CAHD/SSL/floors/asymptotics), `grid.py` (ghost-padded cubic
  grid), `initial_data.py` (minkowski, robust-stability noise, 1D harmonic
  gauge-wave — analytic, det g̃=1 and conformal-traceless Ã to round-off).
  `3D/tests/unit/test_bssn_state.py` (7 green): det g̃=1, g̃^{ij}Ã_ij=0, pytree
  contract, params-as-static-JIT-arg. **Next: 2.2** — derivative bundle (72 grad_ +
  66 grad2_) → transliterate `bssneqs_sympy_cse_wo_derivs.cpp` → Dendro-GR
  bit-compare gate.
- 2026-06-11 — Created alongside the docs/phases rename (Phase 2 = BSSN correctness).
  Grounded in the audited Dendro-GR source: 24 vars, 942-line/623k-op flat-SSA CSE,
  138 derivative inputs (72 `grad_` + 66 `grad2_`, no `agrad_`) → the Phase-1 centred
  FD already supplies them. Strategy narrative unchanged (`bssn_port_plan.md`).
