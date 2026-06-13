# Phase 1 — 3D foundation (MCS in 3D): implementation plan

**Status: 🔵 ACTIVE.** The infrastructure every later phase lands on. CPU-developable;
no GPU on the critical path (benchmark/roofline excepted). Exit = a **validated** 3D
MCS with the full 2D machinery ported, so BSSN (Phase 2) is an RHS swap onto it.

---

## 1. Honest current-state assessment (what already exists)

`3D/src/mcs3d/` is **not bare** — it is a functionally complete 3D MCS solver that
has simply **never been validated or characterized**. Audited 2026-06-11:

| Piece | State | File |
|---|---|---|
| 3D state (10 fields, `WaveState`) | ✅ exists | `common/.../wave_state.py` |
| 6th-order 3D FD (d1, d2, KO), generalized to order 4/6/8 | ✅ exists, **KO sign correct** (`[1,-6,15,-20,15,-6,1]/64`) | `mcs3d/schemes/floating_point.py` |
| Full 3D MCS RHS (CS coupling, constraint damping K1/K2) | ✅ exists | `mcs3d/main.py:252` |
| BCs: periodic (exact), Sommerfeld, zero | ✅ exists | `mcs3d/main.py:206–250` |
| RK4 stepper | ✅ exists | `mcs3d/main.py:312` |
| 3D birefringent oracle + Gaussian ID | ✅ exists (debugged — "FIXED" comments) | `mcs3d/main.py:76` |
| Constraint diagnostics (divE, divB) | ✅ exists | `mcs3d/main.py:323` |
| 3D Ozaki scheme (unfused) | ⚠️ exists (281 lines), **unverified** | `mcs3d/schemes/ozaki.py` |
| **Validation harness (`validate.py`)** | ❌ **not ported** | — |
| **Benchmark / roofline** | ❌ **not ported** | — |
| **Test suite** | ❌ **empty** (`3D/tests/{unit,integration,regression}` are empty; only `conftest.py`) | — |

**Conclusion:** the physics is *plausibly* there but **unproven**. Phase 1's real
work is to *prove it* (port `validate.py` → 3D, build the test suite) and port the
characterization tooling — not to write a 3D RHS.

### Two correctness questions the audit surfaced (resolve in 1.3)
1. **Dispersion branch.** The 3D oracle uses `ω = √(k² + m_cs·k)` (the always-stable
   branch), whereas the 2D CFJ analysis documents `ω² = k² − m_cs·k` (the tachyonic
   branch unstable for `k < m_cs`). One circular polarization takes `+`, the other
   `−`. The 3D semi-discrete symbol (1.3) must confirm the oracle's branch/sign is
   *consistent with the RHS* — this is exactly what the 2D `_symbol` vs oracle check
   does, and is the decisive correctness test.
2. **Polarization-triad singularity.** `_birefringent_wave_3d` builds the triad with
   `norm_factor = √(k_x²+k_y²)` (singular if `k_x=k_y=0`). Default `k` has all
   components nonzero; note the constraint and guard if a test picks an axis-aligned `k`.

---

## 2. Steps

### Step 1.1 — Audit & lock the 3D derivative operators + RHS
The operators BSSN will consume. Mostly verification, since they exist.
- Confirm `SpatialDerivative` (3D) matches the validated 2D stencils bit-for-bit
  (C1, C2, CKO) at order 6; keep the order 4/8 generality (BSSN/8th-KO want it).
- Confirm the RHS matches the validated 2D physics term-by-term (CS coupling signs,
  `Pi₀` background role, constraint-damping `−K·Ψ`/`−K·Φ`). The `_d1_batched` vmap
  pattern (one `compute_d1` call per axis over the 9 differentiated fields) is the
  3D analog of the 2D batching — keep it (it is the shape the Ozaki/staged path needs).
- **No new physics.** Output: a one-page term-by-term 2D↔3D correspondence note +
  unit tests asserting stencil/RHS equality on shared slices.

### Step 1.2 — Verify the 3D BCs
- **Periodic:** ghost wrap is exact; energy conserved to round-off on a periodic
  pure-EM run (mirror `tests/validation/test_boundaries.py::TestPeriodic`).
- **Sommerfeld:** passive (no energy injection), absorbs an outgoing pulse (mirror
  `TestSommerfeld`). Note the existing radial `calc_bc` form; confirm it is passive.
- Light step — the BCs exist; this is guard tests, not new code.

### Step 1.3 — Port `validate.py` → 3D (the bulk of Phase 1)
Mirror `2D/src/mcs2d/validate.py` (777 lines) into `3D/src/mcs3d/validate.py`. The
2D design — *one symbol, many views*, cross-checked to the AD Jacobian — carries
directly; the only change is `kx,ky → kx,ky,kz` (the symbol stays **10×10**, NF=10).

Port, in dependency order:
1. `FullBirefringentOracle` (3D) — reuse the solver's `_birefringent_wave_3d`; wrap
   for `state(X,Y,Z,t)` so error-vs-time is computable.
2. `_symbol(kx,ky,kz, p)` — the linearized semi-discrete MCS operator with the 3D FD
   modified wavenumbers (`fd_symbol_d1/d2` per axis) + KO; **linearize at `Pi=Pi₀`**
   (not `u=0` — at `u=0` the CS coupling vanishes, hiding the physics; this was a
   2D lesson).
3. `_build_jacobian` (3D, AD) — cross-check `_symbol` to ~1e-8 on a small grid (the
   ground-truth that the hand-built symbol is right). **This is the decisive test.**
4. `convergence_study` — vs the 3D oracle, **at fixed physical time with exact
   integer step counts** (non-integer T/dt manufactures spurious error — a 2D lesson).
   Expect **6th order** (KO off) and **~5th** (production 6th-order KO σ=0.05).
5. `spectral_analysis` / `spectrum_table` — no growing modes (max Re ≤ continuum +
   round-off), KO only damps, dispersion is 6th-order, group velocity at low k.
6. `principal_symbol_condition` — **strong hyperbolicity** over `n_dirs` 3D directions.
7. Constraint-damping rate check (rate is **K/2**, underdamped Gundlach — a 2D fact).
- Figures/CSV to `3D/.../step_1.1_results/` (new), via the same `--replot` pattern.

### Step 1.4 — Port benchmark/roofline to 3D
- Mirror `2D/src/mcs2d/benchmark.py` (gpu_info/H200 table, `_step_cost`, roofline).
- Lower priority / GPU-gated: the **regime numbers need the H200** (your 2FA). The
  CPU-side plumbing + a mocked-device unit test land now; the real run batches with
  other GPU work. (MCS 3D is still expected DRAM-bound — this is characterization,
  not the efficiency story, which is BSSN.)

### (Throughout) Build the 3D test suite
`3D/tests/` is empty. Mirror the 2D `tests/validation/` four files against the 3D
solver: `test_convergence.py`, `test_spectrum.py`, `test_boundaries.py`,
`test_constraints.py`. Plus a minimal `unit/` (stencil/RHS equality vs 2D) and an
`integration/` oracle-evolution test. Wire `3D/tests/run.sh` like the 2D `run.sh`.

---

## 3. Reuse vs build

- **Reuse (port, don't reinvent):** the entire `validate.py` structure, the
  benchmark plumbing, the test patterns, `WaveState`, `ioxdmf`. The 2D code is the
  validated template; 3D is `+kz` and `axis=2`.
- **Build:** the 3D `_symbol`/`_build_jacobian` (genuinely new — 3D Brillouin zone),
  the 3D oracle wrapper, the 3D test suite, the benchmark device-cost in 3D.
- **Audit (verify, then lock):** the existing 3D RHS / derivatives / BCs / oracle.

---

## 4. Risks / gotchas (mostly pre-known from 2D)

1. **KO sign is load-bearing** — already correct in 3D (`+δ⁶/64`); do not "fix" it.
2. **Linearize the symbol at `Pi=Pi₀`, not `u=0`** — else the CS coupling vanishes
   and the symbol misses the real physics (birefringence, CFJ).
3. **Convergence needs fixed physical time + exact integer step counts** — fixed
   step counts make `T ∝ dx` and inflate the apparent order by ~1.
4. **Dispersion branch / triad singularity** (see §1) — the symbol-vs-oracle check
   resolves the branch; guard the triad if a test picks axis-aligned `k`.
5. **`_apply` edge-pads then slices the full (incl-ghost) array** — same convention
   as 2D; the BC functions own the ghost truth. Keep consistent.
6. **Stay on explicit-6 (+ generality to 8).** Do NOT wire compact FD here — it
   diverges from the Dendro-GR explicit-FD oracle (Phase 2 bit-compare) and reopens
   boundaries. Compact is a parked Phase-4 lever (see tracker).
7. **3D cost is real** — `(N+2·NG)³` arrays; keep validation grids modest
   (N≈24–96) so CPU runs stay quick.

---

## 5. Exit criteria

- ✅ 3D `_symbol` matches the AD Jacobian to ~1e-8 (decisive correctness gate).
- ✅ Convergence vs the 3D oracle: **6th order** (KO off), **~5th** (production KO),
  at fixed physical time / integer steps.
- ✅ Spectrum: no growing modes beyond continuum/CFJ; KO only damps; strong
  hyperbolicity holds over 3D directions.
- ✅ Constraints: divE/divB damp at rate ~K/2; periodic energy conserved to round-off;
  Sommerfeld passive.
- ✅ 3D test suite green (validation mirror + unit + integration); `run.sh` wired.
- ✅ 3D benchmark/roofline on the H200 (2026-06-11): **memory-bound confirmed**
  (RHS ~0.25 FLOP/byte cost-model, achieved ~354 GFLOP/s ≈ **1% of FP64 peak** —
  the ALUs idle on memory), matching the 2D regime. MCS stays the correctness
  ground; the efficiency story is BSSN. See §7.

---

## 6. Suggested order of work
1.1 audit (fast, unblocks confidence) → 1.3 `_symbol` + AD-Jacobian cross-check (the
decisive gate, do early) → 1.3 convergence + spectrum + the test suite → 1.2 BC
guards → 1.4 benchmark plumbing (GPU run batched). Validation harness and tests grow
together (each `validate.py` view gets a test).

## 7. Validation results (2026-06-11) — exit criteria MET

The harness (`3D/src/mcs3d/validate.py`, ported from 2D) + the full test suite
(`3D/tests/{unit,integration,validation}/`) are built and green. Both correctness
questions the audit flagged in §1 are **resolved positively**:

- **Symbol ↔ AD-Jacobian:** the hand-derived 3D semi-discrete symbol matches the
  exact AD Jacobian of the real RHS to **2.4e-8** (the decisive gate). The 3D
  oracle's dispersion branch `ω=√(k²+m_cs·k)` is therefore confirmed *consistent
  with the RHS*.
- **3D birefringent oracle is an exact solution:** stencil convergence (KO off,
  fixed physical time) is **5.94 ≈ 6th order** in all six EM fields — the oracle
  and the triad are correct (the polarization triad's E·B = 0 keeps Pi at Pi₀ and
  xi growing linearly, as constructed).

Certificates (from `validation_summary.txt`):
- Convergence: stencil **5.94**; production KO-on **5.68–5.79** (KO-limited 5th,
  as predicted); RK4 temporal floor (CFL=0.4) **4.13**.
- Spectrum: strong-hyperbolicity condition number **4.81** (bounded → strongly
  hyperbolic over the 3D direction sphere); constraint damping lands on **−K/2** to
  1e-6; CFJ tachyon present below m_cs and absent above (**1 / 0** growing modes);
  KO only damps; von-Neumann |G|≤1 at the operating CFL in the stable regime.
- Long run (Λ=0.2): L2(Ez−oracle) **3.5e-6**, L2(divB) **6.6e-16** (round-off),
  bounded amplitude.

> The summary's `max RK4 |G|=1.0055` is at Λ=0.4, where the **physical** CFJ
> tachyon (max Re λ ≈ 0.33) lives — not a numerical defect (the von-Neumann *test*
> uses the CFJ-stable Λ=0.2). Mirrors the 2D harness convention exactly.

**Unit-level audit (Step 1.1):** the 3D `SpatialDerivative` is bit-identical to the
validated 2D operator at order 6 (orders 4/8 carry textbook coefficients); the 3D
RHS reduces **exactly** (≤1e-12) to the 2D RHS on z-invariant data — the
term-by-term 2D↔3D correspondence, machine-checked (`tests/unit/test_derivatives.py`).

**Test suite:** 41 tests (39 fast + 2 `@slow`) across `unit/` (derivatives,
benchmark plumbing), `integration/` (oracle evolution + bounded/constraint-clean
guards), and `validation/` (convergence, spectrum, boundaries, constraints).
`run.sh` wired with a `validation` category. Deliverable figures + summary in
`step_1.1_results/`. (The CFJ tachyon is certified spectrally in
`test_spectrum.py`, not dynamically — the birefringent IC sits on the stable
dispersion branch, so a dynamical growth test would be ill-founded; see the note in
`test_simulation.py`.)

**Step 1.4 (benchmark/roofline):** `3D/src/mcs3d/benchmark.py` ported (single
`floating_point` scheme; cube sweep; HLO cost-model + roofline). CPU plumbing
validated by `tests/unit/test_benchmark.py`. **H200 run done (2026-06-11):** real
device, `gpu` backend; 64³ + 128³.

| N | per-step | throughput | RHS FLOP/byte | bound | achieved GFLOP/s | L2 vs oracle |
|---|---|---|---|---|---|---|
| 64 | 6.9 ms | 38 Mpts/s | 0.25 | memory | 354 | 1.0e-9 |
| 128 | 38.6 ms | 54 Mpts/s | 0.28 | memory | 308 | 1.5e-11 |

The decisive number: achieved **~354 GFLOP/s vs the 34 TFLOP/s FP64 peak ≈ 1% of
compute** — the unambiguous memory-bound signature (ALUs idle on memory). RHS
intensity ~0.25 FLOP/byte sits ~28× left of the FP64 ridge (7.1), ~1600× left of
the INT8 ridge (412) → tensor cores have nothing to bite on. **Confirms the
strategy: 3D MCS is the correctness/reproducibility ground; the efficiency story
is BSSN (≈62 FLOP/byte).** (Caveats: the cost-model FLOP/byte is a lower bound — it
over-counts the unfused RHS's intermediate traffic, hence < the ~3.2 quoted for 2D;
`%peakBW ≈ 24%` is an upper bound, not DRAM. The trustworthy facts are the `bound`
label + the ~1%-of-FP64-peak. Same "inefficient, not saturated" picture as 2D.)
Benchmark deliverables land in `3D/src/mcs3d/output/`
(`benchmark_results.{csv,json}`, `benchmark_{throughput,roofline}.png`); the
`validate.py` figures live in `step_1.1_results/`.

## Status / changelog
- 2026-06-11 — Created. Phase 1 opened after the strategy restructure. Audit found
  the 3D solver functionally complete but unvalidated/untested → Phase 1 reframed as
  validation + machinery-port + audit, not physics-building.
- 2026-06-11 — **Steps 1.1–1.4 complete (exit criteria met; GPU benchmark run still
  pending).** Ported `validate.py` + `benchmark.py`; built the full test suite (41
  tests, all green). Symbol↔Jacobian 2.4e-8, stencil 6th-order, hyperbolicity/CFJ/
  constraint certificates all pass. The 3D MCS foundation is validated — BSSN
  (Phase 2) is now an RHS swap onto it. See §7.
