# Step 3.2 — Phase 1 (CPU-only, today): associative reordering + derivative fusion

> **Naming:** "Phase 1 / Phase 2" here are the **CPU-today / H200-tomorrow split** of
> the current Step-3.2 push (the register-resident Pallas BSSN RHS). They are *not* the
> project's canonical Phases 1–2 (3D foundation / BSSN correctness, long done). H200 is
> unavailable today, so everything here is fully CPU-developable; the GPU confirmations
> live in `step_3.2_phase2_h200.md`.

## Anchor — what we know going in
- ✅ **fp32 collapses the algebra spill ~11×** (A100, 2026-06-12: fp64 168 reg / 24232 B
  → fp32 255 reg / **2112 B**). The register math held: the ~100–150-value BSSN working
  set is 2 regs/value in fp64 (overflows 255) but 1 reg/value in fp32 (nearly fits).
- ✅ **Physics-safe:** `precision="fp32_contraction"` (fp64 FD → fp32 algebra → fp64 RK4
  accum) — single-eval err ~1e-6, gauge-wave H/M constraints identical to fp64 (ratio
  1.000), robust stability bounded. `test_bssn_mixed_precision.py`.
- ❓ **Residual 2 KB spill** — likely L2-resident (not HBM); confirmed only by the H200
  `--smi` regime read (Phase 2).
- ⬜ **Two CPU-developable levers remain:** (1.1) associative reordering → push spill→0;
  (1.2) derivative fusion → kill wall 2 (the 138-array read), the actual remaining
  bottleneck (fp32 moved us from spill-dominated 0.17 FLOP/B to deriv-read-dominated
  ~1.7 FLOP/B). Literature grounding: NR-GPU register pressure is an acknowledged,
  largely-unsolved problem for full Einstein (arXiv:2501.14030); associative reordering
  for register pressure is a known technique (Rawat et al. SC18).

---

## 1.1 — Associative reordering (the spill→0 lever)

**Why it can work where 3.2b's schedule didn't:** `ptxas` freely reorders *independent*
instructions (so our 3.2b materialize/recompute washed out — `ptxas` re-derived it), **but
it cannot reassociate floating-point** — that changes rounding, which `ptxas` must
preserve. So a **reassociated computation tree is binding**: `ptxas` can't undo it, and a
lower-Strahler association genuinely lowers the achievable peak liveness.

**Do the cheap prediction FIRST (decisive, free):**
1. Add a **straight-line liveness model** to `staging.py`:
   `straight_line_liveness(order, dtype)` → peak count of simultaneously-live values for a
   given evaluation order, counting fp64 = 2 regs, fp32 = 1 reg. (Distinct from the 3.2b
   `persistent_liveness`, which modelled a materialize/recompute partition.)
2. Add a **min-liveness list scheduler** (greedy: at each step emit the ready node that
   most reduces the live set / whose result is consumed soonest) + **stream-accumulation
   of the wide associative sums** (the contractions — accumulate into one register, feed
   inputs one at a time, instead of holding all addends live).
3. **Predict** the reordered peak liveness in fp32 vs the 255-register budget:
   - Drops comfortably < 255 fp32 → the lever is real → emit + validate (below), confirm
     on H200 tomorrow.
   - Stuck near the structural contraction width → reordering is **dead** (the width is
     irreducible); stop, having spent zero GPU time, and put all weight on 1.2.

**If viable — emit + validate (CPU):**
- Emit a reordered algebra module (`_bssn_rhs_reordered.py` or a `staging` emission mode),
  wired as a `scheme`/flag so it A/Bs against verbatim.
- **Correctness:** `== verbatim` to round-off (reassociation changes fp summation order →
  ~1e-12, not bit-identical). Reuse the oracle single-point + gridded harness.
- **Physics:** run it through the existing `test_bssn_mixed_precision.py` accuracy +
  constraint gate (reassociation + fp32 together must still preserve constraints).

**Deliverable:** a yes/no on spill→0 viability (from the CPU prediction) and, if yes, a
validated reordered kernel ready for the H200 spill A/B (2.2).

---

## 1.2 — Derivative fusion (wall 2, the bigger win)

fp32 left the **138-array derivative read** as the bottleneck (~1.7 FLOP/B). Fusing the
derivative computation on-chip removes it → intensity ~16 FLOP/B → compute-bound.

**Design (the field-streamed 2.5D-SMEM derivative stage; see CLAUDE.md "3D BSSN GPU kernel
architecture" + HANDOFF §4):**
- Stream fields **one at a time**; load each field's halo into SMEM; compute its
  `grad_*`/`grad2_*` on-chip; feed straight into the (fp32) algebra. Never materialise the
  138 arrays to HBM.
- Temporal fusion stays **rejected** for 24-field 3D (halo-redundancy explosion); this is
  spatial halo reuse only.

**Build + validate (CPU, Pallas interpret mode):**
- Prototype the fused kernel (derivative-on-chip + algebra in one `pallas_call`).
- **Interpret-mode correctness:** the fused derivatives reproduce `derivative_bundle`, and
  the full fused RHS matches the verbatim RHS to round-off, on a small grid.
- ⚠️ Interpret mode validates **math only**, not compilation/registers/SMEM occupancy —
  the spill + regime payoff is **H200-gated** (Phase 2.3). Expect a heavier kernel
  (couples the SMEM/halo pool with the register pool); the ~1600 s Pallas compile wall and
  the `OZAKI_GPU_NOTES.md` Triton 0.9.2 constraints (no `dynamic_slice`/`gather`; crop via
  one-hot) apply.

**Deliverable:** an interpret-validated fused kernel, ready for the H200 spill/regime
measurement.

---

## Recommended order today
1. **1.1 prediction** (cheap, decisive, the lever the user likes) → yes/no on spill→0
   *before* any GPU time.
2. **1.2 build** (meatier; H200-gated payoff) while 1.1's verdict settles.

## Phase-1 exit criteria
- ✅ Straight-line liveness model + min-liveness/reassociation pass in `staging.py`, with a
  recorded **predicted peak-liveness (fp32)** and the spill→0 viability verdict.
- ✅ If viable: reordered kernel emitted, `== verbatim` to round-off, constraint gate green.
  *(N/A — verdict was DEAD; no kernel emitted.)*
- ✅ Fused derivative kernel prototyped + interpret-mode-validated (math) vs the FD bundle.
- → Hand the H200-gated confirmations to `step_3.2_phase2_h200.md` (tomorrow).

---

## RESULTS (2026-06-12, CPU)

### 1.1 — Reassociation: **DEAD** (decisive, zero GPU time)
`staging.py` gained the prediction model (`straight_line_liveness`, `min_liveness_order`,
`reassociation_floor`, `predict_reassociation`; reported by `python -m bssn3d.staging`;
`tests/unit/test_bssn_reassoc.py`, 5 green). The numbers (fp32, 255-reg budget):

| quantity | value |
|---|---|
| temps | 826 (**757 multi-use / 69 single-use**) |
| store-everything peak live, file order | **453** |
| store-everything, greedy min-liveness reorder (ptxas-achievable) | **424** |
| **reassociation FLOOR (multi-use peak)** | **432 fp32 / 864 fp64** |

The key fact: ptxas reorders independent ops (so the min-liveness reorder, 453→424, is a
gain it *already* gets) but **cannot reassociate fp**, so reassociation would be binding —
yet it can only collapse **single-use** reduction chains, and only 69 of 826 temps are
single-use. The pressure is the **757 genuinely-shared** tensor-hierarchy temps, ~all
co-resident entering the output block (peak hits at `a_rhs`, statement 826). That floor is
**432 ≫ 255 even in fp32**, invariant under reassociation. **Verdict: reordering is dead;
all weight on 1.2.** (The fp32 near-fit measured on A100 — 2 KB residual — comes from
ptxas's *own* remat/recompute, which reassociation cannot improve and which is the
3.2b lever that already washed out through ptxas.)

### 1.2 — Fused derivative kernel: prototyped + interpret-validated
`fused_backend.py` emits `_bssn_rhs_fused.py` — **one `pallas_call`** that loads the 24
fields, computes all 138 `grad_*`/`grad2_*` **on-chip** (field-streamed, via stencil
coeffs copied as plain floats from the shared `SpatialDerivative` — no captured jax
arrays), then runs the 3.2b scheduled algebra; the 138 derivative arrays are **never
materialized to HBM**. Wired as `scheme="fused"` (`rhs.py` takes the fields+spacings path,
no `derivative_bundle`); `profile_regime`/`spill_probe` accept it.
`tests/integration/test_bssn_fused.py` (2 green, interpret mode):
- on-chip derivatives reproduce `derivative_bundle` **bit-for-bit** (`max |Δ| = 0`);
- fused RHS `== verbatim` to round-off (gridded gauge wave, atol 1e-11 / rtol 1e-9).

⚠️ Interpret mode validates **math only**. The `jnp.pad`/static-slice stencil does **not**
lower under Triton — the SMEM/halo rewrite + occupancy payoff is the H200 build
(`step_3.2_phase2_h200.md` §2.3). Full suite: **75 passed** (4 slow deselected).
