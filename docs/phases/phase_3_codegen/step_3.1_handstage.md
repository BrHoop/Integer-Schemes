# Step 3.1 — Hand-stage the worst offenders (the XLA-controllability probe)

**Status: ✅ DONE (2026-06-12) — controllability answered (see §6).** Barriers
eliminate the spill (1104 B → 0) but at +10 kernels; `remat` is a no-op on the
forward RHS → XLA can register-bound but not de-fragment; the few-kernels/compute-bound
target likely needs Pallas, to be confirmed by §3.3's regime profile of the staged RHS.
Step 3.0 is ✅ done — the production RHS is locked
to the **CAHD+SSL** variant (`bssneqs_SSL_HD_dxsq.cpp`), re-transliterated and
re-validated (oracle **1.93e-16**, apples green). Step 3.1 takes that correct RHS
and asks the one question Phase 3's emission target hinges on:

> **Can XLA be *forced* register-bounded (few kernels, no spill, intermediates
> on-chip) via `remat` / `optimization_barrier` on a hand-staged tensor-hierarchy
> pipeline — or is Pallas (explicit register control) required?**

3.1 is a **probe**, not necessarily a win. A rigorous negative ("XLA won't, Pallas
it is") is a valid, plan-advancing result — it decides 3.2's emission target.

---

## 0. Prerequisite (GPU, user 2FA): re-baseline the spill on the CAHD+SSL RHS

The headline baseline — **79 fusion kernels, 255-reg ceiling, 2536 B spill** — was
measured on the **no-CAHD** variant (Phase 2.2). Step 3.0 switched the production
RHS to CAHD+SSL, which adds two live-set-shifting terms:
- `chi_rhs` now carries the **entire Hamiltonian-constraint expression** (the CAHD
  `dx²/dt` damping) — a large new fan-in into one output.
- `a_rhs` carries the SSL term (`√chi·(√chi−α)·exp(−t²/2σ²)`) — small, but adds
  `sqrt`/`exp` ops.

So **before optimizing anything, re-run `bssn3d/spill_probe.py` on the CAHD+SSL RHS**
(it already compiles the new RHS — `rhs.py` defaults `t=0`, `dt=0.25·dx`, so the probe
is unchanged) to get the *real* target spill map. Record: kernel count, worst-register
fusions, total spill bytes, and **which output kernels** (expect `chi_rhs`/`At_rhs`/
Ricci-heavy fusions) spill. Optimize against *that*, not the stale no-CAHD map.

> This is the only GPU-gated piece of 3.1's *setup*; everything in §1–§3 is
> CPU-developable, and the final spill re-measurement (§4) is the second GPU run.

---

## 1. The central tension (be honest about what we're fighting)

XLA is failing in **two opposite directions at once** on the verbatim RHS:
- **Over-fragmented:** 79 kernels → intermediates round-trip through HBM (memory-bound).
- **Over-fused (within a few):** the big pointwise fusions exceed 255 registers → spill.

The compute-bound target is the **middle**: *few* kernels, each keeping its working
set on-chip and *under* 255 registers. The two XLA levers pull differently:

| Lever | Effect | Helps / hurts |
|---|---|---|
| `optimization_barrier(x)` | materializes `x`, blocks CSE/fusion across it | **splits** → fewer regs/kernel but more HBM traffic |
| `jax.checkpoint`/`remat` | recompute a subtree instead of holding it live | **shrinks live set inside a fused kernel** → fewer regs *without* splitting |

`remat` is the lever that could give "register-bounded fusion" (recompute the ~800
cheap leaves so the big fusion's live set drops below 255 while staying fused).
`optimization_barrier` is the lever that controls *what* fuses (pin the ~69
tensor-hierarchy temps as stage boundaries so XLA can't re-CSE them into a spill).
**The empirical question is whether ptxas register count actually responds to these
JAX-level hints** — XLA's lowering may ignore them for register allocation, in which
case only Pallas (where we own the register file) reaches the target.

---

## 2. The staged structure (what to materialize)

Mirror Dendro's tensor hierarchy (`~/Code/Dendro-GR/CodeGen/bssn.py`/`dendro.py`:
`set_metric → Christoffel → Ricci → lie`). Store the **~69 high-fan-out
tensor-hierarchy temps** as per-point arrays; recompute the cheap leaves inline:

| Stage | Materialize (store) | Why it's the choke |
|---|---|---|
| A. inverse conformal metric | `igt` (6) | needs all 6 `gt` co-resident |
| B. Christoffels | `C1`, `C2`/`Γ̃` (≈36) | feed Ricci + every covariant deriv |
| C. conformal Ricci | `Rij` (6) | **sums all 36 `grad2(gt)`** — the widest live point |
| D. CalGt / trace-split | `CalGt` (3), `tr`, traceless pieces | high fan-out into `At_rhs`/`Gt_rhs` |
| E. assemble 24 outputs | — | recompute leaves from stored tensors |

This is exactly the `[[bssn-codegen-staging]]` store-set (≈133 peak live → fits 255).
The CAHD term adds the Hamiltonian constraint into stage E's `chi_rhs` — its
sub-expressions overlap the Ricci/`igt` temps already stored, so staging should
*help* it, but verify in the re-baseline.

---

## 3. How to build it (CPU dev) — extend the codegen, don't hand-type

Hand-typing 850 SSA lines re-introduces the transliteration-typo risk Phase 2
eliminated. Instead **drive it from the committed generated module**, correct-by-
construction + regen-guarded:

1. **Fan-out analysis** (`_codegen.py` helper, pure Python): parse
   `_bssn_rhs_generated.py`, build the `DENDRO_*` dependency DAG, count downstream
   references (fan-out) and subtree-recompute cost per node. The ~69 high-fan-out
   nodes are the materialization candidates; cross-label them against Dendro's
   tensor names (`bssn.py`) for readability. (This analysis is *also* the seed for
   the 3.2 automatic cut-set generator — build it reusable.)
2. **Staged emission mode**: extend `generate()` with a `staged=True` path that wraps
   each chosen cut temp in `jax.lax.optimization_barrier(...)` and/or groups the
   leaf subtrees under `jax.checkpoint`. Emit `_bssn_rhs_staged.py` alongside the
   verbatim module (keep both; the verbatim one stays the oracle reference).
3. **Wire** a `scheme="staged"` switch in `BSSNSolver` (default stays verbatim) so
   tests/benchmarks can A/B the two with identical inputs.

Start the barrier set from the **measured spill map** (§0): pin the temps feeding the
3 worst spilling fusions first, widen to the full ~69 if needed.

---

## 4. Gates (what 3.1 must show)

- **Correctness (CPU):** staged RHS matches the verbatim RHS to **round-off**
  (~1e-12, *not* 3e-16 — recompute/reorder changes fp summation order) via a new
  `test_staged_equals_verbatim` (reuse the oracle's single-point inputs + a gridded
  diff). Apples tests (constraints ~6th, conformal algebra, robust stability) still
  pass on the staged RHS.
- **The answer (GPU, 2FA):** re-run `spill_probe` on the staged RHS. Success =
  **spill bytes drop (toward 0) and/or kernel count collapses** vs the CAHD+SSL
  baseline. Record the before/after. If barriers/remat move the needle →
  XLA-controllable, 3.2 emits staged JAX. If they don't → **Pallas is required**,
  3.2 emits a Pallas kernel (note the constraints: no fp64 `dot` — irrelevant here,
  the ~1600 s compile wall, `pallas_ozaki` gotchas).

---

## 5. Risks / gotchas (3.1-specific)

1. **`remat` may not lower to fewer ptxas registers.** XLA's register allocation can
   ignore the JAX-level recompute hint. That's the headline risk and the whole point
   of the probe — measure, don't assume.
2. **`optimization_barrier` can make it *worse*** (more kernels, more HBM) if applied
   too aggressively — it splits. Tune the cut-set from the spill map, not blindly.
3. **Round-off, not bit-identity.** The Phase-3 oracle bar is "matches verbatim to
   ~1e-12"; assert that, not 3e-16. Apples remain the physics guard.
4. **Don't drift into 3.2/Phase 4.** 3.1 hand-stages + answers the controllability
   question. The *automatic* cut-set generator is 3.2; no Ozaki/INT8/tensor-core here.
5. **Keep the verbatim RHS as the reference.** The staged module is a second artifact;
   the bit-validated verbatim one stays the oracle anchor and the regression target.

## 6. Exit criteria
- ✅ **CAHD+SSL spill re-baselined on the H200 (2026-06-12).** N=48 verbatim:
  **97 kernels, 2 spilling, 1104 B total, max 255 regs** (vs no-CAHD 79/3/2536 B —
  more fragmented, less spill; the Hamiltonian constraint in `chi_rhs` made XLA split
  rather than spill harder). Worst: `loop_add_multiply_subtract_fusion_1` (255/936 B),
  `loop_add_multiply_fusion_1` (255/168 B). XLA fusion names don't map to BSSN fields.
- ✅ **Fan-out/DAG analysis** (`bssn3d/staging.py`, reusable for 3.2): 826 temps /
  24 outputs / 4527 ops; only **69 single-use, 757 reused** (broad reuse = the spill
  cause). Top-69 store covers only ~59% of `fanout*cone` → structurally broad (Ricci),
  not a spike — corroborates the 584-live measurement.
- ✅ **Staged RHS emitted** (`_bssn_rhs_staged.py`, 69 `optimization_barrier` cuts at
  the top fan-out×cone temps; `_codegen._emit_module` shared with verbatim so the
  oracle module's bytes are unchanged). `scheme="staged"` wired into `BSSNSolver`;
  `spill_probe` honours `BSSN_SCHEME=staged`. Matches verbatim **bit-identically**
  (barriers are numerical no-ops) — `test_bssn_staged.py` 7 green; existing
  oracle/apples/RHS suite still green.
- ✅ **Controllability answered (H200, 2026-06-12).** Staged A/B (`BSSN_SCHEME=staged`)
  vs the 97-kernel/1104 B verbatim baseline:

  | variant | kernels | spilling | total spill | max regs |
  |---|---|---|---|---|
  | verbatim CAHD+SSL | 97 | 2 | 1104 B | **255** (pegged) |
  | **staged (69 barriers)** | 107 | **0** | **0 B** | **254** |

  **The barriers eliminated the spill entirely** (1104 B → 0, max regs 255→254) — so
  `ptxas` *does* respond to `optimization_barrier`: XLA **is register-controllable**
  for the forward RHS. **But it pays in fragmentation** (97 → **107** kernels): the
  barrier lever *splits*, trading spill for more HBM-round-tripping kernels (the §1
  "two opposite directions" tension — we fixed the spill axis, nudged the
  fragmentation axis the wrong way).

  **`remat` is not an available second lever here:** `jax.checkpoint` only controls
  save-vs-recompute *during autodiff*, and the RHS is evaluated **forward** (RK4, no
  grad) → it is a no-op for this jit. So `optimization_barrier` is XLA's *only*
  applicable knob, and it cannot give few-kernels-**and**-register-bound at once.

  **DECISION:** register-bound is XLA-achievable (3.2 *can* emit staged JAX, no spill)
  — but reaching the genuine Phase-3 target (**few kernels, on-chip, compute-bound**)
  past XLA's fragmentation likely needs **Pallas** (one kernel holding the live set in
  registers). The deciding datum is **§3.3's regime profile of the staged RHS**
  (`profile_regime --smi`: is 107-kernel/0-spill compute-bound, or still HBM-bound on
  inter-kernel traffic?). That measurement closes the XLA-vs-Pallas call for 3.2.

## 7. Suggested order
§0 re-baseline (1 GPU run) → §3.1 fan-out analysis (CPU) → §3.2/3.3 staged emission +
`scheme="staged"` (CPU) → §4 CPU correctness gates → §4 one GPU spill re-measure →
record the controllability decision → hand to 3.2 (automate the cut-set).
