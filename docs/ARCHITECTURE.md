# Integer-Schemes — Architecture & Decisions

A reference for *why* the code is shaped the way it is. The end goal is a
BSSN binary-black-hole (BBH) numerical-relativity solver; the current code is a
Maxwell-Chern-Simons (MCS) PDE solver used to build and validate the
infrastructure (high-order finite differences, integer/RNS arithmetic on tensor
cores, and block-structured AMR) that a BBH code needs.

This document is the durable record of architecture decisions and the
problem-solving "tools" the code uses. Implementation details live in the code;
this captures the *reasoning* that the code alone doesn't show.

---

## 1. The physics being solved

**Maxwell-Chern-Simons in 2.5D and 3D** — Maxwell's equations plus a
Chern-Simons (axion-like) coupling, written as a first-order hyperbolic system
in 10 fields: `Ex Ey Ez Bx By Bz xi Pi Psi Phi` (EM fields, the CS scalar `xi`
and its momentum `Pi`, and two constraint-damping potentials `Psi`, `Phi`).

- **Constraints:** `divE` (with a CS correction term) and `divB` must stay ≈ 0.
  Constraint *damping* (`K1`, `K2`) actively suppresses violations.
- **Analytic test solution:** a birefringent (circularly-polarised) plane wave
  with periodic BC — `Ez = E0 sin(kx·x + ky·y − ω t)` — used as the oracle for
  accuracy tests.
- **Known physical instability (Carroll-Field-Jackiw):** the dispersion has a
  tachyonic branch `ω² = k² − m_cs·k` (with `m_cs = 2Λ`), unstable for `k < m_cs`.
  The default `Lambda` was set to **0.4** so the birefringent mode sits in the
  stable regime (`k ≈ 0.89 > m_cs = 0.8`); larger Λ intentionally excites the
  instability. Documented in `params.toml`.

Why MCS as a stepping stone: it is a constrained hyperbolic system with an
analytic solution and a real physical instability — a faithful, cheaper rehearsal
of the numerical challenges of BSSN, without the tensor bookkeeping.

---

## 2. Repository structure

Monorepo of three pip-installable packages (install: `pip install -e ./common ./2D ./3D`):

```
common/   mcs_common  — shared utilities (wave_state, bfp48, jax_config, ioxdmf)
2D/       mcs2d       — primary, fully-developed 2.5D solver + AMR
3D/       mcs3d       — 3D solver (AMR port is Phase 4, not yet done)
```

Each project: `src/<pkg>/` + `tests/{unit,integration,regression}/` with a
`run.sh` orchestrator. Tests work both under `pytest` and as standalone scripts.

**Decision — fully independent 2D/3D projects, shared `common`:** chosen so the
2D and 3D codebases can diverge (3D Ozaki drops to smaller block sizes for SMEM
reasons, etc.) while genuinely-shared utilities live in one place. The PDE/AMR
kernels are *not* shared — their loop bodies differ enough (field count,
stencils, dimensionality) that sharing would be more coupling than reuse.

---

## 3. Core numerical tools

### 3.1 High-order finite differences
- **6th-order centred stencils** for 1st (`C1`) and 2nd (`C2`) derivatives
  (7-point, reach ±3 → ghost width `NG = 3`).
- **Kreiss-Oliger dissipation** (`CKO`) damps grid-scale (k≈π/Δx) modes.
  - **The sign is load-bearing.** `CKO` is the standard 6th central difference
    `[1,−6,15,−20,15,−6,1]/64`. A previous **negated** form made KO
    *anti-dissipative* — it amplified the Nyquist mode (stronger σ → faster
    blow-up), which manifested as a "mysterious" grid-scale instability that
    destroyed long runs. Fixed across all 5 schemes.
  - **Order deferral:** a 6th-order scheme formally wants 8th-order KO (9-pt,
    `NG=4`) so dissipation error stays below truncation error. We keep 6th-order
    KO (`NG=3`) — empirically clean 6th-order accuracy at current resolutions —
    and documented that, if a BBH convergence study ever shows dissipation is the
    floor, the upgrade should be applied *only to the FP path* (the SMEM-bound
    Ozaki/Pallas path should stay `NG=3`).

### 3.2 Time integration
- **RK4** (4th-order), verified clean 4th-order by self-convergence.
- **CFL:** `dt = cfl·dx`, `cfl ≈ 0.05` in 2D.

### 3.3 The numerical schemes (2D)
All compute the *same* PDE RHS; they differ in arithmetic/data-movement:

| scheme | what it does | when |
|---|---|---|
| `floating_point` | FP64 finite differences via XLA | reference / debugging |
| `ozaki` | INT8 Ozaki RNS decomposition on tensor cores | GPU production |
| `fused_floating_point` | tiled FP64, one HBM pass per tile | throughput, no quantisation |
| `fused_ozaki` | Ozaki + fused Pallas/Triton kernel | max throughput |
| `pallas_ozaki` | single Pallas kernel/tile, CRT in shared memory | no L2 round-trips |

**Ozaki / RNS tool:** large-dynamic-range integer arithmetic is done in a
Residue Number System (a set of coprime moduli) so multiply-accumulates run on
**INT8 tensor cores**, with Chinese-Remainder/Garner reconstruction back to
FP64. This is the path to using the GPU's fastest (integer) compute units for a
PDE solver while controlling round-off.

**Pallas/Triton tool:** custom GPU kernels that keep a tile resident in shared
memory across the CRT pipeline, eliminating L2 round-trips between reconstruction
levels. SMEM capacity is the binding constraint here and drives block-size and
ghost-width decisions.

**JAX version pin (`==0.8.1`):** the version where the `pallas_ozaki` Pallas-Triton
kernel works.  **Confirmed on GPU that JAX >= 0.9** tightened the Triton lowering
(power-of-2 tensor sizes, stricter indexing) and breaks that kernel — 0.9 was at
best marginally faster for our workload, so we stay on 0.8.1.  Only `pallas_ozaki`
is affected (it's the sole Pallas/Triton scheme); AMR and every other scheme are
pure JAX.  A 0.9+ `pallas_ozaki` rewrite is Phase-7 work.

---

## 4. AMR architecture (the centerpiece)

Block-structured (Berger-Oliger style) adaptive mesh refinement. The whole
design is organised around **one hard constraint imposed by JAX**:

> **Shape-stable JIT.** Recompilation is expensive (especially for Ozaki). The
> compiled step function must never re-trace as the grid adapts. Therefore all
> array shapes are *fixed for the entire run*; refinement changes *which slots
> are active*, never the array sizes.

Everything below follows from that constraint.

### 4.1 Split: JAX-side fixed arrays vs host-side bookkeeping
- **`AMRState`** (JAX pytree): `blocks` of fixed shape
  `(LEVELS, MAX_BLOCKS, NF, BS+2·NG, BS+2·NG)` and a boolean `active` mask
  `(LEVELS, MAX_BLOCKS)`. Refinement flips `active` bits and writes into slots;
  shapes never change → no recompile.
- **`AMRTopology`** (host-side, plain NumPy/dicts): parent/child links, bbox
  positions, neighbour maps, hysteresis streak counters. Mutated freely on the
  host; JAX never sees it, so it can't cause a recompile.
- **`AMRTopologyArrays`** (NamedTuple of fixed-shape JAX arrays): a *snapshot*
  of the topology (`parent_slot`, `child_cx`, `child_cy`) produced by
  `AMRTopology.to_jax_arrays()`. Passed to step functions as **runtime
  arguments** — their *values* change after each regrid but their *shape* is
  constant, so the compiled step is reused. This is the linchpin that makes
  "regrid without recompile" work, and it is machine-checked
  (`test_no_recompile.py`).

### 4.2 Block geometry & conventions
- `BS = 32` (2D) interior cells per block; `NG = 3` ghost cells per side.
- **Cell-centred refinement:** a coarse cell refines into two fine cells at
  ±0.25 of the coarse spacing. (This convention, vs vertex-centred, was a source
  of early bugs — it is now consistent across prolongation, restriction, and the
  oracle comparisons.)
- **Refinement ratio 2** per axis → 4 children per parent in 2D.

### 4.3 The per-block kernels (jitted, fixed-shape)
- **Prolongation** (`prolongate`): parent → full child block (interior + halo),
  6th-order Lagrange interpolation. Produces the halo too, so it doubles as the
  cross-level ghost fill.
- **Restriction** (`restrict_into_parent`, `restrict_all_into_parents`): 2×2
  averaging of fine cells into the parent (2nd-order). `restrict_all_*` is the
  vmapped multi-slot version used in the time loop.
- **Ghost sync:**
  - within-level, root, periodic (`sync_ghosts_within_level_root_periodic`):
    pad-and-extract on the stitched global grid.
  - across-level (`sync_ghosts_across_levels`): fill a fine block's halo from a
    (time-interpolated) prolongation of its parent; masked by `active`.

### 4.4 Regridding (Phase 2)
Host-driven, JAX-shape-stable:
- **Indicator** (`compute_indicator_gradient`): per-block max ‖∇Ez‖. Note it is
  *resolution-independent*, so it does not self-limit refinement depth — hence:
- **`max_level` depth cap:** bounds the slot budget (a BBH control). Without it
  a feature refines to the deepest level everywhere it lives.
- **Hysteresis** (`compute_flags`): per-block streak counters; only flip after
  K consecutive cycles past threshold — prevents thrashing.
- **Nesting buffer** (`enforce_nesting_buffer`): dilate REFINE flags to
  neighbours (so a moving feature stays inside the refined region between
  regrids) and forbid coarsening a block with active children (no orphans).
- **`apply_flags`:** realises flags — allocate child slots + prolongate (refine);
  restrict + free slots (coarsen); update topology dicts. Raises a clear error
  on slot-budget exhaustion.
- **`evolve_with_regrid`:** time loop that runs `regrid_every` steps under one
  jitted `lax.scan`, regrids on the host, repeats — reusing the compiled step.

### 4.5 Time stepping (Phases 1–3)
Increasing scope, all reusing the per-block RHS kernel:
- **`make_root_step`** — level 0 only.
- **`make_n_level_step`** — full hierarchy, **shared dt** (all levels at the
  finest dt). Simple; wastes root work at depth.
- **`make_subcycled_two_level_step`** / **`make_subcycled_n_level_step`** —
  **Berger-Oliger sub-cycling**: level L uses `dt/2^L` and takes 2 substeps per
  parent step, so each level runs at its own CFL. The finest level takes
  `2^(LEVELS−1)` substeps; the recursion is unrolled at trace time (static →
  one compile).

**Time-interpolation at coarse-fine boundaries — cubic Hermite (4th-order):**
when a fine level sub-steps between coarse times, its ghost cells need coarse
data at intermediate times. We interpolate with cubic Hermite using the coarse
endpoint *values and time-derivatives* (the RHS, already computed — `k1` of the
coarse step, reused from the next step's `k1`). 

- **Why Hermite over linear:** linear is 2nd-order and would cap the boundary at
  O(dt²) regardless of RK4. Hermite is O(dt⁴), matching RK4. Crucially the
  *runtime* cost is ~equal (the interpolation touches only the thin halo; the
  derivatives are reused RHS values), so "linear is faster" is illusory — at
  fixed accuracy Hermite is faster because it needs fewer blocks/buffers.
- **(value, derivative) pairs thread down the recursion** so every level's halo
  — interior and ghosts — is 4th-order in time (`make_subcycled_n_level_step`).

### 4.6 What sets the AMR accuracy floor (measured)
- RK4 + Hermite time integration is **clean 4th-order** (verified on root-only
  self-convergence: order 4.00).
- But the **sub-cycled fine block** is limited not by time but by the **spatial
  coarse-fine boundary**: 2nd-order restriction + prolongation of
  coarse-resolution halo data, ~1e-6 in the smooth-wave test, far above the
  temporal truncation (~1e-11). So the next accuracy lever for BBH is
  **higher-order restriction / conservative flux correction**, not the time
  integrator. (This is why the plan's aspirational "1e-9 / slope −6" is
  boundary-limited.)

---

## 5. Testing strategy

Three tiers (`2D/tests/{unit,integration,regression}`):
- **unit** — per-kernel correctness at fixed shape (prolongation exactness,
  restriction averaging, indicator/flag logic, topology bookkeeping).
- **integration** — multi-step evolution vs the analytic birefringent oracle or
  a non-AMR reference; constraint conservation; refinement tracking.
- **regression** — invariant guards: **no-recompile** (the load-bearing AMR
  promise, machine-checked across refine+coarsen events), **bit-identical AMR vs
  `fused_floating_point`**, **long-term stability** (light-crossing-time scaled),
  **temporal convergence**.

Key recurring test ideas:
- Compare against the **analytic oracle** where possible; else against a
  **shared-dt or single-level reference** (avoids oracle position-convention
  ambiguity).
- **Self-convergence with exact integer step counts** — non-integer `T/dt`
  ends runs at different physical times and manufactures a spurious error that
  swamps the truncation error (a real bug we hit and fixed in the probes).
- **Stability targets in light-crossing times**, not fixed step counts, so they
  stay meaningful as parameters change — the proxy for "would this survive a BBH
  run."

---

## 6. Hard-won lessons (don't re-learn these)

1. **KO dissipation sign is load-bearing** — a negated stencil is
   anti-dissipative and silently destroys long runs. (§3.1)
2. **The birefringent blow-up at Λ=2 was physics, not a bug** — the CFJ tachyon.
   Default Λ lowered to 0.4. (§1)
3. **Shape-stable JIT is the master constraint** for AMR — fixed array shapes +
   active mask + runtime topology arrays = regrid without recompile. (§4.1)
4. **The gradient indicator doesn't self-limit depth** — needs `max_level` (or,
   later, a truncation-error estimator) to bound the budget. (§4.4)
5. **Naive restriction degrades constraints without flux correction** — it's
   opt-in until Phase 3's conservative coupling lands; the coarse-fine boundary
   is the AMR accuracy floor. (§4.6)
6. **Convergence probes need exact integer step counts** and a same-scheme
   reference, or you measure artifacts. (§5)

---

## 7. Status & roadmap (see AMR_PLAN.md for detail)

- **Phase 1 (static AMR foundation):** ✅ complete.
- **Phase 2 (regridding, N-level, no-recompile):** ✅ complete.
- **Phase 3 (sub-cycling in time):** ✅ Hermite Berger-Oliger sub-cycling
  (2-level and recursive N-level), validated and stable. Remaining Phase-3-class
  work for production AMR: **conservative cross-level flux correction** (the
  accuracy floor), within-level sync for non-root levels with multiple blocks.
- **Phase 4 (3D port):** not started — the actual BBH target dimensionality.
