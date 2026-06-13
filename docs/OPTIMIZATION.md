# Optimization Review — Efficiency for BBH

A critical pass over the architecture asking: **are the current choices the most
efficient ones for the eventual BSSN binary-black-hole (BBH) target?** Each
section states *what we do now*, *why*, and *what might be faster*, with an
honest verdict. This is a discrepancy report, not a to-do list — items are
flagged by expected payoff and risk so the high-value ones can be picked off.

Legend: 🟢 keep as-is · 🟡 worth doing · 🔴 likely a real win for BBH

---

## Measured GPU baseline (2026-06-03)

`profile_amr.py` run on the supercomputer GPU (depth-3 hierarchy, sub-cycled
`make_subcycled_n_level_step`), trace at
`traces/amr/plugins/profile/2026_06_03_19_35_04/`. The whole step fused into one
XLA module (570 ms wall, **480 ms GPU-busy, ~84% utilization** — genuinely
compute-bound, not launch overhead). GPU time by op family:

| Category | Time | % | What it is |
|---|---:|---:|---|
| `add` (RHS arith / RK4 axpy) | 147 ms | 30.6% | physics RHS + combine — *includes dead-slot waste* |
| `concatenate` (prolong/sync glue) | 136 ms | 28.3% | cross-level + within-level ghost sync |
| `select` (masking) | 70 ms | 14.5% | active-mask multiply over **inactive** slots |
| `dynamic_slice`/`update_slice` | 72 ms | 14.9% | per-level block read/write |
| `reduce` | 33 ms | 6.8% | KO / restriction reductions |
| `gather` | 21 ms | 4.4% | restriction indexing |
| copy/other | 2 ms | 0.5% | |

**Headline: only ~30% of GPU time is arithmetic (and even that is inflated by
dead slots); ~58% is AMR plumbing** (`concatenate` + `select` + slice/update).
This data confirms — with hardware numbers — the two 🔴 levers below:

- **Ghost-zone sync is the largest non-arithmetic cost** (`concatenate`, 28.3%),
  driven by full-block prolongation. → optimization A (halo-only prolongation).
  This *corrects* the earlier "sync is cheap relative to RHS" framing (§3): on
  GPU the sync is ~0.93× the arithmetic, i.e. **comparable, not cheap**.
- **Dead slots cost ≥14.5% directly** (`select` masking is pure inactive-slot
  tax) plus an unmeasured share of the 30.6% `add` (RHS over inactive blocks).
  → §1 compaction attacks both.

> **Update (post-implementation re-profile).** The above is the *baseline* snapshot.
> A1 footprint prolongation + fused select + calibrated-caps-default have since
> shipped; `concatenate` fell 28.5%→16.8%, calibrated caps gave ~10.5×, and the
> op-mix is now flat. Strip refinement is deprioritized and compaction deferred to
> the BBH regime. See the **Priority summary** at the bottom for the current state.

> **🧭 Memory-bound verdict (cost-model re-profile, `traces/amr/p5_*`).** Every phase
> reports **FLOP/byte ≈ 0.0–0.55** — far below the GPU FP64 roofline crossover (~10).
> The AMR step is **memory-bandwidth/latency bound, not compute bound**. Two distinct
> sub-modes (and two distinct fixes):
> - **Bandwidth-bound** (lots of bytes, big kernels): `full_step` (4.9 GB), `rhs`,
>   `sync_cross` → *move fewer bytes* (fuse stages = fewer HBM round-trips; smaller
>   dtype).
> - **Latency-bound** (thousands of tiny kernels, few bytes): `restrict` (34 MB but
>   **13000 kernels**), `sync_within` → *fewer/bigger kernels* (vectorise/batch).
>
> Consequences: **KO is effectively free** (doubling RHS FLOPs costs ~3% time);
> **Ozaki's value reframes to bandwidth** (INT8 = 8× smaller footprint, not tensor-
> core FLOPs); calibrated caps gave **5.6×** (1.75 vs 9.80 ms/step). The dynamic
> moving-box profile showed **fragmentation stayed 1.0** (lowest-slot allocation
> self-compacts) → **compaction retired**. Host-side regrid is ~20% of the dynamic
> cycle — watch at BBH scale. Next levers: M0 vectorise restriction (✅ done),
> M1 fuse the RK4 stage (bandwidth), M2 precision/Ozaki-footprint.

*Caveat:* the per-region micro-benchmark labels (RHS vs root-sync vs cross-level
sync) fused into one module, so the op-family breakdown above is the reliable
attribution — not separate phase timings. For clean per-phase numbers, the
micro-benchmarks must run as separate `jit`'d captures.

---

## 1. Block compute over inactive slots 🔴 (biggest BBH lever)

**Now.** Every step computes the RHS for *all* `LEVELS × MAX_BLOCKS` slots,
active or not (`rhs_all_levels` vmaps over the full slot array; inactive results
are masked). This is what keeps the step shape-stable → no recompile on regrid.

**Cost.** For a deep, sparse hierarchy — exactly the BBH case — most slots are
inactive. At `LEVELS=8`, `MAX_BLOCKS=64`, with (say) ~10 active blocks per level,
we compute **512 block-RHS per stage** to use ~80. That's ~6× wasted FLOPs *and*
HBM traffic, and it grows with `MAX_BLOCKS`. The waste is the single largest
inefficiency in the design.

**Faster ideas.**
- **Per-level `MAX_BLOCKS` sizing.** `MAX_BLOCKS` need not be uniform across
  levels. Root needs few; the finest levels need the most. Sizing each level to
  its realistic occupancy cuts the dead slots dramatically with zero algorithm
  change. *Low risk, immediate.* 🟡
- **Block compaction / gather-active.** Keep a compact list of active block
  indices and `vmap` only over those. The catch: the active *count* changes on
  regrid → either (a) recompile per distinct count (bad), or (b) pad to a fixed
  "max active" capacity per level (≪ MAX_BLOCKS) and vmap over that. (b) keeps
  shape-stability while slashing the dead-slot count. *Medium risk, large win.* 🔴
- **`scan`-over-blocks instead of `vmap`** when occupancy is low, trading
  parallelism for skipping empties — usually worse on GPU; mention only to
  dismiss. 🟢 (don't)

**Verdict.** The per-level sizing is free and should happen (done). Compaction is
valuable for **deep, dynamic, memory-bound BBH grids** — but, per the re-profile
below, *not* for the static 2D prototype, so its design pass moves to Phase 5/6.

**✅ STATUS — addressed (Phase 1 + Phase 2 auto-calibration).** AMR storage is now
**ragged per-level**: `AMRState.blocks`/`active` are tuples of `(MB_L, …)` arrays
(`MAX_BLOCKS_PER_LEVEL`), and the step `vmap`s each level over its own `MB_L` — so
compute scales with the per-level capacity, not a global `MAX_BLOCKS`. On top of
that, capacities are **driver-held and auto-calibrating**: `AMRTopology.caps` grows
(sticky, up to the hard ceiling `MAX_BLOCKS`) when a level exhausts its slots,
enlarging the arrays so the step retraces **once** for the new shape — a rare
calibration recompile, not per-regrid. Peak occupancy is tracked
(`recommended_caps()`), persisted to a sidecar (`write/read_caps_sidecar`), and a
runaway guard errors on pathological growth. **Calibrated caps are now the
production default** (`make_calibrated_root_state` reads the sidecar to pre-size).

**📊 Re-profile verdict — calibrated caps won; compaction deferred.** Calibrated
caps was the dominant lever (**~10.5×**): proven by `nbx4-uniform` (16 active root
blocks) costing the *same* as `nbx8-uniform` (64 active) — compute is **caps-bound,
not active-bound**. At calibrated caps the prototype sits at **~83% occupancy**, so
the further step — *compaction* to a dense prefix — has little slack to recover
**here**. That ~83% is an artifact of a static, single-feature, cache-resident toy:
compaction's real value (contiguous-block **memory bandwidth/locality**, plus
handling **regrid churn/fragmentation** and **2ᴸ sub-cycling** of dead deep slots)
only appears in the dynamic (Phase 5) / memory-bound 3D (Phase 6) regime, where it's
sequenced and will be measured. *It is deferred there, not discarded.*

---

## 2. Ghost-zone width vs operator reach 🟢/🟡

**Now.** `NG = 3`, set by the 6th-order stencils (reach ±3). KO is 6th-order
(also reach 3) so it fits. Per block we carry `(BS+6)²` cells for a `BS²`
interior — for `BS=32` that's a 27% halo overhead in 2D; in **3D it's
`(BS+6)³/BS³` — for `BS=16` that's ~95% overhead** (more halo than interior!).

**Faster ideas.**
- **Larger `BS` in 3D** amortizes the halo (overhead ~ `6/BS`). But `BS` is
  capped by SMEM on the Ozaki/Pallas path. There's a real tension: bigger blocks
  = less halo waste but less refinement locality and more SMEM. Worth a sweep. 🟡
- **Do NOT go to 8th-order KO** (`NG=4`) globally — it widens every halo (worse
  in 3D) for an accuracy gain we don't currently need. If ever needed, apply it
  on the FP path only (already documented). 🟢
- **Single-level temporal halo** (`NG_T = 4·NG` in `make_fused_step`) is a large
  halo for the temporally-fused kernel; fine for single-grid throughput runs but
  *not* used by AMR (AMR syncs per stage). Keep them separate. 🟢

**Verdict.** Halo overhead is a 3D problem, not a 2D one — fold `BS` selection
into the 3D port (Phase 4) with an SMEM-vs-halo sweep.

---

## 3. Time integration: shared-dt vs sub-cycling 🟢 (resolved)

**Now.** Both exist. `make_n_level_step` (shared dt) is simple but makes the root
take `2^(LEVELS−1)`× more steps than its CFL needs. `make_subcycled_n_level_step`
(Berger-Oliger) runs each level at its own dt — the right choice for BBH where a
shared finest-dt over the whole domain is unaffordable.

**Verdict.** Sub-cycling is implemented and validated; use it for production.
Shared-dt is kept for testing/reference. No change needed. The remaining cost is
the per-stage cross-level sync — **and the GPU baseline shows this is *not*
cheap: the `concatenate` sync glue is 28.3% of GPU time, ~0.93× the arithmetic**
(see Measured GPU baseline). Sub-cycling itself is correct; the sync *mechanism*
(full-block prolongation) is the thing to optimize (→ optimization A, §2/below).

---

## 4. Coarse-fine boundary accuracy 🟡 (in progress)

**Now.** Prolongation 6th-order; restriction just upgraded from 2×2 averaging
(2nd-order) to 6th-order interpolatory; time interpolation cubic-Hermite
(4th-order). With averaging the fine-block accuracy floored at ~1e-6; the
high-order restriction removes that floor.

**Faster/better ideas.**
- **Buffer zones** (reuse the Phase-2 nesting-buffer machinery) so the boundary
  sits away from the physics — cheaper than chasing ever-higher transfer order. 🟡
- **Berger-Colella flux refluxing** — *not* applicable (we're non-conservative
  FD, and BSSN is too). Documented; do not build. 🟢

**Verdict.** High-order restriction done; pair with buffer zones if interface
noise reappears at BBH amplitudes. Don't over-invest in transfer order.

---

## 5. The arithmetic schemes (Ozaki / Pallas) 🟢 with one watch-item

**Now.** Five schemes; the Ozaki RNS path uses INT8 tensor cores with CRT
reconstruction, and `pallas_ozaki` keeps the CRT pipeline resident in shared
memory. JAX pinned `==0.8.1`; **confirmed on GPU that JAX >= 0.9 breaks the SMEM
kernel** (power-of-2 tensor sizes / stricter Triton indexing), and 0.9 was at best
marginally faster, so we stay on 0.8.1.  Only `pallas_ozaki` is affected.

**Watch-items.**
- **JAX pin is technical debt.** Locked to 0.8.x by the Pallas-Ozaki break at 0.9.
  A 0.9+ kernel rewrite (tile padding to powers of 2, stricter-indexing fixes) is
  Phase-7 work; until then `pallas_ozaki` auto-skips on >=0.9 (`_skip_pallas`). 🟡
- **AMR currently runs on the FP/fused-FP kernel, not Ozaki.** The AMR per-block
  kernels call `_make_kernel_fn` from `fused_rhs_pallas` (FP64). The whole point
  of Ozaki (INT8 tensor-core throughput) is *not yet wired into the AMR path*.
  For BBH throughput this is a real gap: AMR + Ozaki is where the speed is. 🔴
- **INT8 dynamic range** for BSSN variables (which span many orders of magnitude
  near a puncture) may need more RNS moduli than MCS — revisit moduli choice in
  the BSSN port. 🟡

**Verdict.** Biggest item here: **wire the Ozaki/Pallas kernel into the AMR
per-block RHS** so deep AMR runs get tensor-core throughput. Currently AMR is
FP64-only. This is a major BBH performance lever, parallel to §1.

---

## 6. Host-side regrid overhead 🟢

**Now.** Regrid bookkeeping is pure-Python/NumPy on host (dicts, `O(MAX_BLOCKS²)`
duplicate-child scan). Called every K steps.

**Cost.** Negligible today (`MAX_BLOCKS=64` → ~4k host ops), but the
`O(MAX_BLOCKS²)` scan and Python dict churn would grow with a large deep grid and
frequent regrids.

**Faster ideas.** If profiling ever shows host regrid as a bottleneck (unlikely
until very deep grids), replace the `O(n²)` child-existence scan with a hash on
bbox and batch the topology updates. *Low priority.* 🟢

**Verdict.** Fine for now; revisit only if a profile flags it.

---

## 7. Precision: FP64 everywhere 🟡

**Now.** All AMR/transfer/evolution in FP64.

**Faster ideas.** GR waveform work often tolerates FP32 in parts of the RHS, and
the Ozaki path exists precisely to exploit lower-precision tensor cores. A
mixed-precision strategy (FP64 where constraints are sensitive, lower elsewhere)
could be a throughput win — but it's a careful, late-stage optimization that
needs the BSSN constraint sensitivity mapped first. 🟡

**Verdict.** Keep FP64 until correctness is locked; revisit mixed precision as a
throughput pass once BBH is running.

---

## Priority summary

Ordered by **measured GPU share** where we have it (see Measured GPU baseline):

| # | Item | Measured | Payoff | Risk | When |
|---|------|---------|--------|------|------|
| C0 | **Rolled sub-cycling** (substeps → `lax.scan`) — **compile** O(2^L)→O(L) | compile 115s→34s @L6; ~24h→~90s @L15 | ✅ **done** (bit-identical) | low | done |
| C1 | Level-agnostic per-block kernel (`dx`/`dt` as data) → 1 Triton compile, not LEVELS | Ozaki compile | 🔴 (Ozaki compile) | medium | before Phase 7 |
| M0 | **Vectorize restriction** (per-slot `fori_loop` → 1 batched scatter) | `restrict` 13000→~handful kernels (~28% of step, latency-bound) | ✅ **done** (HLO flat in slot count) | low | done |
| M1a | **Factor prolongation out of the RK4 stage loop** (prolong brackets once, Hermite-combine per substep — prolongation is linear) | prolongation intermediates ~8→4 prolongs/advance (the dominant memory cost) | ✅ **done** (machine-precision ~1e-15) | medium | done |
| M1b | **Fuse the RK4 stage into one Pallas kernel** (sync→set→rhs→combine; ~500–2600 kernels/step → fewer HBM round-trips) | full-block HBM round-trips (1115 refs/step) | 🔴 large (memory-bound) | high | **FUTURE** — needs a Pallas kernel reading neighbour strips; the biggest remaining bandwidth lever |
| M2 | Mixed precision (FP32 RHS) / reframe Ozaki as INT8 *footprint* | bytes moved (memory-bound) | 🟡→🔴 | high (accuracy) | with Phase 7 |
| A1 | Footprint-only prolongation (slice parent before refining) | `concatenate` 28.5%→16.8% | ✅ **done** (≈2.4×) | low | done |
| A2a | Fuse the two cross-level `where`s into one | `select` (~1.2×) | ✅ **done** | low | done |
| 1c | Calibrated caps as default (`make_calibrated_root_state`) | dead `add`+`select` (~10.5×) | ✅ **done** | low | done |
| A2b | Annulus *strip* refinement (refine only the halo ring) | `concatenate` (~1.8× of 16.8%) | ⛔ **deprioritized** | medium | — |
| 1 | Block compaction to dense prefix | `select` + memory locality | ⛔ **retired** — dynamic profile showed frag stayed **1.0** (lowest-slot alloc self-compacts) | medium | — |
| 5 | Wire Ozaki/Pallas kernel into AMR RHS | `add` (24.7%, FP64) | 🟡 (throughput, not latency) | medium | with 3D |
| B | Multi-block within-level sync | correctness + `concatenate` | 🔴 (correctness) | medium | Phase 5 |
| 2 | `BS`/halo sweep in 3D | halo-carry overhead | 🟡 | low | Phase 6 |
| 4 | Buffer zones at coarse-fine boundary | interface noise | 🟡 | low | Phase 5 |
| 5b | Re-evaluate JAX pin | — | 🟡 | low | periodic |
| 7 | Mixed precision | — | 🟡 | high | late |

**What the full re-profile settled:**
- **A1 worked** (`concatenate` 28.5%→16.8% per step) and the **fused select** (A2a)
  shaved `select` ~1.2×. Both bit-identical.
- **Calibrated caps was the dominant lever (~10.5× alone)** — proven by
  `nbx4-uniform` (16 active root blocks) costing the *same* as `nbx8-uniform` (64
  active): compute is **caps-bound, not active-bound**. Now the production default.
- At throughput scale the op-mix is **flat** (`add` 24.7% back on top; `select` 17,
  `concatenate` 17, slices 21, `reduce` 9, `gather` 7) — **no fat target remains.**

**Strips (A2b) deprioritized — and a correction.** After A1, `concatenate` is only
16.8% and strips save just the interior-discard fraction (~1.8× of it) at 4× the
launches. Earlier this doc implied A2 "matters more in 3D" — that was wrong: it
conflated *halo-storage* overhead `(BS+2NG)ᴰ/BSᴰ` (worse in 3D, but A2 doesn't touch
it) with the *interior-discard* saving `BSᴰ/(BS+2NG)ᴰ`, which is **smaller** in 3D at
small BS (~38% interior at BS=16 vs ~71% at BS=32 in 2D). Net: low value even for BBH.

**Compaction (1) deferred to the BBH regime, not killed.** At calibrated caps the
2D prototype sits at ~83% occupancy, so compaction has little slack to recover *here*
— but that's an artifact of a **static, tiny, single-feature, cache-resident** toy.
BBH flips the three axes that make it pay: **memory-bound + large** (scattered active
blocks waste bandwidth/cache; compaction packs them contiguous), **dynamic** (moving
punctures → regrid churn + fragmentation + time-varying occupancy), and **depth ×
sub-cycling** (a dead slot at level L is computed 2ᴸ× per root step). It can only be
*validated* against a dynamic (Phase 5) / 3D-memory-bound (Phase 6) workload, so it's
sequenced there.
