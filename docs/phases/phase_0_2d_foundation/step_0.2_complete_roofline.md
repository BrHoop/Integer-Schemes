# Step 1.2 — Complete the Roofline (achieved bandwidth + efficiency)

**Phase 1 · Step 1.2 · Status: ACTIVE (next to implement)**

## Purpose
Step 1.1 measured the *arithmetic intensity* of the FD RHS (**3.2 FLOP/byte →
memory-bound**) and the throughput plateau (**fused FP64 ≈ 225 Mpts/s, flat
across 256²/512²**). But the GPU metadata detection failed on the H200
(`device: "cuda:0"`, `vram_gb: null`), so `peak_bw_GBs` was never resolved and
the benchmark reported **no achieved bandwidth and no % of peak**. We have the
roofline's x-axis but not where the kernel sits relative to the roof.

This step finishes the roofline so we know **how far the FP64 baseline is from
the HBM bandwidth ceiling** — which decides the *first move* of Step 1.3:
- far below peak → memory-utilization (occupancy / access pattern / TMA) is
  leaving performance on the table; fix that before/with fusion;
- near peak → genuinely bandwidth-saturated; the only lever is reducing traffic
  via temporal fusion (raising arithmetic intensity). That is the 1.3 plan.

It is deliberately a **small, measurement-only** step: complete the picture
before the heavier 1.3 fusion work, not after.

## Scope & decisions
- **No physics, no kernel changes.** Pure profiling infrastructure + one figure.
- **Authoritative source is deferred.** The numerator here is XLA's HLO
  cost-model "bytes accessed" (an *estimate* of HBM traffic). The ground-truth
  DRAM throughput comes from Nsight in **1.7**; 1.2's job is the cheap,
  always-on estimate and the roofline framing. Cross-check the two at 1.7.
- **GPU-only for the real numbers** (H200, via `sync.sh`); the detection
  fallbacks and metric plumbing are CPU-unit-testable with a mocked device.

## Background — what 1.1 left open
From `step_1.1_results/benchmark_results.json`:
- `fused_floating_point`: 225 Mpts/s, 3.2 FLOP/byte, `bound: memory`, L2 ≈ 1e-13.
- `device: "cuda:0"`, `vram_gb: null` → `pynvml` unavailable/threw, so `device`
  fell back to `str(jax.devices()[0])`; `_BW_TABLE` has **no H200 key**, so
  `peak_bw_GBs` stayed `None` and efficiency was never computed.
- Rough envelope (≈300 GB/s achieved vs ≈4.8 TB/s peak ≈ **~7%**) — but that
  used `rhs_MB × 4`, which over-counts the *fused* path (it reads state once,
  not 4×). The real number needs the compiled-step byte count. Measure, don't
  guess.

## Tasks

### T1 — Robust device / VRAM / peak-BW detection (`gpu_info`)
- Keep the `pynvml` path; add fallbacks when it fails:
  1. `jax.devices()[0].device_kind` (usually `"NVIDIA H200"`).
  2. `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.
- Extend `_BW_TABLE` with Hopper-HBM3e and successors:
  `"H200": 4800` (H200 SXM, 141 GB HBM3e, ≈4.8 TB/s), and while here
  `"H200 NVL": 4800`, `"B200": 8000`, `"GH200": 4900` (best-effort; refine when
  seen). Keep the case-insensitive substring match.
- Result: on Marylou, `gpu_info()` returns name `H200`, `vram_gb ≈ 141`,
  `peak_bw_GBs = 4800`.

### T2 — Achieved bandwidth + efficiency metric
- Add `_step_cost(step_fn, state)` (mirror of `_rhs_cost`) that pulls
  `"bytes accessed"` from the compiled **step** function's `cost_analysis()` —
  this is the per-step HBM traffic XLA actually moves (correct for the fused
  single-pass step; `rhs_MB × stages` is not).
- Derive and record per (scheme, grid):
  - `step_MB_accessed`  = cost-model bytes / 1e6
  - `hlo_GBs`           = step_bytes / `per_step_us`·1e-6   (cost-model UB)
  - `hlo_bytes_pct_peakBW` = `hlo_GBs` / `peak_bw_GBs` × 100  (UB, **not DRAM**)
- Thread these into the row dict and into `save_csv` / `save_json` / `print_table`.
- (See the FINDING/RELABEL notes below — these are upper bounds, not DRAM.)

### T3 — Roofline figure (`benchmark_roofline.png`)
- Log-log: arithmetic intensity (FLOP/byte, x) vs achieved compute (GFLOP/s, y).
- Draw the H200 ceilings:
  - HBM bandwidth line `y = peak_BW · x` (slope = 4800 GB/s);
  - **FP64 vector** ceiling ≈ 34 TFLOP/s (the current CUDA-core path);
  - **INT8 tensor-core** ceiling ≈ 1979 TOP/s dense / ≈3958 2:4-sparse — drawn
    as the *headroom the thesis targets* (annotate, don't imply we're there).
- Plot the scheme point(s). The picture should show the FP64 baseline sitting on
  the memory-bound (left) side, below the roof.

### T4 — Re-run on H200 + pull (user action, 2FA)
`./sync.sh push` → on a GPU node `python -m mcs2d.benchmark 2D/params.toml
docs/phases/phase_0_2d_foundation/step_0.1_results` → `./sync.sh pull results`.
(`sync.sh` now shares one SSH connection → a single 2FA per push/pull.)

### FINDING (T4 run, 2026-06-10) — the cost model cannot measure DRAM
The H200 run detected the device correctly (H200, 140.4 GB, 4800 GB/s) but
exposed that **HLO cost-model "bytes accessed" is the wrong tool for bandwidth**:
`step_MB_accessed ≈ 2574 MB at 512²` is **120× the 21.5 MB state** — physically
impossible as DRAM traffic.  The cost model counts the temporal fusion's
**redundant on-chip halo reads** (the `NG_T = 12` overlap), most of which hit
L1/L2/shared, not HBM.  So the derived "46% of peak BW" is an **upper bound that
includes on-chip work, not DRAM utilization**.  Honest DRAM bracket:
- floor (perfect single-pass, 2×state): 226 Mpts/s · 160 B/pt ≈ **36 GB/s ≈ 0.8%**;
- ceiling (cost-model all-bytes): **≈46%**; truth is in between → needs counters.

Consequence: the "memory-utilization vs fusion" fork for 1.3 was the wrong
question (the baseline is *already* temporally fused).  The real question —
**DRAM-bound or compute/occupancy-bound?** — the cost model cannot answer.
Nsight is banned here and counters are locked, but a counter-free nsys *trace*
(T5 below) answered it anyway: **DRAM-bound.**

### RELABEL (done) — stop the metric from misleading
`benchmark.py` metrics renamed and captioned as cost-model **upper bounds**:
`achieved_GBs→hlo_GBs`, `achieved_GFLOPs→hlo_GFLOPs`,
`bw_eff_pct→hlo_bytes_pct_peakBW`.  The table header is now `HLO cost (UB)` with a
`* not DRAM — real DRAM needs HW counters (locked); use profile_regime --smi`
footnote; the roofline plots the trustworthy **RHS intensity (3.2)** on x, draws
the achieved point **hollow** (`HLO-UB`), and captions that the true point needs
HW counters.  The ridges/ceilings (the valuable, correct content) are unchanged.

### T5 — regime verdict: **DRAM-bound** (answered, counters not needed)

**Nsight is banned on this cluster** ("we cannot have NVIDIA Nsight Systems or
Nsight Compute available … due to security concerns"), and GPU performance
counters are locked (`ERR_NVGPUCTRPERM`) — so `ncu`/`nsys --gpu-metrics` give
nothing.  But a one-off `nsys` *trace* (kernel/graph timeline needs no counters)
was captured, and its SQLite export answered the regime question outright:

| evidence (from the nsys SQLite) | value | rules out |
|---|---|---|
| GPU busy fraction (`GRAPH_TRACE`, 200 steps) | **99.2%** (10 µs idle / 1161 µs step) | launch/dispatch-bound |
| theoretical occupancy (regs ≤26, 128-thr blocks) | **100%** | occupancy-bound |
| achieved compute (1.7 GFLOP / 1161 µs) | **~4% of FP64 peak** | compute-bound |
| **shared memory used** (all kernels, static+dynamic) | **0 bytes** | — |

Not launch, not occupancy, not compute ⇒ **DRAM-bandwidth-bound**.  And the cause
is structural: XLA's `fused_floating_point` step is **~15 small element-wise
kernels** (`wrapped_broadcast`, `loop_dynamic_slice_fusion` / `…_update_slice…`
= the stencil, `loop_multiply/divide/add_fusion`), each a **full-grid
global-memory pass with zero on-chip reuse** — XLA implements the 6th-order
stencil as global-memory `dynamic_slice` shifts and **uses no shared memory at
all**.  That redundant global traffic is the bottleneck.

→ **1.3 first move = tile the stencil into shared memory + temporally fuse the RK
stages** (Pallas, then tensor cores), to eliminate the global-memory passes.
This is the evidence-backed mechanism behind the "~130× IA gap" below.

#### Going forward — permission-free profiling (`mcs2d/profile_regime.py`)
Nsight is out, but NVML utilizations are not counter-gated, so the cluster's
allowed tools still give the DRAM-vs-compute tell:
```bash
# SM% + MEM% during the step loop (MEM% = memory-controller busy ≈ DRAM tell)
python -m mcs2d.profile_regime --smi --nx 512 --seconds 15
# per-XLA-op device-time breakdown (Perfetto / TensorBoard) — JAX analog of the
# PyTorch profiler; use for before/after-fusion comparison
python -m mcs2d.profile_regime --jax-trace traces/jax --nx 512 --steps 300
# live: salloc a node, ssh to it, `module load nvtop; nvtop`, run with no flag
```
For **Step 1.7** (confirming the optimized kernel flipped to compute-bound), the
permission-free signal is **MEM% drops while SM% stays high** under `--smi`, plus
a measured throughput win — no counters required.  An exclusive-reservation
request for counter access is still worth filing for the precise TC-utilization
numbers, but it no longer blocks anything.

## The quantitative target this sets for 1.3 (ridge points)
The roofline **ridge** (where memory- and compute-bound meet) is
`peak_compute / peak_BW`:
- FP64 vector: 34 000 / 4800 ≈ **7.1 FLOP/byte**. At 3.2 we sit just left of it —
  modest fusion would already make the *FP64* path compute-bound.
- INT8 tensor core: 1 979 000 / 4800 ≈ **~410 OP/byte**. To make INT8 TCs
  compute-bound, temporal fusion must raise arithmetic intensity by **~130×**
  from today's 3.2. That number is the concrete fusion target for 1.3 and the
  reason naive (unfused) tensor-core use cannot win.

## Deliverables
- Patched `gpu_info` (device/VRAM/peak-BW with fallbacks + H200 in `_BW_TABLE`).
- New benchmark columns: `step_MB_accessed`, `hlo_GBs`, `hlo_GFLOPs`,
  `hlo_bytes_pct_peakBW` (cost-model upper bounds — CSV + JSON + printed table).
- `benchmark_roofline.png` in `step_1.1_results/`.
- `mcs2d/profile_regime.py` — permission-free GPU regime profiling (`--smi`,
  `--jax-trace`) for this Nsight-banned cluster.
- Recorded regime verdict: **DRAM-bandwidth-bound** → 1.3 = tile + temporally fuse.

## Reuse vs build
- **Reuse:** `_rhs_cost` pattern, the `measure()` row plumbing, the existing
  `_BW_TABLE`, `save_csv/json`, `--replot` path (re-render roofline from JSON).
- **Build:** `_step_cost`, the 3 derived metrics, `save_roofline_plot`, the
  `gpu_info` fallbacks + H200 entry.

## Exit criteria
- ✅ On Marylou, `gpu_info()` reports `H200`, `vram_gb ≈ 140`, `peak_bw_GBs = 4800`.
- ✅ Benchmark JSON/CSV/table include the HLO cost-model metrics (`hlo_GBs`,
  `hlo_GFLOPs`, `hlo_bytes_pct_peakBW`), captioned as upper bounds (not DRAM).
- ✅ `benchmark_roofline.png` shows the FP64 baseline (RHS IA 3.2) against the HBM
  + FP64 + INT8-TC ceilings/ridges, with the achieved point marked HLO-UB.
- ✅ **T5 — regime verdict (counter-free):** `DRAM-bandwidth-bound`, established
  from the nsys SQLite (99.2% GPU-busy, 100% occupancy, ~4% FP64 peak, 0 B shared
  memory; stencil = global `dynamic_slice` passes).  `profile_regime.py --smi`
  provides the permission-free SM%/MEM% confirmation going forward (Nsight banned).
- ✅ The 1.3 first-move decision is written down: **tile the stencil into shared
  memory + temporally fuse the RK stages** (XLA uses 0 B shared memory today).
- ⬜ (optional, non-blocking) exclusive-reservation counter request, for precise
  TC-utilization numbers in 1.7.

**Step 1.2 is complete** — regime established (DRAM-bound), 1.3 direction set.

## Status / changelog
- 2026-06-10 — Created. Active step after 1.1 closed (`pallas_fp` retired;
  `fused_rhs_pallas` → `fused_rhs_fp`). Pure measurement step; no kernel work.
- 2026-06-10 — **T1–T3 implemented + CPU-tested** in `benchmark.py`:
  - T1: `gpu_info` now has pynvml → `device_kind` → `nvidia-smi` fallbacks;
    `_BW_TABLE` gains H200 (4800) + Hopper/Blackwell entries; new `_COMPUTE_TABLE`
    (FP64 vector + INT8-TC ceilings) → `peak_fp64_GFLOPs`, `peak_int8_TOPs`.
  - T2: `_step_cost` (per-step HLO bytes/FLOPs) → `achieved_GFLOPs`,
    `achieved_GBs`; `_add_bandwidth` → `bw_eff_pct`. New CSV/JSON columns + a
    "Roofline" line in the printed table (FLOP/byte · GFLOP/s · GB/s · % HBM peak).
  - T3: `save_roofline_plot` → `benchmark_roofline.png` (HBM line, FP64 ridge 7.1,
    INT8 ridge ~412, scheme point with % BW). Wired into `_render_plots` + `replot`.
  - Verified on CPU with a mocked H200 and confirmed the no-GPU path degrades
    gracefully (roofline skipped, run + data survive).
- 2026-06-10 — **T4 ran on H200** (device detection ✅). Surfaced that cost-model
  bytes ≠ DRAM (step bytes = 120× state). Two follow-ups landed:
  - **Relabel:** metrics → `hlo_*` upper bounds; table `HLO cost (UB)` + footnote;
    roofline uses RHS IA on x, hollow `HLO-UB` point, Nsight caveat. Figures
    regenerated from the real H200 JSON.
  - `sync.sh` now uses one shared SSH connection (single 2FA per push/pull).
- 2026-06-10 — **T5 resolved; Step 1.2 complete.** Discovered the cluster bans
  Nsight and locks GPU counters (`ERR_NVGPUCTRPERM`), so `ncu`/`nsys --gpu-metrics`
  are unavailable. A counter-free `nsys` trace was still captured; parsing its
  SQLite (`GRAPH_TRACE` + kernel launch configs) gave the verdict directly:
  **DRAM-bandwidth-bound** (99.2% busy, 100% occupancy, ~4% FP64 peak, **0 B
  shared memory**; XLA does the stencil as global `dynamic_slice` passes).
  - Replaced `nsight_target.py` with **`profile_regime.py`** (banned-Nsight-aware):
    `--smi` (nvidia-smi dmon SM%/MEM%, permission-free DRAM tell), `--jax-trace`
    (Perfetto/TensorBoard op breakdown), plain (for live `nvtop`).
  - **1.3 direction set:** tile the stencil into shared memory + temporally fuse
    the RK stages (kill XLA's redundant global-memory passes).
