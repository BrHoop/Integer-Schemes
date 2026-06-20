# Integer-Schemes — a GPU-optimized BSSN numerical-relativity pipeline

JAX/CUDA-based numerical-relativity code targeting NVIDIA GPUs (H100/H200), built from a
Maxwell-Chern-Simons (MCS) toy model up to a **3D BSSN binary-black-hole RHS** at extreme
mass ratios (senior thesis, advisor Dr. David Neilsen, BYU). The thesis question is *"can
we make the BSSN GR RHS as fast as possible on the GPU?"* — the headline result is the
**M4 fused CUDA RHS, 2.28× faster than the XLA-verbatim baseline** (2.07× on the full RK4
step), with the time integrator (MSRK) adding 1.31× on top.

- **2D/3D MCS** (`mcs2d`, `mcs3d`) — the correctness / Ozaki-bit-reproducibility ground +
  block-structured AMR.
- **3D BSSN** (`bssn3d`) — the efficiency ground: a Dendro-GR-validated RHS, fused into one
  register-resident CUDA kernel.
- **Multipatch** (`multipatch`) — Llama 7-patch grid + a node-centered 3D block-AMR port.

> **What's the contribution / how is this different from other BSSN-on-GPU solvers?** See
> [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md), the survey
> [`docs/LITERATURE_SURVEY.md`](docs/LITERATURE_SURVEY.md) §5, and the **"What's novel"**
> section near the bottom of this file.

## Repository layout

This is a monorepo of three installable Python packages.

```
Integer-Schemes/
├── common/                   # mcs_common — shared utilities
│   └── src/mcs_common/
│       ├── wave_state.py
│       ├── bfp48.py
│       ├── jax_config.py
│       └── ioxdmf.py
├── 2D/                       # mcs2d — primary, fully tested 2.5D solver + AMR
│   ├── params.toml
│   ├── src/mcs2d/
│   │   ├── main.py
│   │   ├── benchmark.py / profile.py / visualize.py
│   │   ├── schemes/          # floating_point, ozaki, pallas_ozaki, fused_*
│   │   └── amr/              # state, kernels, evolve  (Phase 1 complete)
│   └── tests/                # unit / integration / regression
├── 3D/                       # 3D solvers
│   ├── params.toml
│   ├── src/mcs3d/            # 3D MCS solver (floating-point + unfused ozaki)
│   ├── src/bssn3d/          # 3D BSSN RHS: verbatim → M4 fused CUDA (cuda/), MSRK, validation
│   ├── src/multipatch/      # Llama 7-patch grid + node-centered 3D block-AMR (amr/)
│   └── tests/                # unit / integration / validation / multipatch_amr
└── pyproject.toml            # workspace root: shared pytest config
```

## Installation

```bash
# CPU / development — install all three packages editable.
pip install -e ./common ./2D ./3D

# GPU (required for ozaki / fused_ozaki schemes)
pip install -e "./2D[gpu]" "./3D[gpu]"
```

Verify GPU is visible:
```bash
python -c "import jax; print(jax.devices())"
```

## Running simulations

```bash
# 2D simulation (uses 2D/params.toml)
python -m mcs2d.main                  # or: python 2D/src/mcs2d/main.py
python -m mcs2d.main /path/to/custom.toml

# 3D simulation
python -m mcs3d.main
```

## Tests

Each subproject has its own categorized test layout:

```
2D/tests/
├── conftest.py
├── unit/                # kernel-level, fixed-shape, <5 s each
├── integration/         # multi-step evolution against analytic references
└── regression/          # invariant guards (no recompile, bit-identical)
```

Run via pytest or via the `run.sh` orchestrator:

```bash
# All 2D tests
./2D/tests/run.sh
# Just the unit tests
./2D/tests/run.sh unit
# Skip anything marked @pytest.mark.slow
./2D/tests/run.sh fast
# AMR tests across all categories
./2D/tests/run.sh amr

# Or run an individual file as a standalone script:
python 2D/tests/unit/test_amr_kernels.py
```

The 3D suite mirrors the same layout under `3D/tests/`.

## Numerical schemes (2D)

| `scheme` value         | Description | When to use |
|---|---|---|
| `floating_point`       | Pure FP64 finite differences via XLA. Always correct. | Debugging, reference solution |
| `ozaki`                | INT8 Ozaki RNS decomposition. Uses GPU INT8 tensor cores. | Production 2D runs on GPU |
| `fused_floating_point` | Tiled FP64; single HBM pass per tile. | Throughput on GPU without quantisation |
| `fused_ozaki`          | Ozaki + fused Pallas/Triton kernel; CRT spills to L2. | Max throughput, B200/H100 |
| `pallas_ozaki`         | Single Pallas kernel per tile; CRT in shared memory. | No L2 round-trips; fastest Ozaki path |

## AMR (2D complete; 3D port active)

The **2D block-structured Berger-Oliger AMR is complete** (static foundation, regridding with
zero mid-evolution recompile, Hermite sub-cycling, GPU profiling with calibrated caps ~10.5×,
multi-block within-level sync, moving-feature tracking). A **node-centered 3D block-AMR port**
onto the Llama multipatch grid is an active CPU parallel track (`3D/src/multipatch/amr/`, Phase A
done, 25 tests). See `docs/archive/AMR_PLAN.md` for the 2D detail, `docs/FUTURE_WORK.md` §7 for
open levers, and `docs/phases/README.md` for live project status.

## What's novel (vs other BSSN-on-GPU solvers)

- **A fused single-kernel BSSN RHS (M4).** The 138 finite-difference derivatives are computed
  into **register scalars** and the whole ~850-statement Christoffel/Ricci/gauge algebra runs in
  **one CUDA kernel** — derivatives and intermediates never touch HBM. 2.28× over XLA-verbatim,
  *despite* register spill, because spill is the lesser evil vs the multi-GB HBM derivative
  round-trip. Dendro-GR/GR-Athena++/AthenaK keep the derivative and algebra in separate passes.
- **A published-grade register/spill analysis of the full BSSN *evolution* RHS on GPU** — the
  thing NRPyElliptic/Dendro flag qualitatively and park as "challenging." Includes a **four-way
  ptxas-invariance result** (source-level CSE/register restructuring spanning 24→2390 temps lands
  at the same register count) and localization of the irreducible spill to the Ã_ij/Ricci outputs.
- **The honest negative-plus-positive thesis:** the high-order GR RHS is **latency/spill-bound at
  ~3% of fp64 peak**, tensor cores are the *wrong* tool for it (non-GEMM pointwise algebra), and
  the win is locality (fusion) + fewer evals (MSRK), not precision tricks — Ozaki-II relocates to
  the initial-data elliptic solve where reduced precision is actually safe.
- **Block-structured + shape-stable-JIT + multipatch (cubed-sphere) grid**, a different GPU
  tradeoff from Dendro's wavelet octree (regular/coalesced memory vs finer adaptivity).

## Key parameters

Each subproject's `params.toml` has inline documentation. Most-used knobs:

| Parameter | Effect |
|---|---|
| `Nx`, `Ny` (`Nz` in 3D) | Grid resolution |
| `Nt`                    | Number of time steps |
| `scheme`                | Numerical scheme (see table) |
| `cfl`                   | Time-step size; 0.05 for 2D, 0.02 for 3D |
| `Lambda`                | Chern-Simons coupling constant |
| `enable_cs`             | Set to `0.0` to reduce to Maxwell + scalar wave |
| `ko_sigma`              | Kreiss-Oliger dissipation (0 to disable) |
