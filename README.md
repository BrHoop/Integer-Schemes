# Maxwell-Chern-Simons Numerical Solver

JAX-based solver for the 2.5D and 3D Maxwell-Chern-Simons (MCS) PDE system, with multiple numerical schemes targeting NVIDIA GPUs (B200 / H100). Designed as a stepping stone to a BSSN binary-black-hole code, so the architecture prioritizes block-structured AMR and shape-stable JIT.

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
├── 3D/                       # mcs3d — 3D solver (AMR planned, Phase 4)
│   ├── params.toml
│   ├── src/mcs3d/
│   └── tests/
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

## AMR (2D, in development)

Phase 1 (static foundation) is complete: block-structured layout, 6th-order prolongation, 2nd-order restriction, ghost-zone sync within and across levels, RK4 stepping at the root level, 2-level static evolution validated against the analytic birefringent wave. See `AMR_PLAN.md` for the multi-phase roadmap.

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
