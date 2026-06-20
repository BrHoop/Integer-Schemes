# Step 3.2f Increment-1 — warp-cooperative microbenchmark (the go/no-go)

The first concrete build of the warp-cooperative fp64 path (Phase 3.2f). It validates the
**mechanism** — fp64 warp-shuffle contraction reductions over a 4-lane group — on the
inverse-metric → conformal-Christoffel sub-computation, and reports the register/spill/
timing numbers that decide whether to proceed.

See `../../../../docs/archive/phases/phase_3/step_3.2f_warp_coop.md` for the full plan.

## Files
- **1a — mechanism** (`christoffel_ref.py` + `microbench_christoffel.cu`): inverse metric →
  conformal Christoffel. Validates the fp64 4-lane `__shfl` reduction. Ref validated to
  6.7e-16. **DONE on H200 (2026-06-15): correct (6.66e-16), 0 spill, warpcoop 40 regs vs
  naive 56, 1.71× slower on this small piece — mechanism GREEN.**
- **1b — register fit (single tensor)** (`ricci_ref.py` + `microbench_ricci.cu`): conformal
  Ricci R~_{ij}. Ref validated against **SymPy** to 1.1e-16. **H200 result: neither spilled
  (naive 128 regs / arrays in local memory), warpcoop 2.54× slower — Ricci alone is BELOW
  the overflow threshold, so it tests only warp-coop overhead, not the benefit.**
- **1c — register fit (FULL algebra, the right-scale test)** (`gen_algebra_cuda.py` +
  `microbench_algebra.cu`): transliterates the actual 850-statement CSE into a naive
  1-thread CUDA kernel (138 derivs + 826 temps live) and asks whether the *whole* algebra
  spills in nvcc/ptxas — the baseline a wall-B fix must beat. Vectors validated against the
  **JAX verbatim oracle**. *This is the scale where the register wall actually appears.*
- `build.sh` — builds all present microbenches with `nvcc -arch=sm_90a -Xptxas -v`.
- `*_vectors.bin`, `_bssn_algebra_kernel.cuh` — generated (raw float64 / CUDA; not committed).

## Run (local: generate vectors) — CPU, no GPU
```
cd 3D/src
python -m bssn3d.cuda.christoffel_ref   # writes cuda/test_vectors.bin
python -m bssn3d.cuda.ricci_ref         # writes cuda/ricci_vectors.bin (needs sympy for the self-test)
python -m bssn3d.cuda.gen_algebra_cuda  # writes cuda/_bssn_algebra_kernel.cuh + algebra_vectors.bin
```

## Run (Marylou H200: build + execute) — needs 2FA push
```
# after ./sync.sh push, on the cluster:
cd 3D/src/bssn3d/cuda
bash build.sh                                 # -Xptxas -v register/spill for all present
./microbench_christoffel test_vectors.bin     # 1a: mechanism
./microbench_ricci ricci_vectors.bin          # 1b: single-tensor (below threshold)
./microbench_algebra algebra_vectors.bin      # 1c: FULL algebra — the real wall-B test
bash sweep_regs.sh                            # 1d: -maxrregcount sweep (is occupancy the lever?)
```

**1c result (H200, 2026-06-15):** naive full algebra = **255 regs / 4192 B spill / 0.948 ms
per 262k pts**, correct (1.67e-16 vs JAX oracle). Wall B is real in CUDA but **mild** (~4 KB,
vs fused_tiled's ~25 KB) because the 138 derivs are inputs ptxas reloads from L1/L2 — only
the 826 CSE temps press. The kernel is **occupancy-limited** (~8 warps/SM at 255 regs), not
spill-bound. → `sweep_regs.sh` (1d) asks whether trading registers for occupancy is faster.
Paste back the full stdout (the `-Xptxas -v` lines + each program's report).

## What to read from the 1b (Ricci) output — the register-fit go/no-go
1. **`max|err| vs ref`** for both kernels ~1e-12 or better (fp64 round-off) — confirms the
   warp-cooperative Ricci (shuffle-REDUCE for `d_l G^l_ij` + shuffle-BROADCAST for
   `d_j G^l_il`) is correct.
2. **The decisive contrast: `naive` vs `warpcoop g=4` register + local-memory footprint.**
   The win is warp-coop holding the trunk in registers (low `regs`/low `spill(local)`)
   while naive overflows. **Read both `-Xptxas -v` ("spill stores/loads" = true register
   spill) AND the runtime `spill(local)` (= `localSizeBytes` = stack frame + spill + big
   local arrays).** Interpretation nuance: a big array (`dG[3][3][3][3]` = 81 doubles in
   naive) may land in *local memory* rather than be labeled a register "spill" — either
   way it is off-register, HBM-backed traffic. If naive shows large `localSizeBytes` /
   spill and warpcoop is materially smaller, **wall B's register-distribution premise is
   validated**. If naive does NOT overflow (ptxas kept it in registers somehow) the trunk
   isn't big enough yet → extend toward the full 128-temp trunk.
3. **`warpcoop / naive time ratio`** — with a real register win, warp-coop should now be
   *competitive or faster* (unlike 1a's tiny piece), since naive pays HBM-backed local
   traffic. A warp-coop that is faster here is the first direct evidence the approach pays.

## Next (pending these numbers)
- If 1b shows naive overflow + warpcoop resident → the register-distribution premise holds;
  integrate the trunk into the 3.2e 2.5D kernel via JAX FFI (`scheme="warp_coop"`).
- If g=4 looks tight/awkward → the parked g=8 fallback is a one-parameter change.
