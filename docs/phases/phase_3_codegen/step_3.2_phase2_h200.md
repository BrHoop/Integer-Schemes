# Step 3.2 — Phase 2 (H200, tomorrow): confirm the spill/regime wins

> **Naming:** the H200-tomorrow half of the Step-3.2 push (CPU-today half:
> `step_3.2_phase1_cpu.md`). Not the project's canonical Phase 2 (BSSN correctness, done).
> Everything here is **GPU-gated and needs the user's 2FA** — the agent cannot run it.

## What Phase 1 hands up (CPU-validated artifacts)
- The `precision="fp32_contraction"` path + the fp32 Pallas kernel (`BSSN_PALLAS_FP32=1`).
- (If 1.1 viable) an associative-reordered kernel, predicted to drop peak liveness < 255.
- (If 1.2 done) an interpret-validated fused derivative kernel (derivs on-chip).

## The runs (in priority order)

All on a Marylou H200 GPU node, in `~/Integer-Schemes` (env activated), after
`./sync.sh push` from local. Paste back the `_kernel` + `SUMMARY` / `SM`/`MEM` lines.

### 2.1 — Regime confirm: is the 2 KB fp32 spill L2-resident? *(highest priority)*
The deferred measurement. `spill_probe` says *how much* spills; `--smi` says whether it
**hits HBM**. NR-GPU literature (arXiv:2501.14030 + the L2-bound finding) says optimized NR
RHS bottoms out L2-cache-bound — confirm we're there.
```bash
BSSN_PALLAS_FP32=1 BSSN_PALLAS_BS=32 BSSN_PALLAS_WARPS=1 \
  python -m bssn3d.profile_regime --smi --scheme pallas --n 128
```
- MEM% **moderate / lower than the 96–98 % fp64 baseline** → the 2 KB is L2-resident; the
  algebra is at its best practical regime (L2-bound) → wall 2 (2.3) is the next lever.
- MEM% **still ~96 %** → the 2 KB *is* hitting HBM; revisit (associative reordering 2.2,
  or selective fp16) before fusion.

### 2.2 — Associative-reordered spill → 0? *(only if 1.1 predicted a win)*
```bash
BSSN_PALLAS_FP32=1 BSSN_PALLAS_BS=32 BSSN_PALLAS_WARPS=1 BSSN_SCHEME=<reordered> \
  python -m bssn3d.spill_probe 2>&1 | grep -E "_kernel|SUMMARY"
```
- Spill **→ ~0** → the reassociation lever is real (and `ptxas`-binding, as predicted);
  lock the reordered kernel as the default.
- Spill **unchanged ~2 KB** → reordering can't beat the structural width on-device; drop it
  and rely on the L2-residency from 2.1.

### 2.3 — Fused derivative kernel: does removing wall 2 flip the regime? *(the big one)*
```bash
# spill of the fused kernel:
BSSN_PALLAS_FP32=1 BSSN_PALLAS_BS=32 BSSN_PALLAS_WARPS=1 BSSN_SCHEME=<fused> \
  python -m bssn3d.spill_probe 2>&1 | grep -E "_kernel|SUMMARY"
# and its regime:
BSSN_PALLAS_FP32=1 BSSN_PALLAS_BS=32 BSSN_PALLAS_WARPS=1 \
  python -m bssn3d.profile_regime --smi --scheme <fused> --n 128
```
- **MEM%↓ / SM%↑ toward compute-bound** → the no-CUDA path worked: fp32 (register-fit) +
  fusion (wall 2 gone) → compute-bound BSSN algebra in Pallas. **This is the Phase-3 prize.**
- **Still memory-bound** → wall 2 not fully removed (halo traffic / SMEM occupancy);
  diagnose with the per-op `--jax-trace`, iterate the SMEM tiling.

## Watch-outs (Marylou, per project workflow)
- Push → (reinstall only if a *new* package appeared) → run → paste. One 2FA per push.
- Pallas compile can be slow (the fused kernel is large; ~1600 s wall risk for unrolled
  forms — watch it; drop `BSSN_PALLAS_BS` if it OOMs/stalls).
- `spill_probe` register/spill numbers are **arch-portable** (255-reg file is universal);
  `--smi` regime numbers are **H200-specific** (don't trust them off a consumer/A100 card).

## RESULTS (H200, 2026-06-13)

| run | result | reading |
|---|---|---|
| fp32 pallas spill (N=48) | **255 regs / 1936 B / 1 kernel** | A100's ~2112 B transfers → fp32 algebra nearly register-resident ✓ |
| fp32 pallas regime (N=128) | **SM 100% / MEM 83%** → still memory-bound | down from fp64 ~96% (the ~13-pt drop = ~25 KB spill removed); residual 83% = **wall 2, the 138-array deriv read** — algebra-only can't remove it |
| `fused` (fp32) spill | **Triton lowering FAILS** — array (54,54,54) not power-of-2 | whole-grid prototype is GPU-incompatible (didn't even reach `jnp.pad`) |
| `fused_fp64` (order 8) spill | **Triton lowering FAILS** — array (56,56,56) not power-of-2 | same — confirms the tiled SMEM rewrite is mandatory |

**Headline:** fp32 is **vindicated and shown insufficient alone** — it kills the spill
(96→83% MEM) but leaves wall 2, so the algebra-only kernel stays HBM-bound. **Fusion is
confirmed make-or-break**, and the whole-grid fused prototype **cannot lower under Triton**
(power-of-2 requirement) → the **tiled power-of-2 SMEM-halo kernel (3.2d-GPU / `docs/algebra.md`
Increment 2) is the mandatory next build** (BS=8 power-of-2 tiles, one-hot halo crop,
trunk→SMEM scratch). fp32 accuracy is benign (CPU strong-field measure, `[[algebra-speedup-ideas]]`),
so the live target is a **tiled fp32 fused kernel**; the fp64+SMEM-trunk sibling is the
accuracy-safe fallback. Not captured this session: clean same-session fp64 pallas baseline
(prior 96% used); associative-reordering on-device (verdict was DEAD on CPU → not run).

## Phase-2 exit criteria / decision to record
- ✅ fp32 spill regime characterized: **register-resident (1936 B), regime still HBM-bound on
  wall 2 (MEM 83%)** — fp32 necessary, not sufficient.
- ✅ Associative-reordering verdict: **DEAD (CPU, decisive)** — not run on-device.
- ⬜ Fused-kernel regime: **blocked — whole-grid prototype won't lower (power-of-2)**; needs
  the tiled SMEM rewrite first. **This is now the make-or-break build.**
- → README 3.2c/3.2d + `[[bssn-codegen-staging]]` updated. The "compute-bound without CUDA?"
  question is **still open**, now narrowed to: *does the tiled fp32 fused kernel flip MEM%↓?*
