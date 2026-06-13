# Step 3.2d — fp32 tiled SMEM-halo fused BSSN kernel (the regime-flip build)

> The make-or-break build: derivatives computed **on-chip** in a tiled Pallas kernel so the
> 138-array HBM read (wall 2) disappears → the only thing that can flip the BSSN RHS off HBM.
> fp32 algebra is register-resident + accuracy-benign (measured); fp64 derivatives are
> accurate + Triton-safe. The roofline predicts compute/L2-bound once derivs are on-chip.

## Status (2026-06-13)

**Increment 1 — tiled on-chip derivative core — DONE (CPU + GPU-lowering confirmed).**
`bssn3d/tiled_deriv.py` (`tiled_derivative_bundle`, scheme not yet wired). Computes the 138
`grad_*`/`grad2_*` over power-of-2 haloed tiles via the proven `pallas_ozaki` pattern:
- Wrapper edge-pads + `vmap(dynamic_slice)`-cuts overlapping HP=16 haloed tiles (BS=8).
- Kernel: every derivative = three per-axis `(BS,HP)` matrices (`D1`/`D2` stencil, which also
  crops HP→BS, or `ID` interior-select) applied by **broadcast-multiply-reduce** (NOT `dot`/
  transpose). Unrolled per field (138 derivs).
- **CPU:** fp64 = 1.46e-13 (math exact), fp32 = 5e-5. `test_bssn_tiled_deriv.py` 2 green.
- **H200:** the kernel **LOWERS and runs** (fp32 confirmed, ~1e-4).

**Precision decision:** derivatives in **fp64** (broadcast-reduce lowers fp64 — only fp64
`dot` hangs Triton) → cast to fp32 → fp32 algebra = the validated `fp32_contraction`
precision. The fp32-everything path gives ~1e-4 (2nd-deriv cancellation); Ozaki-II is the
Phase-4 *speed* play, NOT a correctness requirement.

## Triton-0.9.2 constraints cleared (learned the hard way — keep for the fuse)
1. **2D-only `dot`** → use broadcast-multiply-reduce, not `tensordot`/`dot_general`.
2. **All array sizes power-of-2** → HP=16, BS=8; pad field count to `NF_PAD` ONLY if batching
   over fields (the unrolled per-field kernel indexes single fields → no NF_PAD needed).
3. **No `slice` on computed arrays** → select via one-hot reduce or ref-load (`tile_ref[0,fidx]`).
4. **Batching over fields is a TRAP** → the un-reduced broadcast transient ×NF ≈ 16 MB/op →
   explodes compile + spills. Keep per-field (~256 KB transients). Compile is the cost, but
   it is **cached + shape-keyed** → one-time per (resolution,config), NOT per-run/step/initial-
   data. Not a production concern; for dev use the `BSSN_TILE_NDERIV` cap to iterate fast.

## Plan for the rest (in order)

1. **Bank progress** — commit the session's work (tiled_deriv, fp32 strong-field test,
   reconciled memory/docs) to a **branch** (we're on `main`).
2. **Build the full fused tiled kernel** (CPU) — unrolled fp64 derivs (this core) + the fp32
   algebra schedule, in ONE `pallas_call`; wire as a `scheme` (e.g. `"fused_tiled"`). Per
   tile: compute the 138 derivs (BS³ interior), then the algebra as elementwise ops over the
   BS³ points → 24 outputs. The algebra-only Pallas kernel already lowers, so the combined
   one should too (slow compile, cached).
3. **Interpret-validate** (CPU) the fused kernel == verbatim RHS to round-off — correctness
   with zero GPU time.
4. **ONE decisive GPU push** — `spill_probe` + `profile_regime --smi` on the fused kernel:
   does MEM%↓/SM%↑ (off HBM)? Expect a few iterations to clear remaining Triton/SMEM limits.

## Known risk
SMEM/register pressure with derivs + algebra working set + field halo co-resident is the
genuine open question (the architecture's central tension). fp32 algebra + field-streaming
help. A **"landed L2-bound, not strictly compute-bound"** result is still a thesis-valid win
(off HBM is the real target; matches the NR-GPU literature, arXiv:2501.14030).
