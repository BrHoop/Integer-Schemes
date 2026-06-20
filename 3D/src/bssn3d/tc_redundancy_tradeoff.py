"""Is the tensor-core derivative worth the slab-redundancy it forces? (CPU break-even screen)

The Step-3.2-seam model fixed the fused slab by SMEM capacity: the best fp64 slab is **T=13**
(halo redundancy 2.61×), but using tensor cores (Ozaki INT8 / WGMMA) needs the point-batch T² to
fill an M=64 tile → **T=8** (the only SMEM-feasible WGMMA-aligned slab), whose halo redundancy is
**4.00×**. So tensor cores buy GEMM throughput but cost 4.00/2.61 = 1.53× more redundant derivative
work. Worth it?

**Key simplification.** A step = derivative_stage + algebra_stage. The algebra is point-wise
(N³·~4500 ops, NO halo, register-bound) → **identical work in both configs, cancels.** The whole
T=13-fp64 vs T=8-TC choice lives in the derivative stage:

    T=13 fp64 : deriv work ∝ redundancy(13)=2.61, on CUDA cores  (s = 1 by definition)
    T= 8  TC  : deriv work ∝ redundancy(8) =4.00, on tensor cores at *effective* speedup s

    TC wins  ⇔  redundancy(8)·s < redundancy(13)  ⇔  s < redundancy(13)/redundancy(8) = 0.652

i.e. the TC derivative must be **≥ 1/0.652 = 1.53× faster per useful op** just to break even. This
module reports that threshold, what `s` bundles (the factors that push it toward/past 1), the
total-step magnitude under Amdahl (it is small — the algebra dominates), and the sensitivity to a
*larger* TC-aligned slab (the only thing that flips the sign cheaply).

This is a SCREEN, not a verdict: the real `s` is a GPU measurement (fp64 deriv stage @T=13 vs TC
deriv stage @T=8). The model says the threshold to beat and that the derivative stage must be
COMPUTE-bound for TC to matter at all — if it is memory/latency-bound (the occupancy risk), TC does
nothing and T=13 fp64 wins by default.

Run:  `python -m bssn3d.tc_redundancy_tradeoff`
"""

from __future__ import annotations

from . import gpu_profiles
from .gpu_profiles import redundancy

# Slabs are SOURCED per-GPU from the gpu_profiles registry (single source of truth). The
# module defaults to H200; `break_even_for(profile)` / `main` report every GPU. On a device
# whose fp64 slab is already small (A100 T=11), the TC slab penalty is *smaller* → the
# break-even is easier than H200's — i.e. tensor cores are relatively more attractive there.
_REC = gpu_profiles.recommend(gpu_profiles.PROFILES[gpu_profiles.DEFAULT])   # H200
T_FP64 = _REC.T_fp64     # best SMEM-feasible fp64 slab (lowest redundancy) — H200: 13
T_TC = _REC.T_tc         # largest MMA-aligned feasible slab — H200: 8


def break_even_s(t_fp64: int = T_FP64, t_tc: int = T_TC) -> float:
    """Largest TC effective-speedup factor s (TC deriv time / fp64 deriv time, per useful op) at
    which the TC slab still wins. s < this ⇒ TC worth it. = redundancy(t_fp64)/redundancy(t_tc)."""
    return redundancy(t_fp64) / redundancy(t_tc)


def total_step_speedup(s: float, f: float, t_fp64: int = T_FP64, t_tc: int = T_TC) -> float:
    """Whole-step speedup of (TC @t_tc) over (fp64 @t_fp64), with derivative stage = fraction `f`
    of the fp64 baseline step. Algebra (1-f) is common. >1 means TC is faster overall."""
    rr = redundancy(t_tc) / redundancy(t_fp64)          # extra redundancy of the TC slab
    tc_step = (1.0 - f) + f * rr * s
    return 1.0 / tc_step


def break_even_for(profile) -> dict:
    """Per-GPU TC break-even: uses the GPU's own fp64 slab (T_fp64) and MMA-aligned slab (T_tc).
    A device whose fp64 slab is already small has a smaller TC redundancy penalty → easier
    break-even (TC relatively more attractive)."""
    rec = gpu_profiles.recommend(profile)
    if rec.T_tc is None:
        return {"gpu": profile.name, "T_fp64": rec.T_fp64, "T_tc": None, "s_star": None}
    s_star = break_even_s(rec.T_fp64, rec.T_tc)
    return {"gpu": profile.name, "T_fp64": rec.T_fp64, "T_tc": rec.T_tc,
            "red_fp64": redundancy(rec.T_fp64), "red_tc": redundancy(rec.T_tc),
            "s_star": s_star, "speedup_needed": 1.0 / s_star, "mma": profile.tc_mma}


def main() -> None:
    s_star = break_even_s()
    rr = redundancy(T_TC) / redundancy(T_FP64)

    print("=" * 86)
    print("Tensor-core derivative vs the slab-redundancy it forces — break-even screen")
    print("=" * 86)
    _h200 = gpu_profiles.PROFILES["H200"]
    print(f"  fp64 best slab  T={T_FP64}: redundancy {redundancy(T_FP64):.2f}x  "
          f"(MMA-aligned: {gpu_profiles.tc_aligned(T_FP64, _h200)})")
    print(f"  TC   slab       T={T_TC}: redundancy {redundancy(T_TC):.2f}x  "
          f"(MMA-aligned: {gpu_profiles.tc_aligned(T_TC, _h200)})  → {rr:.2f}x more redundant work")
    print(f"\n  BREAK-EVEN: TC wins iff effective s < {s_star:.3f}  "
          f"(i.e. TC derivative ≥ {1/s_star:.2f}× faster per useful op).")
    print(f"  The algebra stage is identical in both configs (point-wise, no halo) → it CANCELS;")
    print(f"  the decision is the derivative stage alone, so this threshold is f-independent.\n")

    # what `s` bundles -------------------------------------------------------------------------
    print("-" * 86)
    print("What the effective s bundles (all push s UP, toward/past the 0.65 threshold):")
    print("-" * 86)
    print("  • Ozaki-II multi-pass: a bit-accurate fp64 result is several INT8 GEMMs (trunc level)")
    print("    → s multiplied by the pass count; a 2:1 emulation alone nearly eats the 1.53× budget.")
    print("  • Small-K zero-waste: the stencil contraction is K=7 (deriv) / K=9 (KO) padded to the")
    print("    INT8 K=32 tile → ~4.6× wasted MACs unless 2:4-sparse/strided (SPIDER) reclaims it.")
    print("  • Achieved-vs-peak: small/banded GEMMs rarely hit peak TC throughput.")
    print("  • Cast + recombine (fp64→INT8 split, partial-sum reassembly) is non-GEMM overhead.")
    print("  → realistic s is NOT obviously < 0.65; this is precisely why it must be MEASURED.\n")

    # total-step magnitude (Amdahl) ------------------------------------------------------------
    print("-" * 86)
    print("IF TC clears the bar, how big is the WHOLE-step win? (Amdahl: algebra dominates)")
    print("  deriv-stage fraction f of the fp64 T=13 baseline step, across plausible TC speedups s")
    print("-" * 86)
    s_grid = (0.20, 0.30, 0.45, 0.652, 0.80)
    print(f"  {'f \\ s':>7} " + " ".join(f"{s:>7.3f}" for s in s_grid)
          + "   (s=1/x ⇒ TC x× faster/op)")
    for f in (0.25, 0.35, 0.45):
        cells = []
        for s in s_grid:
            sp = total_step_speedup(s, f)
            cells.append(f"{sp:6.2f}x")
        print(f"  {f:>7.2f} " + " ".join(f"{c:>7}" for c in cells))
    print(f"\n  read: at s=0.652 every cell is 1.00x (break-even, by construction). Even an")
    print(f"  aggressive s=0.30 (TC ~3.3×/op) gives only ~1.1–1.3× whole-step — the T=8 redundancy")
    print(f"  tax (1.53×) plus the ~70% algebra cap (Amdahl, the original ~1.4× TC ceiling) eat it.\n")

    # the cheap sign-flip: a larger TC-aligned slab --------------------------------------------
    print("-" * 86)
    print("The only cheap way to flip the sign: a LARGER WGMMA-aligned slab (lower redundancy)")
    print("-" * 86)
    print(f"  {'T_tc':>5} {'aligned':>8} {'redund':>7} {'break-even s':>13} {'vs T=13 fp64'}")
    for t in (8, 16, 24, 32):
        bs = break_even_s(T_FP64, t)
        note = ("needs >227KB SMEM for 138 fp64 deriv tiles → INFEASIBLE unless deriv storage"
                if t >= 16 else "the only SMEM-feasible TC slab today")
        flip = "TC slab already LOWER redundancy" if redundancy(t) < redundancy(T_FP64) else ""
        print(f"  {t:5d} {str(gpu_profiles.tc_aligned(t, _h200)):>8} {redundancy(t):6.2f}x "
              f"{bs:12.3f}  {flip or note}")
    print(f"\n  → at T=16 (redundancy 2.25 < 2.61) the TC slab would be CHEAPER than the fp64 slab")
    print(f"    and break-even jumps to s<{break_even_s(T_FP64,16):.2f} (easy). But T=16 needs 276 KB")
    print(f"    of fp64 deriv tiles (seam model) → only reachable if derivs can be stored in <8 B")
    print(f"    (fp32/INT8), which is the SAME accuracy question fp32 lost. That is the real lever.\n")

    # per-GPU break-even -----------------------------------------------------------------------
    print("-" * 86)
    print("PER-GPU break-even (each device uses ITS OWN fp64 slab vs MMA-aligned slab)")
    print("-" * 86)
    print(f"  {'GPU':>6} {'T_fp64':>7} {'T_tc':>5} {'redund fp64/tc':>15} {'break-even s':>13} "
          f"{'TC must be':>12}")
    for nm in gpu_profiles.PROFILES:
        b = break_even_for(gpu_profiles.PROFILES[nm])
        if b["s_star"] is None:
            print(f"  {nm:>6} {b['T_fp64']:>7} {'—':>5} {'(no INT8 MMA)':>15}")
            continue
        print(f"  {nm:>6} {b['T_fp64']:>7} {b['T_tc']:>5} "
              f"{b['red_fp64']:>6.2f}/{b['red_tc']:<6.2f}    {b['s_star']:>11.3f}  "
              f"{b['speedup_needed']:>9.2f}×")
    print("  → smaller-SMEM GPUs have a smaller fp64 slab, so the TC slab's extra redundancy is")
    print("    LESS → easier break-even. On A40/L40/V100 fp64 and TC share T=8 (no penalty: s<1).\n")

    print("=" * 86)
    print("VERDICT (screen)")
    print("=" * 86)
    print(f"• Decision collapses to ONE measurable number: the TC derivative stage must run")
    print(f"  ≥ {1/s_star:.2f}× faster per useful op than the fp64 stage to overcome the T=8 redundancy.")
    print(f"• Realistic s (Ozaki multi-pass + K=7 zero-waste + small-GEMM under-util) is NOT clearly")
    print(f"  under 0.65, and even if it is, the whole-step win is only ~1.1–1.3× (algebra-capped).")
    print(f"• So default to T=13 fp64; pursue TC only if a measurement shows s<0.65 AND ideally if a")
    print(f"  larger aligned slab (T=16) is unlocked by lower-precision deriv storage.")
    print(f"• Prerequisite: the derivative stage must be COMPUTE-bound. If it is memory/latency-bound")
    print(f"  (the open occupancy risk), TC buys nothing and T=13 fp64 wins outright.")
    print(f"• MEASURE: fp64 deriv-stage time @T=13  vs  TC deriv-stage time @T=8 (the Ozaki path).")
    print("=" * 86)


if __name__ == "__main__":
    main()
