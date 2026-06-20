"""GPU hardware profiles — the single source of truth for device-dependent kernel tuning.

The fused-BSSN cost models (`fused_peak_model.py`, `tc_redundancy_tradeoff.py`) and, eventually,
the CUDA kernel's runtime slab selection all depend on a handful of per-architecture facts: how
much **shared memory** a block can opt into (this caps the 2.5D slab width T, since all 138
derivative tiles must be SMEM-resident), the **register file** (caps the algebra working set and
occupancy), the **threads/SM** (occupancy), and the **tensor-core MMA shape** (the Phase-4 Ozaki
alignment). Hard-coding H200 numbers makes the kernel hyper-specialised to one device; this module
makes the tuning a *function of the GPU* so the same code is optimal across Hopper, Ampere, Ada,
Volta, Blackwell.

The slab-selection logic (`max_smem_slab`) takes the SMEM cap as a plain int, so the **runtime**
kernel can call the exact same function with the value it queries from
`cudaDevAttrMaxSharedMemoryPerBlockOptin` — design-time recommendation and run-time selection share
one code path.

SMEM-per-block values are the documented `MaxSharedMemoryPerBlockOptin` per compute capability
(CUDA C Programming Guide, Table "Technical Specifications per Compute Capability"). Bandwidth /
peak-FLOP mirror `mcs3d.benchmark._BW_TABLE` / `_PEAK_TABLE`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

KB = 1024
MB = 1024 * 1024


@dataclass(frozen=True)
class GPUProfile:
    """Per-device facts that drive fused-kernel tuning."""
    name: str
    sm_arch: str                      # nvcc -arch target, e.g. "sm_90a"
    smem_per_block: int               # bytes — MaxSharedMemoryPerBlockOptin (caps the slab)
    regs_per_thread: int = 255        # architectural max usable per thread
    regs_per_sm: int = 65536          # 32-bit register file per SM
    threads_per_sm: int = 2048        # hardware thread cap per SM (occupancy)
    l2_bytes: int = 40 * MB           # L2 size (the "24 fields >> L2" wall-A argument)
    peak_bw_GBs: float = 0.0          # HBM bandwidth
    peak_fp64_GFLOPs: float = 0.0     # fp64 vector peak
    peak_int8_TOPs: float = 0.0       # INT8 tensor-core peak (Phase-4 Ozaki)
    tc_mma: str = "mma"               # "wgmma" (Hopper m64) | "mma" (Ampere/Ada) | "none"
    tc_m_tile: int = 16               # MMA M dimension → point-batch alignment granularity


# Registry. SMEM-per-block = MaxSharedMemoryPerBlockOptin for the compute capability:
#   sm_70 V100 96 KB · sm_75 T4 64 KB · sm_80 A100 163 KB · sm_86/89 (A40/L40/RTX) 99 KB ·
#   sm_90 Hopper 227 KB · sm_100 Blackwell 227 KB.
PROFILES: Dict[str, GPUProfile] = {
    "H200": GPUProfile(
        "H200", "sm_90a", 227 * KB, l2_bytes=50 * MB,
        peak_bw_GBs=4800, peak_fp64_GFLOPs=34000, peak_int8_TOPs=1979,
        tc_mma="wgmma", tc_m_tile=64),
    "H100": GPUProfile(
        "H100", "sm_90a", 227 * KB, l2_bytes=50 * MB,
        peak_bw_GBs=3350, peak_fp64_GFLOPs=34000, peak_int8_TOPs=1979,
        tc_mma="wgmma", tc_m_tile=64),
    "GH200": GPUProfile(
        "GH200", "sm_90a", 227 * KB, l2_bytes=50 * MB,
        peak_bw_GBs=4900, peak_fp64_GFLOPs=34000, peak_int8_TOPs=1979,
        tc_mma="wgmma", tc_m_tile=64),
    "B200": GPUProfile(
        "B200", "sm_100a", 227 * KB, l2_bytes=50 * MB,
        peak_bw_GBs=8000, peak_fp64_GFLOPs=40000, peak_int8_TOPs=4500,
        tc_mma="wgmma", tc_m_tile=64),
    "A100": GPUProfile(
        "A100", "sm_80", 163 * KB, l2_bytes=40 * MB,
        peak_bw_GBs=2039, peak_fp64_GFLOPs=9700, peak_int8_TOPs=624,
        tc_mma="mma", tc_m_tile=16),
    "A40": GPUProfile(   # sm_86 server Ampere (also ~A6000, RTX 3090)
        "A40", "sm_86", 99 * KB, threads_per_sm=1536, l2_bytes=6 * MB,
        peak_bw_GBs=696, peak_fp64_GFLOPs=584, peak_int8_TOPs=299,
        tc_mma="mma", tc_m_tile=16),
    "L40": GPUProfile(   # sm_89 Ada (also ~L4, RTX 4090)
        "L40", "sm_89", 99 * KB, threads_per_sm=1536, l2_bytes=96 * MB,
        peak_bw_GBs=864, peak_fp64_GFLOPs=1414, peak_int8_TOPs=724,
        tc_mma="mma", tc_m_tile=16),
    "V100": GPUProfile(  # sm_70 — fp16 tensor cores only (no INT8 MMA → Ozaki N/A)
        "V100", "sm_70", 96 * KB, l2_bytes=6 * MB,
        peak_bw_GBs=900, peak_fp64_GFLOPs=7800, peak_int8_TOPs=0.0,
        tc_mma="none", tc_m_tile=0),
}

DEFAULT = "H200"


def get(name: str) -> GPUProfile:
    """Look up a profile by (case-insensitive, fuzzy) name; raises if unknown."""
    if name in PROFILES:
        return PROFILES[name]
    key = name.strip().upper().replace(" ", "")
    for k, v in PROFILES.items():
        if k.upper() == key or k.upper() in key or key in k.upper():
            return v
    raise KeyError(f"unknown GPU {name!r}; known: {', '.join(PROFILES)}")


# --- slab selection (shared by the CPU cost model AND the runtime kernel) ---------------------
def field_window_bytes(T: int, ng: int = 4, dtype_bytes: int = 8) -> int:
    """One field's 2.5D SMEM window: (T+2ng)² in-plane halo × (2ng+1) z-planes."""
    return (T + 2 * ng) ** 2 * (2 * ng + 1) * dtype_bytes


def slab_smem_bytes(T: int, n_derivs: int = 138, ng: int = 4, dtype_bytes: int = 8) -> int:
    """Peak SMEM of the fused slab: 138 T²-deriv-tiles (persistent) + one field window."""
    tiles = n_derivs * T * T * dtype_bytes
    return tiles + field_window_bytes(T, ng, dtype_bytes)


def max_smem_slab(smem_per_block: int, n_derivs: int = 138, ng: int = 4,
                  dtype_bytes: int = 8, t_max: int = 64) -> Optional[int]:
    """Largest slab width T whose deriv tiles + field window fit ``smem_per_block``.

    Takes the SMEM cap as a plain int so the **runtime** kernel can pass the value queried from
    ``cudaDevAttrMaxSharedMemoryPerBlockOptin`` — same logic, design-time and run-time."""
    best = None
    for T in range(4, t_max + 1):
        if slab_smem_bytes(T, n_derivs, ng, dtype_bytes) <= smem_per_block:
            best = T
    return best


def redundancy(T: int, ng: int = 4) -> float:
    """In-plane halo recompute factor of a z-marched slab: ((T+2ng)/T)²."""
    return ((T + 2 * ng) / T) ** 2


def tc_aligned(T: int, profile: GPUProfile) -> bool:
    """Does the slab's T² point-batch fill whole MMA M-tiles (Phase-4 Ozaki)? Hopper WGMMA wants
    M=64; Ampere/Ada `mma.sync` wants M=16; Volta has no INT8 MMA."""
    if profile.tc_m_tile <= 0:
        return False
    return (T * T) % profile.tc_m_tile == 0


@dataclass(frozen=True)
class SlabRec:
    gpu: str
    smem_cap_kb: int
    T_fp64: int                 # max SMEM-feasible slab (lowest redundancy)
    redundancy_fp64: float
    blocks_per_sm: int
    T_tc: Optional[int]         # largest MMA-aligned feasible slab (Phase-4)
    redundancy_tc: Optional[float]
    occ_threads_per_sm: int     # register-capped occupancy ceiling
    tc_mma: str


def recommend(profile: GPUProfile, n_derivs: int = 138, ng: int = 4) -> SlabRec:
    """The optimal fused-slab configuration for ``profile`` (all-SMEM derivs, fp64 algebra)."""
    T = max_smem_slab(profile.smem_per_block, n_derivs, ng)
    blocks = profile.smem_per_block // slab_smem_bytes(T, n_derivs, ng) if T else 0
    aligned = [t for t in range(4, (T or 0) + 1) if tc_aligned(t, profile)]
    T_tc = max(aligned) if aligned else None
    # the 255-reg algebra caps occupancy regardless of slab
    occ = min(profile.threads_per_sm, profile.regs_per_sm // profile.regs_per_thread)
    return SlabRec(
        gpu=profile.name, smem_cap_kb=profile.smem_per_block // KB,
        T_fp64=T, redundancy_fp64=redundancy(T, ng), blocks_per_sm=blocks,
        T_tc=T_tc, redundancy_tc=redundancy(T_tc, ng) if T_tc else None,
        occ_threads_per_sm=occ, tc_mma=profile.tc_mma)


def main() -> None:
    print("=" * 92)
    print("Fused-BSSN slab recommendation per GPU (all 138 derivs SMEM-resident, fp64, NG=4)")
    print("=" * 92)
    print(f"  {'GPU':>6} {'arch':>8} {'SMEM/blk':>9} {'T_fp64':>7} {'halo':>6} {'blk/SM':>7} "
          f"{'T_tc(MMA)':>10} {'halo_tc':>8} {'occ thr/SM':>11} {'MMA':>6}")
    for name in PROFILES:
        r = recommend(PROFILES[name])
        tctxt = f"{r.T_tc}" if r.T_tc else "—"
        htc = f"{r.redundancy_tc:.2f}x" if r.redundancy_tc else "—"
        print(f"  {r.gpu:>6} {PROFILES[name].sm_arch:>8} {r.smem_cap_kb:>7}KB "
              f"{r.T_fp64:>7} {r.redundancy_fp64:>5.2f}x {r.blocks_per_sm:>7} "
              f"{tctxt:>10} {htc:>8} {r.occ_threads_per_sm:>11} {r.tc_mma:>6}")
    print("\n  T_fp64 = largest slab whose 138 fp64 deriv tiles + field window fit SMEM (lowest")
    print("  halo redundancy). T_tc = largest MMA-aligned slab (Phase-4 Ozaki tiebreaker). The")
    print("  255-reg algebra caps occupancy at ~regs_per_sm/255 threads/SM regardless of slab.")
    print("  Runtime: the kernel queries MaxSharedMemoryPerBlockOptin and calls max_smem_slab().")


if __name__ == "__main__":
    main()
