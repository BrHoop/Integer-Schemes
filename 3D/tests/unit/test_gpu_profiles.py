"""GPU profile registry + per-device slab selection (the multi-GPU tuning source of truth).

`gpu_profiles` makes the fused-kernel slab a function of the device rather than hard-coded to
H200. These tests pin the per-architecture slab the SMEM cap implies (Hopper T=13, A100 T=11,
Ampere/Ada T=8), the MMA-alignment tiebreaker, and the runtime/design-time parity of the
selection logic (the kernel will call `max_smem_slab` with a queried `MaxSharedMemoryPerBlockOptin`
— same function).
"""

import pytest

from bssn3d import gpu_profiles as gp


def test_registry_has_core_gpus():
    for name in ("H200", "A100", "A40", "L40", "V100"):
        assert name in gp.PROFILES
    # SMEM-per-block ordering reflects the compute-capability table
    assert gp.PROFILES["H200"].smem_per_block == 227 * 1024
    assert gp.PROFILES["A100"].smem_per_block == 163 * 1024
    assert gp.PROFILES["A40"].smem_per_block == 99 * 1024


# (cap_bytes, expected max all-SMEM slab) for the documented per-block SMEM caps
@pytest.mark.parametrize("gpu,T", [("H200", 13), ("A100", 11), ("A40", 8), ("L40", 8), ("V100", 8)])
def test_max_smem_slab_per_gpu(gpu, T):
    cap = gp.PROFILES[gpu].smem_per_block
    assert gp.max_smem_slab(cap) == T
    # the chosen slab fits, and one wider does NOT (it is the true max)
    assert gp.slab_smem_bytes(T) <= cap
    assert gp.slab_smem_bytes(T + 1) > cap


def test_runtime_design_time_parity():
    """The kernel will call max_smem_slab with a value queried from the device at runtime; an
    arbitrary cap must select consistently with slab_smem_bytes (no hidden device coupling)."""
    for cap in (96 * 1024, 120 * 1024, 163 * 1024, 200 * 1024, 227 * 1024):
        T = gp.max_smem_slab(cap)
        assert gp.slab_smem_bytes(T) <= cap < gp.slab_smem_bytes(T + 1)


def test_mma_alignment_is_per_arch():
    h200, a100, v100 = gp.PROFILES["H200"], gp.PROFILES["A100"], gp.PROFILES["V100"]
    # Hopper WGMMA M=64 → T² multiple of 64 (T multiple of 8)
    assert gp.tc_aligned(8, h200) and not gp.tc_aligned(13, h200)
    # Ampere mma M=16 → T² multiple of 16 (gentler): T=8 and T=12 align, T=11 does not
    assert gp.tc_aligned(8, a100) and gp.tc_aligned(12, a100) and not gp.tc_aligned(11, a100)
    # Volta has no INT8 MMA → never aligned (Ozaki N/A)
    assert not gp.tc_aligned(8, v100)


def test_recommend_consistency():
    for name, prof in gp.PROFILES.items():
        r = gp.recommend(prof)
        assert r.T_fp64 == gp.max_smem_slab(prof.smem_per_block)
        assert gp.slab_smem_bytes(r.T_fp64) <= prof.smem_per_block
        assert r.blocks_per_sm >= 1
        # occupancy is register-capped (~regs_per_sm/255), independent of slab
        assert r.occ_threads_per_sm == min(prof.threads_per_sm,
                                           prof.regs_per_sm // prof.regs_per_thread)
        if r.T_tc is not None:                      # MMA-aligned slab must itself be feasible
            assert r.T_tc <= r.T_fp64 and gp.tc_aligned(r.T_tc, prof)


def test_smaller_smem_gives_smaller_or_equal_slab():
    """Monotonicity: a device with less SMEM never gets a wider slab (the tuning is sane)."""
    by_cap = sorted(gp.PROFILES.values(), key=lambda p: p.smem_per_block)
    slabs = [gp.recommend(p).T_fp64 for p in by_cap]
    assert slabs == sorted(slabs)


def test_get_fuzzy_lookup():
    assert gp.get("A100") is gp.PROFILES["A100"]
    assert gp.get("a100").name == "A100"
    with pytest.raises(KeyError):
        gp.get("TPUv5")
