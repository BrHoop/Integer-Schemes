"""Step 3.2-seam gate: the fused (2.5D + algebra) peak-live cost model.

`fused_peak_model.py` answers whether the FUSED RHS kernel — where the 138 derivatives are
produced on-chip and held live for the point-wise algebra — fits the register file, and at what
SMEM/slab cost. The 1c CUDA result ("255 reg / ~4 KB spill, wall B mild") was the STANDALONE
algebra, with derivatives as HBM *inputs*; fusion makes them held nodes and reopens wall B.

These tests pin the model's premises and its decision-relevant numbers so a refreshed CSE (or a
changed ghost width / register file) re-triggers the analysis instead of silently shifting the
slab recommendation:

  1. **Premise** — the derivatives are leaves in the algebra DAG (the "reserve" the standalone
     model excludes); there are exactly 138 of them (72 grad1 + 33 diag + 33 mixed).
  2. **The seam** — holding all 138 in registers blows the working set past 4× the 255-reg file
     and adds ~120 deriv-values over the standalone algebra → wall B reopens.
  3. **The lever** — all-SMEM residency is feasible only up to a small slab (T=13); the handoff's
     T=32 hosts only 14/138 deriv outputs; recompute-from-window is SMEM-unfavorable.
  4. **Tiebreakers** — T=8 is 2 blocks/SM and WGMMA-aligned (T²=64 = M-tile); the 255-reg algebra
     caps occupancy regardless of slab; the field window is sized by KO's NG=4.
"""

import collections

import pytest

from bssn3d import fused_peak_model as M
from bssn3d.staging import build_dag, min_liveness_order, _peak_overlap
from bssn3d._codegen import parse, DENDRO_CSE


@pytest.fixture(scope="module")
def ctx():
    statements, _, _ = parse(DENDRO_CSE)
    dag = build_dag(statements)
    return statements, dag, dag.order, min_liveness_order(dag)


# --- 1. premise: derivatives are leaves, and there are exactly 138 -----------------------------
def test_derivatives_are_leaves_not_dag_nodes(ctx):
    """The model's whole reason to exist: the algebra DAG treats grad_* as leaves (excluded from
    the live set = the 'derivative-read reserve'), which is the standalone-1c view fusion breaks."""
    statements, dag, order, _ = ctx
    derivs = M.extract_derivs(statements, order)
    nodeset = set(dag.order)
    assert not (set(derivs) & nodeset)          # no derivative is a DAG node
    # and they really are referenced by the algebra (leaves with consumers, not dead)
    assert all(d.fanout >= 1 for d in derivs.values())


def test_derivative_count_and_families(ctx):
    statements, _, order, _ = ctx
    derivs = M.extract_derivs(statements, order)
    fam = collections.Counter(d.family for d in derivs.values())
    assert len(derivs) == 138
    assert fam == {"g1": 72, "g2_diag": 33, "g2_mixed": 33}
    assert sum(d.fanout for d in derivs.values()) == 603


# --- 2. the seam: holding all derivs in registers reopens wall B -------------------------------
def test_standalone_algebra_peak(ctx):
    """The 1c regime: derivs are inputs, so only the algebra temps press → 453 (file) / 424
    (ptxas reorder). Reorder never worse than file order."""
    _, dag, order, reorder = ctx
    file_peak = _peak_overlap(M.algebra_spans(dag, order))
    reord_peak = _peak_overlap(M.algebra_spans(dag, reorder))
    assert file_peak == 453
    assert reord_peak == 424
    assert reord_peak <= file_peak


def test_seam_reopens_wall_b(ctx):
    """Holding all 138 derivs in registers: the fused working set jumps past 4× the register file
    and adds ~120 deriv-values the standalone algebra never carried."""
    statements, dag, order, reorder = ctx
    d_file = M.extract_derivs(statements, order)
    d_reord = M.extract_derivs(statements, reorder)
    seam_file = M.fused_peak_values(dag, order, d_file, reg_names=list(d_file))
    seam_reord = M.fused_peak_values(dag, reorder, d_reord, reg_names=list(d_reord))
    assert seam_file == 565
    assert seam_reord == 544
    # > 4× the 255-reg file in fp64 → not the 1c "mild" regime
    assert seam_reord * M.REG_PER_FP64 > 4 * M.REG_FILE
    # fusion adds ~120 deriv-values over the standalone algebra peak (424)
    assert seam_reord - 424 >= 100


# --- 3. the lever: SMEM residency, and the slab-width frontier ----------------------------------
def test_all_smem_feasible_up_to_T13(ctx):
    """All 138 deriv tiles + a field window fit SMEM only at a small slab: T=13 fits, T=14 does
    not. The boundary is what makes the slab small."""
    assert M.peak_smem_bytes(138, 13) <= M.SMEM_CAP
    assert M.peak_smem_bytes(138, 14) > M.SMEM_CAP
    # the register pool is unchanged (algebra only) → same ~4 KB spill as 1c
    statements, dag, order, _ = ctx
    derivs = M.extract_derivs(statements, order)
    assert M.all_smem(dag, order, derivs, 8).reg_peak_values == 453


def test_handoff_T32_ruled_out():
    """At T=32 only 14 of 138 deriv outputs fit SMEM (a field window alone is ~112 KB); the rest
    can live in neither registers nor resident windows. The handoff §5.2 slab is too wide."""
    assert M.peak_smem_bytes(138, 32) > M.SMEM_CAP
    assert M.max_smem_derivs_at_T(32) == 14
    assert M.max_smem_derivs_at_T(32) < 138


def test_recompute_from_window_is_smem_unfavorable(ctx):
    """Recomputing a field's derivs frees its T² tiles but forces its (T+2NG)²·9 window resident
    into phase 2 — the window always exceeds the tiles freed. So: store in SMEM, don't recompute."""
    statements, _, order, _ = ctx
    derivs = M.extract_derivs(statements, order)
    per_field = collections.Counter(d.fld for d in derivs.values())
    maxd = max(per_field.values())
    for T in (8, 12, 32):
        assert not M.recompute_is_smem_favorable(T, maxd)


# --- 4. tiebreakers: occupancy, WGMMA alignment, KO ghost width ---------------------------------
def test_T8_is_two_blocks_and_wgmma_aligned():
    """T=8 wins both secondary axes: 2 blocks/SM (vs T=13's 1) and T²=64 = one WGMMA M-tile
    (Phase-4 Ozaki). The price is 4× halo redundancy vs T=13's 2.61×."""
    assert M.peak_smem_bytes(138, 8) == 89088
    assert M.blocks_per_sm(M.peak_smem_bytes(138, 8)) == 2
    assert M.blocks_per_sm(M.peak_smem_bytes(138, 13)) == 1
    assert M.wgmma_aligned(8) and not M.wgmma_aligned(13)
    assert M.redundancy(8) == pytest.approx(4.0)
    assert M.redundancy(13) == pytest.approx(2.6095, abs=1e-3)


def test_occupancy_is_register_capped():
    """The 255-reg algebra caps the kernel at ~257 threads/SM (~8 warps) regardless of slab — the
    1d low-occupancy regime. (Whether the latency-bound derivative stage tolerates it is the open
    GPU question, not a CPU-decidable one.)"""
    assert 250 <= M.REGS_PER_SM // M.REG_FILE <= 260


def test_field_window_sized_by_ko_ng4():
    """The 2.5D field window is sized by KO's 8th-order ghost width NG=4 (9 z-planes), NOT the
    6th-order derivative's NG=3 — KO stays load-bearing for the tile budget regardless of scheme."""
    assert M.NG == 4
    assert M.field_window_bytes(8) == (8 + 2 * 4) ** 2 * (2 * 4 + 1) * 8
