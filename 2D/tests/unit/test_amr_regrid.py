"""
Unit tests for the AMR regridding pipeline.

Covers:
  * `compute_indicator_gradient` — kernel returns the analytic max |∇field|
    for a smooth synthetic field, and zero for a flat field.
  * `compute_flags` — hysteresis correctly delays refinement until K consecutive
    cycles above threshold.
  * `apply_flags` — REFINE actually creates 4 child slots with correctly
    prolongated data; COARSEN removes a fine slot and restores into the parent.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import (
    BS, NG, NF, MAX_BLOCKS, LEVELS, REFINE_RATIO,
    AMRState, AMRTopology, make_root_state,
)
from mcs2d.amr.kernels import (
    prolongate, restrict_into_parent,
    compute_indicator_gradient,
)
from mcs2d.amr.regrid import (
    compute_flags, enforce_nesting_buffer, apply_flags, regrid,
    REFINE, KEEP, COARSEN,
)


EZ_IDX = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_blocks_field(func, n_active: int = 1) -> jnp.ndarray:
    """Build (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG) where the first n_active blocks
    have field EZ filled by `func(x_idx, y_idx)`, others are zero."""
    coords = np.arange(BS + 2*NG)
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    field = func(X, Y)
    blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=np.float64)
    for s in range(n_active):
        blocks[s, EZ_IDX] = field
    return jnp.asarray(blocks)


# ── Indicator kernel ──────────────────────────────────────────────────────────

class TestIndicatorGradient:

    def test_flat_field_returns_zero(self):
        blocks = _make_blocks_field(lambda X, Y: np.ones_like(X) * 3.14, n_active=1)
        out = np.asarray(compute_indicator_gradient(blocks, dx=0.1, field_idx=EZ_IDX))
        assert out.shape == (MAX_BLOCKS,)
        assert np.max(np.abs(out)) < 1e-14, f"flat field: max |∇| = {np.max(out):.2e}"

    def test_linear_x_field(self):
        """For f = a*x with cell index x, ∇f = a / dx_phys per cell.  After
        dividing by 2*dx_phys (centered diff), result = a (constant)."""
        a = 0.7
        dx = 0.05
        # f at index i = a * i (in index units).  Physical x = i * dx.
        # In physical units, ∂f/∂x = a / dx.
        blocks = _make_blocks_field(lambda X, Y: a * X.astype(np.float64), n_active=1)
        out = np.asarray(compute_indicator_gradient(blocks, dx=dx, field_idx=EZ_IDX))
        expected = a / dx
        # All interior cells have the same gradient → max = expected.
        assert abs(out[0] - expected) < 1e-12, \
            f"got {out[0]:.6e}, expected {expected:.6e}"
        # Inactive blocks return 0.
        assert np.all(out[1:] == 0.0)

    def test_sinusoidal_field_returns_max_gradient(self):
        """For f = sin(k*x), max |∇f| = k.  Test with a wavelength large enough
        that the discrete maximum is close to k."""
        # Use coordinates in physical units, dx = 1.0 for simplicity.
        # f(i) = sin(k * i * dx) with dx = 0.5, k = π/16 → wavelength 32 cells.
        dx = 0.5
        k = np.pi / 16.0
        blocks = _make_blocks_field(
            lambda X, Y: np.sin(k * X.astype(np.float64) * dx), n_active=1
        )
        out = np.asarray(compute_indicator_gradient(blocks, dx=dx, field_idx=EZ_IDX))
        # Max |∇f| in physical units = k.  Centered FD underestimates slightly.
        # Tolerance 10% — order-of-magnitude check is the point here.
        assert 0.8 * k < out[0] < 1.1 * k, \
            f"got {out[0]:.4e}, expected ~{k:.4e}"


# ── Hysteresis flag logic ─────────────────────────────────────────────────────

class TestComputeFlags:

    def test_streak_must_reach_K_to_refine(self):
        """A block with consistently high indicator should be flagged only
        after K consecutive cycles."""
        topo = AMRTopology()
        topo.add_block(0, slot=0, bbox_ij=(0, 0))
        # Indicator above threshold for level 0; rest blank.
        high = np.zeros(MAX_BLOCKS)
        high[0] = 5.0
        low  = np.zeros(MAX_BLOCKS)   # below coarsen_threshold

        K = 3
        ind_per_level = [high] + [np.zeros(MAX_BLOCKS)] * (LEVELS - 1)
        for cycle in range(K):
            flags = compute_flags(
                ind_per_level, topo,
                refine_threshold=1.0, coarsen_threshold=0.1,
                hysteresis_K=K,
            )
            if cycle < K - 1:
                assert flags[0, 0] == KEEP, \
                    f"cycle {cycle}: shouldn't be flagged yet"
            else:
                assert flags[0, 0] == REFINE, \
                    f"cycle {cycle}: should be flagged now"

    def test_decay_resets_streak(self):
        """A streak should decay if the indicator drops back into the keep
        zone — prevents refining on a single spike."""
        topo = AMRTopology()
        topo.add_block(0, slot=0, bbox_ij=(0, 0))
        high = np.zeros(MAX_BLOCKS); high[0] = 5.0
        mid  = np.zeros(MAX_BLOCKS); mid[0]  = 0.5
        K = 3

        # 2 high cycles → streak = 2 (not yet refined)
        for _ in range(2):
            compute_flags([high] + [np.zeros(MAX_BLOCKS)] * (LEVELS - 1), topo,
                          refine_threshold=1.0, coarsen_threshold=0.1, hysteresis_K=K)
        assert topo.streaks[0, 0] == 2

        # 1 mid cycle → streak decays to 1
        compute_flags([mid] + [np.zeros(MAX_BLOCKS)] * (LEVELS - 1), topo,
                      refine_threshold=1.0, coarsen_threshold=0.1, hysteresis_K=K)
        assert topo.streaks[0, 0] == 1

        # Now even 1 high won't flag — need K consecutive again.
        flags = compute_flags(
            [high] + [np.zeros(MAX_BLOCKS)] * (LEVELS - 1), topo,
            refine_threshold=1.0, coarsen_threshold=0.1, hysteresis_K=K)
        assert flags[0, 0] == KEEP
        assert topo.streaks[0, 0] == 2

    def test_root_level_cannot_coarsen(self):
        """Level 0 blocks must never be flagged for coarsening — they're the
        root tiling, which has no parent."""
        topo = AMRTopology()
        topo.add_block(0, slot=0, bbox_ij=(0, 0))
        low = np.zeros(MAX_BLOCKS)   # below coarsen threshold
        K = 2
        for _ in range(K):
            flags = compute_flags([low] + [np.zeros(MAX_BLOCKS)] * (LEVELS - 1),
                                  topo, refine_threshold=1.0,
                                  coarsen_threshold=0.5, hysteresis_K=K)
        # Indicator 0 < coarsen threshold 0.5, would normally coarsen, but
        # level 0 is protected.
        assert flags[0, 0] == KEEP, f"root coarsened: flags={flags[0, 0]}"


# ── Nesting buffer ────────────────────────────────────────────────────────────

class TestNestingBuffer:
    """enforce_nesting_buffer: buffer dilation + no-orphan coarsening."""

    def _root_grid_topo(self, nbx, nby):
        """A topology with an nbx×nby root tiling, all active."""
        topo = AMRTopology()
        for bi in range(nbx):
            for bj in range(nby):
                slot = bi * nby + bj
                topo.add_block(0, slot, (bi * BS, bj * BS))
        return topo

    def test_buffer_dilates_to_neighbors(self):
        """A single REFINE flag should spread to its 4 edge-neighbours (and, at
        n_buffer=1, the diagonal too — Chebyshev distance)."""
        nbx, nby = 4, 4
        topo = self._root_grid_topo(nbx, nby)
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        # Flag the block at grid (1,1) = slot 1*4+1 = 5.
        center_slot = 1 * nby + 1
        flags[0, center_slot] = REFINE

        out = enforce_nesting_buffer(flags, topo, n_buffer=1)

        # All 8 Chebyshev-neighbours of (1,1) plus itself → 9 blocks REFINE.
        refined = {s for s in range(MAX_BLOCKS) if out[0, s] == REFINE}
        expected = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                bi, bj = 1 + di, 1 + dj
                if 0 <= bi < nbx and 0 <= bj < nby:
                    expected.add(bi * nby + bj)
        assert refined == expected, f"got {refined}, expected {expected}"

    def test_buffer_zero_is_noop(self):
        """n_buffer=0 should not dilate (reach 0 → only the flagged block)."""
        nbx, nby = 4, 4
        topo = self._root_grid_topo(nbx, nby)
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 5] = REFINE
        out = enforce_nesting_buffer(flags, topo, n_buffer=0)
        refined = {s for s in range(MAX_BLOCKS) if out[0, s] == REFINE}
        assert refined == {5}, f"n_buffer=0 dilated: {refined}"

    def test_no_orphan_coarsen(self):
        """A block with active children must not be coarsened."""
        # Root block 0 with a child at level 1.
        topo = AMRTopology()
        topo.add_block(0, 0, (0, 0))
        topo.add_block(1, 0, (0, 0), parent=(0, 0))

        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = COARSEN   # try to coarsen a block that has a child
        out = enforce_nesting_buffer(flags, topo, n_buffer=0)
        assert out[0, 0] == KEEP, "coarsen of a parent-with-children not blocked"

    def test_leaf_coarsen_allowed(self):
        """A childless block CAN be coarsened (not blocked by nesting)."""
        topo = AMRTopology()
        topo.add_block(0, 0, (0, 0))
        topo.add_block(1, 0, (0, 0), parent=(0, 0))   # leaf at level 1
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[1, 0] = COARSEN
        out = enforce_nesting_buffer(flags, topo, n_buffer=0)
        assert out[1, 0] == COARSEN, "leaf coarsen wrongly blocked"


# ── apply_flags: end-to-end refine/coarsen on a static state ──────────────────

class TestApplyFlagsRefine:

    def _setup_one_block_smooth_state(self):
        """Build an AMR state with a single coarse block carrying a smooth
        polynomial field (so prolongation gives a known answer)."""
        def p(X, Y):
            return X + Y + 0.1 * X * Y
        coords = np.arange(BS + 2*NG)
        Xc, Yc = np.meshgrid(coords, coords, indexing='ij')
        single = np.broadcast_to(p(Xc, Yc), (NF, BS + 2*NG, BS + 2*NG))

        blocks = np.zeros((LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG))
        blocks[0, 0] = single
        active = np.zeros((LEVELS, MAX_BLOCKS), dtype=bool)
        active[0, 0] = True
        state = AMRState(blocks=jnp.asarray(blocks), active=jnp.asarray(active))
        topo = AMRTopology()
        topo.add_block(0, 0, (0, 0))
        return state, topo, single

    def test_refine_creates_four_children(self):
        state, topo, parent_field = self._setup_one_block_smooth_state()
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE
        new_state, new_topo = apply_flags(state, topo, flags)

        # Level 1 should now have 4 active blocks.
        assert int(np.asarray(new_state.active[1]).sum()) == 4
        # Each child's bbox should be one of the 4 quadrants of the parent.
        children = new_topo.children[(0, 0)]
        assert len(children) == 4
        expected_corners = {(0, 0), (0, BS), (BS, 0), (BS, BS)}
        got_corners = {new_topo.bbox_ijk[c] for c in children}
        assert got_corners == expected_corners, \
            f"got {got_corners}, expected {expected_corners}"

        # Each child's interior must match prolongating the parent for that corner.
        parent_jax = state.blocks[0][0]
        for child_key in children:
            (cL, cs) = child_key
            child_bbox = new_topo.bbox_ijk[child_key]
            cx = child_bbox[0] // BS    # children's bbox is at integer multiples of BS
            cy = child_bbox[1] // BS
            expected_block = np.asarray(prolongate(parent_jax, (cx, cy)))
            got_block = np.asarray(new_state.blocks[cL][cs])
            assert np.allclose(got_block, expected_block, atol=1e-14), \
                f"child ({cx},{cy}): prolongation mismatch"


class TestApplyFlagsCoarsen:

    def test_coarsen_removes_child_and_restricts(self):
        """REFINE then COARSEN the same block in two consecutive apply_flags
        calls.  After coarsening, the parent's quadrant should equal the
        2×2 average of the child's interior."""
        # Build root state with a polynomial field.
        def p(X, Y):
            return X * Y - X + 2 * Y
        coords = np.arange(BS + 2*NG)
        Xc, Yc = np.meshgrid(coords, coords, indexing='ij')
        single = np.broadcast_to(p(Xc, Yc), (NF, BS + 2*NG, BS + 2*NG))

        blocks = np.zeros((LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG))
        blocks[0, 0] = single
        active = np.zeros((LEVELS, MAX_BLOCKS), dtype=bool)
        active[0, 0] = True
        state = AMRState(blocks=jnp.asarray(blocks), active=jnp.asarray(active))
        topo = AMRTopology()
        topo.add_block(0, 0, (0, 0))

        # Refine.
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        assert int(np.asarray(state.active[1]).sum()) == 4

        # Pick one child to coarsen.
        children = topo.children[(0, 0)]
        target_child = children[0]   # (level 1, some slot)
        (cL, cs) = target_child
        child_bbox = topo.bbox_ijk[target_child]
        cx = child_bbox[0] // BS
        cy = child_bbox[1] // BS
        child_interior_before = np.asarray(state.blocks[cL][cs, :, NG:NG+BS, NG:NG+BS])
        parent_before = np.asarray(state.blocks[0][0])

        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[cL, cs] = COARSEN
        state, topo = apply_flags(state, topo, flags)

        # Slot deactivated.
        assert not bool(state.active[cL][cs])
        assert int(np.asarray(state.active[1]).sum()) == 3
        # Parent quadrant should equal 2×2 average of the child interior.
        half_bs = BS // 2
        x0 = NG + cx * half_bs; y0 = NG + cy * half_bs
        got_quad = np.asarray(state.blocks[0][0, :, x0:x0+half_bs, y0:y0+half_bs])
        expected_quad = child_interior_before.reshape(
            NF, half_bs, REFINE_RATIO, half_bs, REFINE_RATIO
        ).mean(axis=(2, 4))
        assert np.allclose(got_quad, expected_quad, atol=1e-14), \
            f"coarsened quadrant mismatch (max |Δ|={np.max(np.abs(got_quad - expected_quad)):.2e})"
        # Cells OUTSIDE that quadrant should be untouched.
        mask = np.ones(parent_before.shape, dtype=bool)
        mask[:, x0:x0+half_bs, y0:y0+half_bs] = False
        diff = (np.asarray(state.blocks[0][0]) - parent_before)[mask]
        assert np.max(np.abs(diff)) < 1e-14, "coarsen leaked into other cells"


# ── regrid() smoke test ───────────────────────────────────────────────────────

class TestRegridSmoke:
    """End-to-end: feed a state with a localised feature, call regrid, confirm
    the right block gets refined after enough hysteresis cycles."""

    def test_localised_feature_gets_refined(self):
        """Put a steep gradient inside block (0, 0); blocks (0, 1) and beyond
        carry a flat field.  After K regrid cycles, only block 0 should be
        flagged and refined."""
        nbx, nby = 2, 2
        # Block 0 has a sin gradient; others are flat.
        coords = np.arange(BS + 2*NG)
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        steep = np.sin(np.pi / 8.0 * X.astype(np.float64))   # k = π/8, large gradient
        flat  = np.zeros_like(steep)

        blocks = np.zeros((LEVELS, MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG))
        for slot in range(nbx * nby):
            blocks[0, slot, EZ_IDX] = steep if slot == 0 else flat
        active = np.zeros((LEVELS, MAX_BLOCKS), dtype=bool)
        active[0, :nbx*nby] = True

        state = AMRState(blocks=jnp.asarray(blocks), active=jnp.asarray(active))
        topo = AMRTopology()
        for slot in range(nbx * nby):
            bi = slot // nby; bj = slot % nby
            topo.add_block(0, slot, (bi * BS, bj * BS))

        # k = π/8, dx_phys = 0.5 → max |∇f| ≈ k/dx = π/4 ≈ 0.785
        dx_per_level = [0.5, 0.25, 0.125, 0.0625]
        K = 2
        # After K cycles, slot 0 should be REFINE-flagged + refined.
        # n_buffer=0: isolate the indicator/flag logic (no buffer dilation —
        # on a 2×2 grid every block neighbours slot 0, so a buffer would
        # refine all of them; buffer dilation is covered in TestNestingBuffer).
        for cycle in range(K):
            state, topo = regrid(
                state, topo, dx_per_level,
                field_idx=EZ_IDX,
                refine_threshold=0.5, coarsen_threshold=0.01,
                hysteresis_K=K, n_buffer=0,
            )

        # Slot 0 should have 4 children; slots 1..3 should not.
        assert (0, 0) in topo.children, "block (0,0) should be refined"
        assert len(topo.children[(0, 0)]) == 4
        for slot in range(1, nbx * nby):
            assert (0, slot) not in topo.children, \
                f"block (0,{slot}) should NOT have been refined"

        # Level 1 active count should be exactly 4.
        assert int(np.asarray(state.active[1]).sum()) == 4


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
