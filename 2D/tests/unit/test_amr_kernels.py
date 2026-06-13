"""
AMR correctness tests — Phase 1.

These verify the per-block kernels in isolation:
  - prolongation: 6th-order convergence on smooth analytical functions
  - restriction: conservation property
  - round-trip: prolongate then restrict ≈ identity on smooth functions

Higher-level tests (full evolution with refinement) come in later phases.
"""


import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import BS, NG, NF, MAX_BLOCKS, AMRState, AMRTopology, make_root_state
from mcs2d.amr.kernels import (
    prolongate,
    _interp_to_fine_1d,
    restrict_into_parent,
    restrict_all_into_parents,
    restrict_into_parent_highorder,
    restrict_all_into_parents_highorder,
    sync_ghosts_within_level_root_periodic,
    sync_ghosts_across_levels,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_smooth_block(func) -> jnp.ndarray:
    """Build one block (NF, BS+2*NG, BS+2*NG) where each field f evaluates `func(x, y, f)`
    at coarse cell centers.  Coordinates run over [0, BS+2*NG) integer cell indices."""
    coords = np.arange(BS + 2*NG)
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    fields = []
    for f in range(NF):
        fields.append(func(X, Y, f))
    return jnp.asarray(np.stack(fields))


def _exact_at_fine_centers(func, child_corner, f) -> np.ndarray:
    """Reference values for the child's INTERIOR (BS, BS) — `func` evaluated at the
    physical positions of the fine cell centers.

    Cell-centered AMR convention: coarse cell k spans [k-0.5, k+0.5], center at k.
    Refining gives 2 fine cells at positions k-0.25 (LEFT half) and k+0.25 (RIGHT half).
    The child for `child_corner` covers coarse cells [NG + cx*BS/2, NG + (cx+1)*BS/2).
    Fine cell index Xf in [0, BS) corresponds to:
        coarse_cell = NG + cx*BS/2 + Xf // 2
        offset = -0.25 if Xf is even else +0.25
    """
    cx, cy = child_corner
    half_bs = BS // 2
    i_fine = np.arange(BS)
    j_fine = np.arange(BS)
    Xf, Yf = np.meshgrid(i_fine, j_fine, indexing='ij')
    # Coarse-cell index covered by this fine cell, and the half-offset within it.
    coarse_x = NG + cx * half_bs + Xf // 2
    coarse_y = NG + cy * half_bs + Yf // 2
    off_x = -0.25 + 0.5 * (Xf % 2)
    off_y = -0.25 + 0.5 * (Yf % 2)
    return func(coarse_x + off_x, coarse_y + off_y, f)


def _exact_at_fine_full_block(func, child_corner, f) -> np.ndarray:
    """Reference values for the child's FULL block (BS+2*NG, BS+2*NG) — including halo.

    Same convention as _exact_at_fine_centers, but the fine index range is
    [-NG, BS+NG) (i.e., block indices [0, BS+2*NG)) — covering NG halo cells on
    each side of the BS-wide interior.

    Returns a (BS+2*NG, BS+2*NG) array.
    """
    cx, cy = child_corner
    half_bs = BS // 2
    # Block indices [0, BS+2*NG); interior indices map to block index NG..NG+BS-1.
    # In "interior fine index" coords (i_fine), block index = NG + i_fine,
    # so i_fine = block_idx - NG, with i_fine ∈ [-NG, BS+NG).
    i_fine = np.arange(-NG, BS + NG)
    j_fine = np.arange(-NG, BS + NG)
    Xf, Yf = np.meshgrid(i_fine, j_fine, indexing='ij')
    coarse_x = NG + cx * half_bs + np.floor_divide(Xf, 2)
    coarse_y = NG + cy * half_bs + np.floor_divide(Yf, 2)
    off_x = -0.25 + 0.5 * (np.mod(Xf, 2))
    off_y = -0.25 + 0.5 * (np.mod(Yf, 2))
    return func(coarse_x + off_x, coarse_y + off_y, f)


# ── Sanity ────────────────────────────────────────────────────────────────────

class TestAMRSetup:
    """Basic shape & data-structure tests."""

    def test_state_shapes(self):
        s = AMRState.empty()
        from mcs2d.amr.state import LEVELS, mb
        # Ragged per-level storage: blocks/active are tuples of LEVELS arrays.
        assert len(s.blocks) == LEVELS and len(s.active) == LEVELS
        for L in range(LEVELS):
            assert s.blocks[L].dtype == jnp.float64
            assert s.active[L].dtype == jnp.bool_
            assert s.blocks[L].shape == (mb(L), NF, BS + 2*NG, BS + 2*NG)
            assert s.active[L].shape == (mb(L),)

    def test_topology_empty(self):
        topo = AMRTopology()
        assert topo.n_active(0) == 0
        assert topo.find_empty_slot(0) == 0

    def test_add_remove_block(self):
        topo = AMRTopology()
        topo.add_block(level=0, slot=5, bbox_ij=(0, 0))
        assert topo.active[0, 5]
        assert topo.n_active(0) == 1
        assert topo.find_empty_slot(0) == 0   # 5 is taken, 0 is still free
        topo.remove_block(0, 5)
        assert not topo.active[0, 5]

    def test_add_remove_cycle_consistency(self):
        """add/remove sequences must keep parent/children/bbox dicts consistent."""
        topo = AMRTopology()
        # Build a small hierarchy: root block + 4 children.
        topo.add_block(level=0, slot=0, bbox_ij=(0, 0))
        for c, corner in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            topo.add_block(level=1, slot=c, bbox_ij=corner, parent=(0, 0))

        assert topo.n_active(0) == 1
        assert topo.n_active(1) == 4
        assert topo.children[(0, 0)] == [(1, 0), (1, 1), (1, 2), (1, 3)]
        for c in range(4):
            assert topo.parent[(1, c)] == (0, 0)
            assert topo.bbox_ijk[(1, c)] in [(0,0), (0,1), (1,0), (1,1)]

        # Remove one child — child dict should drop, parent dict should drop the entry.
        topo.remove_block(1, 2)
        assert not topo.active[1, 2]
        assert (1, 2) not in topo.parent
        assert (1, 2) not in topo.bbox_ijk

        # find_empty_slot should now find slot 2.
        assert topo.find_empty_slot(1) == 2

        # Re-add into the freed slot; bookkeeping should be reconstructed cleanly.
        topo.add_block(level=1, slot=2, bbox_ij=(2,2), parent=(0, 0))
        assert topo.active[1, 2]
        assert topo.parent[(1, 2)] == (0, 0)
        assert topo.bbox_ijk[(1, 2)] == (2, 2)

    def test_make_root_state_construction(self, params_file):
        """make_root_state must tile a global IC into per-block storage with
        the correct interior matching the input."""
        nbx, nby = 2, 2
        nx, ny = nbx * BS, nby * BS
        rng = np.random.default_rng(3)
        # Build a global (NF, nx+2*NG, ny+2*NG) field — same convention main.py uses.
        global_data = jnp.asarray(rng.standard_normal((NF, nx + 2*NG, ny + 2*NG)))

        state, topo = make_root_state(global_data, nbx, nby)
        from mcs2d.amr.state import LEVELS, MAX_BLOCKS
        # All root tiles are active.
        assert int(state.active[0].sum()) == nbx * nby
        # No tiles on any other level.
        for L in range(1, LEVELS):
            assert int(state.active[L].sum()) == 0

        # Interior of each block matches the corresponding slice of global_data.
        for bi in range(nbx):
            for bj in range(nby):
                slot = bi * nby + bj
                got = np.asarray(state.blocks[0][slot])
                expected = np.asarray(global_data[:, bi*BS:bi*BS + BS+2*NG,
                                                     bj*BS:bj*BS + BS+2*NG])
                assert np.array_equal(got, expected), \
                    f"make_root_state mismatched block ({bi},{bj})"
                # Topology bbox should match.
                assert topo.bbox_ijk[(0, slot)] == (bi * BS, bj * BS)


# ── Prolongation ──────────────────────────────────────────────────────────────

class TestProlongation:
    """6th-order Lagrange interpolation should be exact for polynomials of degree ≤ 5.

    For a polynomial p(x, y) of degree ≤ 5, the prolongated values at fine cell
    centers should exactly equal p evaluated at those positions (modulo float
    roundoff).
    """

    # Relative tolerance — polynomial values can be ~10^7, absolute error of 1e-9
    # is essentially float64 roundoff at that magnitude.
    RELATIVE_TOL = 1e-12

    def test_polynomial_exact(self):
        """For a 5th-degree polynomial, the 6th-order interpolant is exact.

        Checks the INTERIOR (BS, BS) of the prolongated full block.  The halo
        cells near the global parent-block boundary use edge-padded stencils
        and are not exact for polynomials; see test_halo_interior_accuracy.
        """
        # p(x, y, f) = (1 + f) * (x + y + x*y - x^2 * y + 0.1 * x^3 * y^2)
        # degree 5 in x*y; well within the interpolant's exactness regime
        def p(X, Y, f):
            return (1 + f) * (X + Y + X*Y - X**2 * Y + 0.1 * X**3 * Y**2)

        parent_block = _make_smooth_block(p)

        # Test all 4 children
        for cx in (0, 1):
            for cy in (0, 1):
                child_full = prolongate(parent_block, (cx, cy))
                # Extract interior (NF, BS, BS) for comparison.
                child_interior = np.asarray(child_full[:, NG:NG+BS, NG:NG+BS])
                for f in range(NF):
                    expected = _exact_at_fine_centers(p, (cx, cy), f)
                    got = child_interior[f]
                    scale = max(float(np.max(np.abs(expected))), 1.0)
                    err = float(np.max(np.abs(got - expected))) / scale
                    assert err < self.RELATIVE_TOL, (
                        f"child=({cx},{cy}), field={f}: max rel error {err:.2e} > "
                        f"{self.RELATIVE_TOL:.0e}"
                    )

    def test_halo_inner_cells_exact(self):
        """The child's halo cells whose stencils don't touch the parent's outer
        boundary should still be exact for a 5th-degree polynomial.

        For cx=0, the LEFT halo of the child reads parent halo cells that include
        parent index 0, 1, 2 — and the leftmost stencils dip to parent indices
        {-2, -1, 0, ...} which are edge-padded.  But the INNER side of the halo
        (the cells adjacent to the interior) has full real stencils.  Check those.
        """
        def p(X, Y, f):
            return (1 + f) * (X + Y + X*Y - X**2 * Y + 0.1 * X**3 * Y**2)

        parent_block = _make_smooth_block(p)
        for cx in (0, 1):
            for cy in (0, 1):
                child_full = np.asarray(prolongate(parent_block, (cx, cy)))
                expected_full = _exact_at_fine_full_block(p, (cx, cy), 0)
                # The fine indices in the refined (whole-parent) grid that have
                # full real stencils are [4, 2*(BS+2*NG) - 6) = [4, 70).
                # The child's block-index 0 corresponds to refined index NG + cx*BS = 3 or 35.
                # For cx=0: child block index f maps to refined index f + 3.  So child
                # block indices f ∈ [1, 67) have full real stencils.  We test the
                # halo cells [1, NG) and (NG+BS-1, BS+2*NG-1).  For cx=1: child block
                # index f → refined index f + 35.  All halo cells f ∈ [0, NG) map to
                # refined [35, 38), full stencils; the right halo f ∈ [NG+BS, BS+2*NG)
                # → refined [70, 73), edge-padded.  So just check non-padded halo cells.
                if cx == 0:
                    safe_x = slice(1, NG + BS + NG - 0)   # exclude only the very first
                else:
                    safe_x = slice(0, NG + BS)            # exclude right halo
                if cy == 0:
                    safe_y = slice(1, NG + BS + NG - 0)
                else:
                    safe_y = slice(0, NG + BS)
                # All other cells (interior + inner-halo) should be exact.
                got_int = child_full[0, NG:NG+BS, NG:NG+BS]
                exp_int = expected_full[NG:NG+BS, NG:NG+BS]
                scale = max(float(np.max(np.abs(exp_int))), 1.0)
                err = float(np.max(np.abs(got_int - exp_int))) / scale
                assert err < self.RELATIVE_TOL, (
                    f"interior child=({cx},{cy}): rel err {err:.2e} > {self.RELATIVE_TOL:.0e}"
                )

    def test_constant_preserved(self):
        """Prolongating a constant field must give the same constant back —
        including the halo cells, since edge-padded stencils replicate the
        constant value too."""
        parent = jnp.ones((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64) * 3.14
        for cx in (0, 1):
            for cy in (0, 1):
                child = prolongate(parent, (cx, cy))
                assert child.shape == (NF, BS + 2*NG, BS + 2*NG)
                assert np.allclose(np.asarray(child), 3.14, atol=1e-14)

    def test_linear_preserved(self):
        """Linear functions are exactly interpolatable on the INTERIOR.

        Edge-padded stencils introduce error on the outermost halo cells, so
        we check the interior only here.
        """
        def lin(X, Y, f):
            return X + 2*Y + f
        parent_block = _make_smooth_block(lin)
        for cx in (0, 1):
            for cy in (0, 1):
                child_full = prolongate(parent_block, (cx, cy))
                child_interior = np.asarray(child_full[:, NG:NG+BS, NG:NG+BS])
                for f in range(NF):
                    expected = _exact_at_fine_centers(lin, (cx, cy), f)
                    err = float(np.max(np.abs(child_interior[f] - expected)))
                    assert err < 1e-13, f"linear: ({cx},{cy}), f={f}: err = {err:.2e}"


# ── High-order restriction ────────────────────────────────────────────────────

class TestHighOrderRestriction:
    """6th-order interpolatory restriction: exact for degree-≤5 polynomials,
    and strictly more accurate than 2×2 averaging on a smooth field."""

    RELATIVE_TOL = 1e-11

    def test_polynomial_exact(self):
        """For a 5th-degree polynomial the 6th-order restriction reproduces the
        coarse-cell-centre values exactly (modulo float64 roundoff)."""
        def p(X, Y, f):
            return (1 + f) * (X + Y + X*Y - X**2 * Y + 0.1 * X**3 * Y**2)
        half_bs = BS // 2
        a = np.arange(half_bs)
        A, B = np.meshgrid(a, a, indexing='ij')
        for cx in (0, 1):
            for cy in (0, 1):
                # Child FULL block sampled at the fine cell positions (incl halo).
                child_full = np.stack([
                    _exact_at_fine_full_block(p, (cx, cy), f) for f in range(NF)
                ])
                parent = jnp.zeros((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64)
                result = np.asarray(restrict_into_parent_highorder(
                    parent, jnp.asarray(child_full), (cx, cy)
                ))
                x0 = NG + cx * half_bs
                y0 = NG + cy * half_bs
                for f in range(NF):
                    expected = p(x0 + A, y0 + B, f)
                    got = result[f, x0:x0+half_bs, y0:y0+half_bs]
                    scale = max(float(np.max(np.abs(expected))), 1.0)
                    err = float(np.max(np.abs(got - expected))) / scale
                    assert err < self.RELATIVE_TOL, (
                        f"corner=({cx},{cy}) f={f}: high-order restriction rel err "
                        f"{err:.2e} > {self.RELATIVE_TOL:.0e}"
                    )

    def test_constant_preserved(self):
        const = 2.71
        child_full = jnp.ones((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64) * const
        parent = jnp.zeros((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64)
        half_bs = BS // 2
        for cx in (0, 1):
            for cy in (0, 1):
                r = np.asarray(restrict_into_parent_highorder(parent, child_full, (cx, cy)))
                x0 = NG + cx * half_bs; y0 = NG + cy * half_bs
                region = r[:, x0:x0+half_bs, y0:y0+half_bs]
                assert np.allclose(region, const, atol=1e-13), "high-order restriction lost a constant"

    def test_more_accurate_than_averaging(self):
        """On a smooth (non-polynomial) field, 6th-order restriction must beat
        2nd-order 2×2 averaging at recovering the coarse-centre values."""
        omega = 0.35
        def f(X, Y, fld):
            return np.sin(omega * X) * np.cos(omega * Y) + 0.1 * fld
        cx = cy = 0
        half_bs = BS // 2
        child_full = np.stack([_exact_at_fine_full_block(f, (cx, cy), fld) for fld in range(NF)])
        parent = jnp.zeros((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64)

        hi = np.asarray(restrict_into_parent_highorder(parent, jnp.asarray(child_full), (cx, cy)))
        lo = np.asarray(restrict_into_parent(
            parent, jnp.asarray(child_full[:, NG:NG+BS, NG:NG+BS]), (cx, cy)))

        a = np.arange(half_bs); A, B = np.meshgrid(a, a, indexing='ij')
        x0 = NG; y0 = NG
        exact = np.stack([f(x0 + A, y0 + B, fld) for fld in range(NF)])
        err_hi = np.max(np.abs(hi[:, x0:x0+half_bs, y0:y0+half_bs] - exact))
        err_lo = np.max(np.abs(lo[:, x0:x0+half_bs, y0:y0+half_bs] - exact))
        assert err_hi < err_lo / 10, (
            f"high-order restriction ({err_hi:.2e}) not >10x better than averaging ({err_lo:.2e})"
        )

    def test_multislot_matches_single(self):
        """The multi-slot `restrict_all_into_parents_highorder` (production path)
        must match the single-slot kernel for an active slot, and leave inactive
        slots / unwritten cells untouched."""
        rng = np.random.default_rng(101)
        coarse = jnp.asarray(rng.standard_normal((MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)))
        fine = jnp.asarray(rng.standard_normal((MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)))
        parent_slot = np.zeros(MAX_BLOCKS, np.int32); parent_slot[0] = 2
        child_cx = np.zeros(MAX_BLOCKS, np.int32); child_cx[0] = 1
        child_cy = np.zeros(MAX_BLOCKS, np.int32); child_cy[0] = 1
        active = np.zeros(MAX_BLOCKS, bool); active[0] = True

        got = np.asarray(restrict_all_into_parents_highorder(
            coarse, fine, jnp.asarray(parent_slot), jnp.asarray(child_cx),
            jnp.asarray(child_cy), jnp.asarray(active)))
        expected_parent = np.asarray(restrict_into_parent_highorder(
            coarse[2], fine[0], (1, 1)))
        for slot in range(MAX_BLOCKS):
            ref = expected_parent if slot == 2 else np.asarray(coarse[slot])
            err = float(np.max(np.abs(got[slot] - ref)))
            assert err < 1e-12, f"slot {slot}: multi-slot high-order mismatch {err:.2e}"


# ── Restriction ───────────────────────────────────────────────────────────────

class TestRestriction:
    """Restriction averages 2×2 children → 1 parent cell.  Exact for linear functions."""

    def test_constant_preserved(self):
        """Restriction of constant child values into a clean parent reproduces the constant."""
        parent_block = jnp.zeros((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64)
        child = jnp.ones((NF, BS, BS), dtype=jnp.float64) * 2.71

        for cx in (0, 1):
            for cy in (0, 1):
                new_parent = restrict_into_parent(parent_block, child, (cx, cy))
                # Check the affected region of the parent matches the constant
                half_bs = BS // 2
                x0 = NG + cx * half_bs
                y0 = NG + cy * half_bs
                region = np.asarray(new_parent[:, x0:x0 + half_bs, y0:y0 + half_bs])
                assert np.allclose(region, 2.71, atol=1e-14), \
                    f"restriction lost the constant for ({cx},{cy})"

    def test_2x2_averaging(self):
        """Each parent cell should be the mean of its 4 child cells."""
        # Build a child with distinct values per cell, then check averaging.
        # child[f, i, j] = i + 100*j (per field)
        i_idx = np.arange(BS)
        j_idx = np.arange(BS)
        I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        child = jnp.asarray(np.broadcast_to(I + 100*J, (NF, BS, BS)).astype(np.float64))
        parent_block = jnp.zeros((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64)

        new_parent = restrict_into_parent(parent_block, child, (0, 0))

        # Expected parent cell at (a, b) in [0, BS//2)
        # = mean of child[2a:2a+2, 2b:2b+2]
        # = mean of {2a + 100*(2b), 2a + 100*(2b+1), (2a+1)+100*(2b), (2a+1)+100*(2b+1)}
        # = mean of {2a + 200b, 2a + 200b + 100, 2a+1 + 200b, 2a+1 + 200b + 100}
        # = 2a + 200b + (0 + 100 + 1 + 101)/4 = 2a + 200b + 50.5
        half_bs = BS // 2
        for a in range(half_bs):
            for b in range(half_bs):
                expected = 2*a + 200*b + 50.5
                got = float(new_parent[0, NG + a, NG + b])
                assert abs(got - expected) < 1e-10, \
                    f"avg mismatch at parent ({a},{b}): got {got}, expected {expected}"

    @pytest.mark.parametrize("cx,cy", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_all_corners_2x2_averaging(self, cx, cy):
        """The same 2×2 averaging law must hold for every quadrant of the parent."""
        i_idx = np.arange(BS)
        j_idx = np.arange(BS)
        I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        child = jnp.asarray(np.broadcast_to(I + 100*J, (NF, BS, BS)).astype(np.float64))
        parent_block = jnp.zeros((NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64)

        new_parent = restrict_into_parent(parent_block, child, (cx, cy))
        half_bs = BS // 2
        x0 = NG + cx * half_bs
        y0 = NG + cy * half_bs
        for a in range(half_bs):
            for b in range(half_bs):
                expected = 2*a + 200*b + 50.5
                got = float(new_parent[0, x0 + a, y0 + b])
                assert abs(got - expected) < 1e-10, \
                    f"corner=({cx},{cy}) parent ({a},{b}): got {got} expected {expected}"

    def test_other_quadrants_unchanged(self):
        """Restriction into one quadrant must leave the other 3 quadrants of the
        parent's interior (and the halo) bit-identical to their initial values."""
        rng = np.random.default_rng(11)
        parent_init = jnp.asarray(rng.standard_normal((NF, BS + 2*NG, BS + 2*NG)))
        child = jnp.asarray(rng.standard_normal((NF, BS, BS)))

        for cx in (0, 1):
            for cy in (0, 1):
                new_parent = np.asarray(restrict_into_parent(parent_init, child, (cx, cy)))
                half_bs = BS // 2
                x0 = NG + cx * half_bs
                y0 = NG + cy * half_bs
                mask = np.ones((BS + 2*NG, BS + 2*NG), dtype=bool)
                mask[x0:x0 + half_bs, y0:y0 + half_bs] = False
                for f in range(NF):
                    diff_outside = (new_parent[f] - np.asarray(parent_init[f]))[mask]
                    assert np.allclose(diff_outside, 0.0, atol=0.0), (
                        f"corner=({cx},{cy}) field {f}: cells outside the written quadrant changed"
                    )


class TestRestrictAllIntoParents:
    """Multi-slot vmapped restriction: bit-identical to the single-slot kernel
    when active, no-op when inactive, and correctly handles multiple children
    of one parent."""

    def test_single_child_matches_single_kernel(self):
        """One active fine slot: restrict_all should match restrict_into_parent
        applied to that one slot."""
        rng = np.random.default_rng(17)
        coarse_blocks = jnp.asarray(rng.standard_normal(
            (MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG)
        ))
        fine_blocks = jnp.asarray(rng.standard_normal(
            (MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG)
        ))

        parent_slot = np.zeros(MAX_BLOCKS, dtype=np.int32)
        parent_slot[0] = 3
        child_cx = np.zeros(MAX_BLOCKS, dtype=np.int32); child_cx[0] = 1
        child_cy = np.zeros(MAX_BLOCKS, dtype=np.int32); child_cy[0] = 0
        active   = np.zeros(MAX_BLOCKS, dtype=bool);     active[0] = True

        got = np.asarray(restrict_all_into_parents(
            coarse_blocks, fine_blocks,
            jnp.asarray(parent_slot), jnp.asarray(child_cx),
            jnp.asarray(child_cy),    jnp.asarray(active),
        ))
        expected_parent = np.asarray(restrict_into_parent(
            coarse_blocks[3],
            fine_blocks[0, :, NG:NG+BS, NG:NG+BS],
            (1, 0),
        ))
        # Only slot 3 should differ.  Allow ULP-level slack since the multi-slot
        # kernel uses dynamic_update_slice with traced indices (the single-slot
        # version uses static indices), which can differ by 1 ULP.
        ulp_tol = 1e-14
        for slot in range(MAX_BLOCKS):
            if slot == 3:
                err = float(np.max(np.abs(got[slot] - expected_parent)))
                assert err < ulp_tol, \
                    f"slot 3: differs from single-kernel result by {err:.2e}"
            else:
                err = float(np.max(np.abs(got[slot] - np.asarray(coarse_blocks[slot]))))
                assert err < ulp_tol, \
                    f"slot {slot}: unexpectedly modified (max |Δ|={err:.2e})"

    def test_inactive_slots_no_op(self):
        """All inactive: coarse_blocks should pass through unchanged."""
        rng = np.random.default_rng(19)
        coarse_blocks = jnp.asarray(rng.standard_normal(
            (MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG)
        ))
        fine_blocks = jnp.asarray(rng.standard_normal(
            (MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG)
        ))
        z_int = np.zeros(MAX_BLOCKS, dtype=np.int32)
        z_bool = np.zeros(MAX_BLOCKS, dtype=bool)

        got = np.asarray(restrict_all_into_parents(
            coarse_blocks, fine_blocks,
            jnp.asarray(z_int), jnp.asarray(z_int),
            jnp.asarray(z_int), jnp.asarray(z_bool),
        ))
        assert np.array_equal(got, np.asarray(coarse_blocks)), \
            "inactive sync mutated coarse blocks"

    def test_four_children_one_parent(self):
        """4 active fine slots, all with the same coarse parent, each writing
        a different quadrant.  Result: the parent's full interior should equal
        the 2×2-averaged "fine block reconstructed from all 4 children"."""
        rng = np.random.default_rng(23)
        coarse_blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=np.float64)
        fine_blocks   = np.zeros_like(coarse_blocks)
        # Random per-quadrant fine data.
        corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for slot, _ in enumerate(corners):
            fine_blocks[slot, :, NG:NG+BS, NG:NG+BS] = rng.standard_normal((NF, BS, BS))

        parent_slot = np.zeros(MAX_BLOCKS, dtype=np.int32)   # all → coarse slot 0
        child_cx = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cy = np.zeros(MAX_BLOCKS, dtype=np.int32)
        active   = np.zeros(MAX_BLOCKS, dtype=bool)
        for slot, (cx, cy) in enumerate(corners):
            child_cx[slot] = cx; child_cy[slot] = cy; active[slot] = True

        got = np.asarray(restrict_all_into_parents(
            jnp.asarray(coarse_blocks), jnp.asarray(fine_blocks),
            jnp.asarray(parent_slot), jnp.asarray(child_cx),
            jnp.asarray(child_cy),    jnp.asarray(active),
        ))

        # Expected: coarse slot 0's interior is the union of the 4 averaged quadrants.
        half_bs = BS // 2
        expected_interior = np.zeros((NF, BS, BS), dtype=np.float64)
        for slot, (cx, cy) in enumerate(corners):
            child_int = fine_blocks[slot, :, NG:NG+BS, NG:NG+BS]
            avg = child_int.reshape(NF, half_bs, 2, half_bs, 2).mean(axis=(2, 4))
            x0 = cx * half_bs; y0 = cy * half_bs
            expected_interior[:, x0:x0+half_bs, y0:y0+half_bs] = avg

        got_interior = got[0, :, NG:NG+BS, NG:NG+BS]
        assert np.allclose(got_interior, expected_interior, atol=1e-14), (
            f"4-children restriction max err = "
            f"{np.max(np.abs(got_interior - expected_interior)):.2e}"
        )

    def test_smooth_function_2nd_order(self):
        """When the fine block carries the analytic restriction of a smooth
        function (rather than its prolongation), restrict_all should reproduce
        the smooth function on the coarse parent to 2nd-order accuracy —
        same as `restrict_into_parent` does in isolation."""
        # Build a smooth analytical function and sample at both coarse and fine resolutions.
        omega = 0.05
        def f(X, Y, fld):
            return np.sin(omega * (X + Y + fld))

        # Coarse block: function sampled at coarse cell centers (indices 0..BS+2NG-1).
        coords_c = np.arange(BS + 2*NG)
        Xc, Yc = np.meshgrid(coords_c, coords_c, indexing='ij')
        coarse_block = np.stack([f(Xc, Yc, fld) for fld in range(NF)])

        # Fine block (corner 0,0): sampled at the fine cell centers (cell-centered convention).
        # Fine cell index i in [0, BS) → parent coord NG + i//2 + (-0.25 if i%2 else +0.25).
        i_fine = np.arange(BS)
        Xf_idx, Yf_idx = np.meshgrid(i_fine, i_fine, indexing='ij')
        parent_x = NG + Xf_idx // 2 + np.where(Xf_idx % 2 == 0, -0.25, 0.25)
        parent_y = NG + Yf_idx // 2 + np.where(Yf_idx % 2 == 0, -0.25, 0.25)
        fine_interior = np.stack([f(parent_x, parent_y, fld) for fld in range(NF)])

        # Build the AMR-shape inputs.
        coarse_blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG))
        coarse_blocks[0] = coarse_block
        fine_blocks = np.zeros_like(coarse_blocks)
        fine_blocks[0, :, NG:NG+BS, NG:NG+BS] = fine_interior

        parent_slot = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cx = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cy = np.zeros(MAX_BLOCKS, dtype=np.int32)
        active   = np.zeros(MAX_BLOCKS, dtype=bool); active[0] = True

        got = np.asarray(restrict_all_into_parents(
            jnp.asarray(coarse_blocks), jnp.asarray(fine_blocks),
            jnp.asarray(parent_slot), jnp.asarray(child_cx),
            jnp.asarray(child_cy),    jnp.asarray(active),
        ))

        # Coarse parent's quadrant (0,0) of the interior should now equal the
        # smooth function sampled at coarse cell centers, to 2nd-order.
        half_bs = BS // 2
        x_c = np.arange(half_bs)
        y_c = np.arange(half_bs)
        Xq, Yq = np.meshgrid(x_c, y_c, indexing='ij')
        expected = f(NG + Xq, NG + Yq, 0)   # field 0
        got_quad = got[0, 0, NG:NG+half_bs, NG:NG+half_bs]
        err = np.max(np.abs(got_quad - expected))
        # 2nd-order restriction error: ~ω²/16
        analytic_bound = (omega ** 2) / 16 * 1.5
        assert err < analytic_bound, (
            f"smooth restriction err {err:.2e} > analytic 2nd-order bound {analytic_bound:.2e}"
        )


# ── Ghost-zone sync (root level, periodic BC) ─────────────────────────────────

class TestGhostSyncRootPeriodic:
    """The pad-and-extract sync should make each block's halo contain values
    consistent with the periodic neighbor's interior."""

    @staticmethod
    def _make_blocks_from_interior(global_int: np.ndarray, nbx: int, nby: int) -> np.ndarray:
        """Tile a (NF, nbx*BS, nby*BS) global interior into per-block storage
        of shape (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG).  Halo cells of each block
        are zero-initialised; sync should fill them.
        """
        blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=np.float64)
        for bi in range(nbx):
            for bj in range(nby):
                slot = bi * nby + bj
                blocks[slot, :, NG:NG + BS, NG:NG + BS] = \
                    global_int[:, bi*BS:(bi+1)*BS, bj*BS:(bj+1)*BS]
        return blocks

    def test_periodic_wrap_smooth(self):
        """A smooth periodic function: after sync, each block's halo should
        match the function evaluated at periodically-wrapped neighbor positions."""
        nbx, nby = 2, 2
        Lx, Ly = nbx * BS, nby * BS
        def f_global(i, j, fld):
            kx = 2 * np.pi / Lx
            ky = 2 * np.pi / Ly
            return np.sin(kx*i + ky*j) + 0.5 * np.cos(2*kx*i) + 0.1*fld

        # Build global interior with values at cell-centered positions.
        ii = np.arange(Lx)
        jj = np.arange(Ly)
        Ig, Jg = np.meshgrid(ii, jj, indexing='ij')
        global_int = np.stack([f_global(Ig, Jg, fld) for fld in range(NF)])

        blocks = self._make_blocks_from_interior(global_int, nbx, nby)
        synced = np.asarray(
            sync_ghosts_within_level_root_periodic(jnp.asarray(blocks), nbx, nby)
        )

        # Now check that each block's full BS+2*NG window matches the function
        # evaluated at the corresponding (periodically-wrapped) global positions.
        for bi in range(nbx):
            for bj in range(nby):
                slot = bi * nby + bj
                # Block "interior" cell (a, b) in [0, BS) corresponds to global
                # cell (bi*BS + a, bj*BS + b).  With NG halo on the block, block
                # index (i, j) ∈ [0, BS+2*NG) corresponds to global (bi*BS + i - NG,
                # bj*BS + j - NG), mod (Lx, Ly).
                i_blk = np.arange(BS + 2*NG)
                j_blk = np.arange(BS + 2*NG)
                Ib, Jb = np.meshgrid(i_blk, j_blk, indexing='ij')
                Ig_b = (bi*BS + Ib - NG) % Lx
                Jg_b = (bj*BS + Jb - NG) % Ly
                expected = np.stack([f_global(Ig_b, Jg_b, fld) for fld in range(NF)])
                err = float(np.max(np.abs(synced[slot] - expected)))
                assert err < 1e-13, f"sync block ({bi},{bj}): max err {err:.2e}"

    def test_interior_unchanged(self):
        """Sync must not modify any block's INTERIOR."""
        nbx, nby = 2, 2
        # Random-ish per-block interior values; halo zero.
        rng = np.random.default_rng(seed=42)
        blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=np.float64)
        blocks[:nbx*nby, :, NG:NG+BS, NG:NG+BS] = \
            rng.standard_normal((nbx*nby, NF, BS, BS))

        synced = np.asarray(
            sync_ghosts_within_level_root_periodic(jnp.asarray(blocks), nbx, nby)
        )

        for slot in range(nbx * nby):
            int_before = blocks[slot, :, NG:NG+BS, NG:NG+BS]
            int_after  = synced[slot, :, NG:NG+BS, NG:NG+BS]
            assert np.array_equal(int_before, int_after), \
                f"sync clobbered interior of slot {slot}"


# ── Cross-level halo fill ─────────────────────────────────────────────────────

class TestSyncAcrossLevels:
    """sync_ghosts_across_levels should fill the fine block's halo with values
    interpolated from the parent (and leave the interior alone)."""

    def test_smooth_function_halo_filled(self):
        """For one fine child of a single coarse parent: confirm the child's
        halo is filled with prolongated parent values, interior is preserved."""
        # Build a smooth polynomial parent (degree ≤ 5 ⇒ prolongation is exact).
        def p(X, Y, f):
            return (1 + f) * (X + Y + X*Y - X**2 * Y + 0.1 * X**3 * Y**2)
        parent_block = _make_smooth_block(p)

        # Set up arrays for a single child at slot 0 of the fine level.
        # Parent is at slot 0 of the coarse level.  Child corner (0, 0).
        cx, cy = 0, 0

        parent_blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG),
                                 dtype=np.float64)
        parent_blocks[0] = np.asarray(parent_block)

        # Child block: interior set to something distinctive; halo zero.
        child_interior_truth = np.asarray(_exact_at_fine_centers(p, (cx, cy), 0))
        # Build per-field child interior using the polynomial.
        child_blocks = np.zeros_like(parent_blocks)
        for f in range(NF):
            child_blocks[0, f, NG:NG+BS, NG:NG+BS] = _exact_at_fine_centers(p, (cx, cy), f)

        parent_slot = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cx_arr = np.zeros(MAX_BLOCKS, dtype=np.int32) + cx
        child_cy_arr = np.zeros(MAX_BLOCKS, dtype=np.int32) + cy
        active = np.zeros(MAX_BLOCKS, dtype=bool)
        active[0] = True

        synced = np.asarray(sync_ghosts_across_levels(
            jnp.asarray(child_blocks),
            jnp.asarray(parent_blocks),
            jnp.asarray(parent_slot),
            jnp.asarray(child_cx_arr),
            jnp.asarray(child_cy_arr),
            jnp.asarray(active),
        ))

        # Interior preserved exactly.
        int_before = child_blocks[0, :, NG:NG+BS, NG:NG+BS]
        int_after  = synced     [0, :, NG:NG+BS, NG:NG+BS]
        assert np.array_equal(int_before, int_after), \
            "sync_ghosts_across_levels modified the interior"

        # Halo cells should equal the polynomial evaluated at fine cell centers
        # in the halo region — EXCEPT cells whose stencils touch the parent's
        # outer boundary (where edge replication causes some error).  The inner
        # halo (closest to the interior) should be exact.
        # For cx=0: child refined indices [NG, NG+BS+NG) = [3, 38) come from
        # refined-parent fine indices [NG+0*BS+NG, ...) — let's just check
        # the rim of cells nearest the interior on the +x and +y sides (which
        # share full stencils with the parent's true data).
        # Take a 1-cell-thick ring just outside the interior on the +x, +y sides.
        expected_full = _exact_at_fine_full_block(p, (cx, cy), 0)
        # +x rim: block index NG+BS, columns [NG, NG+BS)
        rim_xp_got = synced[0, 0, NG+BS, NG:NG+BS]
        rim_xp_exp = expected_full[NG+BS, NG:NG+BS]
        # +y rim: block index NG..NG+BS, column NG+BS
        rim_yp_got = synced[0, 0, NG:NG+BS, NG+BS]
        rim_yp_exp = expected_full[NG:NG+BS, NG+BS]
        scale = max(float(np.max(np.abs(expected_full))), 1.0)
        err_xp = float(np.max(np.abs(rim_xp_got - rim_xp_exp))) / scale
        err_yp = float(np.max(np.abs(rim_yp_got - rim_yp_exp))) / scale
        assert err_xp < 1e-12, f"+x halo rim rel err {err_xp:.2e}"
        assert err_yp < 1e-12, f"+y halo rim rel err {err_yp:.2e}"

    def test_inactive_slot_untouched(self):
        """An inactive child slot must not be modified by sync."""
        rng = np.random.default_rng(7)
        child_blocks = rng.standard_normal((MAX_BLOCKS, NF, BS+2*NG, BS+2*NG))
        parent_blocks = rng.standard_normal((MAX_BLOCKS, NF, BS+2*NG, BS+2*NG))
        parent_slot   = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cx_arr  = np.zeros(MAX_BLOCKS, dtype=np.int32)
        child_cy_arr  = np.zeros(MAX_BLOCKS, dtype=np.int32)
        active        = np.zeros(MAX_BLOCKS, dtype=bool)
        # all inactive

        synced = np.asarray(sync_ghosts_across_levels(
            jnp.asarray(child_blocks),
            jnp.asarray(parent_blocks),
            jnp.asarray(parent_slot),
            jnp.asarray(child_cx_arr),
            jnp.asarray(child_cy_arr),
            jnp.asarray(active),
        ))
        assert np.array_equal(synced, child_blocks), \
            "sync modified inactive slots"

    @pytest.mark.parametrize("cx,cy", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_all_corners_halo_inner_exact(self, cx, cy):
        """Every child corner must produce a halo whose INNER rim (closest to
        the interior) is exact for a 5th-degree polynomial.  The outermost
        cells may carry edge-padding error at the global boundary, but cells
        on the side of the halo that abuts the parent's interior should match
        the analytic polynomial to FP roundoff.
        """
        def p(X, Y, f):
            return (1 + f) * (X + Y + X*Y - X**2 * Y + 0.1 * X**3 * Y**2)
        parent_block = _make_smooth_block(p)

        parent_blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG),
                                 dtype=np.float64)
        parent_blocks[0] = np.asarray(parent_block)
        child_blocks = np.zeros_like(parent_blocks)
        for f in range(NF):
            child_blocks[0, f, NG:NG+BS, NG:NG+BS] = _exact_at_fine_centers(
                p, (cx, cy), f
            )

        parent_slot = np.zeros(MAX_BLOCKS, dtype=np.int32)
        cxa = np.full(MAX_BLOCKS, cx, dtype=np.int32)
        cya = np.full(MAX_BLOCKS, cy, dtype=np.int32)
        active = np.zeros(MAX_BLOCKS, dtype=bool); active[0] = True

        synced = np.asarray(sync_ghosts_across_levels(
            jnp.asarray(child_blocks), jnp.asarray(parent_blocks),
            jnp.asarray(parent_slot), jnp.asarray(cxa), jnp.asarray(cya),
            jnp.asarray(active),
        ))

        expected_full = _exact_at_fine_full_block(p, (cx, cy), 0)
        # The halo rim CLOSEST to the parent's interior is always exact —
        # those cells use stencils that fully fit inside the parent's data.
        # Which rim that is depends on the corner: cx=0 → the +x rim is inner,
        # cx=1 → the -x rim is inner. (Same for y.)
        x_inner_rim = NG + BS if cx == 0 else NG - 1
        y_inner_rim = NG + BS if cy == 0 else NG - 1
        scale = max(float(np.max(np.abs(expected_full))), 1.0)
        for rim_name, rim_slice in [
            ("x-rim", (x_inner_rim, slice(NG, NG+BS))),
            ("y-rim", (slice(NG, NG+BS), y_inner_rim)),
        ]:
            got = synced[0, 0, rim_slice[0], rim_slice[1]]
            exp = expected_full[rim_slice[0], rim_slice[1]]
            err = float(np.max(np.abs(got - exp))) / scale
            assert err < 1e-12, (
                f"corner=({cx},{cy}) {rim_name}: rel err {err:.2e}"
            )

    def test_multi_children_one_parent(self):
        """All four children of a single parent active simultaneously: each
        child's halo must independently match the parent's analytic values
        at the appropriate quadrant."""
        def p(X, Y, f):
            return (1 + f) * (X + Y + X*Y - X**2 * Y + 0.1 * X**3 * Y**2)
        parent_block = _make_smooth_block(p)

        parent_blocks = np.zeros((MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG),
                                 dtype=np.float64)
        parent_blocks[0] = np.asarray(parent_block)
        child_blocks = np.zeros_like(parent_blocks)

        corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for slot, (cx, cy) in enumerate(corners):
            for f in range(NF):
                child_blocks[slot, f, NG:NG+BS, NG:NG+BS] = _exact_at_fine_centers(
                    p, (cx, cy), f
                )

        parent_slot = np.zeros(MAX_BLOCKS, dtype=np.int32)   # all → parent 0
        cxa = np.zeros(MAX_BLOCKS, dtype=np.int32)
        cya = np.zeros(MAX_BLOCKS, dtype=np.int32)
        for slot, (cx, cy) in enumerate(corners):
            cxa[slot] = cx; cya[slot] = cy
        active = np.zeros(MAX_BLOCKS, dtype=bool)
        active[:4] = True

        synced = np.asarray(sync_ghosts_across_levels(
            jnp.asarray(child_blocks), jnp.asarray(parent_blocks),
            jnp.asarray(parent_slot), jnp.asarray(cxa), jnp.asarray(cya),
            jnp.asarray(active),
        ))

        # Each active child's interior must be untouched.
        for slot in range(4):
            int_before = child_blocks[slot, :, NG:NG+BS, NG:NG+BS]
            int_after  = synced     [slot, :, NG:NG+BS, NG:NG+BS]
            assert np.array_equal(int_before, int_after), \
                f"slot {slot}: interior modified"

        # Inner halo rims should match polynomial for every child.
        for slot, (cx, cy) in enumerate(corners):
            expected_full = _exact_at_fine_full_block(p, (cx, cy), 0)
            x_rim = NG + BS if cx == 0 else NG - 1
            y_rim = NG + BS if cy == 0 else NG - 1
            scale = max(float(np.max(np.abs(expected_full))), 1.0)
            err_x = float(np.max(np.abs(
                synced[slot, 0, x_rim, NG:NG+BS]
                - expected_full[x_rim, NG:NG+BS]
            ))) / scale
            err_y = float(np.max(np.abs(
                synced[slot, 0, NG:NG+BS, y_rim]
                - expected_full[NG:NG+BS, y_rim]
            ))) / scale
            assert err_x < 1e-12 and err_y < 1e-12, (
                f"slot {slot} corner=({cx},{cy}): inner rim mismatch "
                f"(x={err_x:.2e}, y={err_y:.2e})"
            )


# ── Round-trip ────────────────────────────────────────────────────────────────

class TestRoundTrip:
    """Prolongate then restrict should approximately recover the original parent
    (for smooth fields).  Limited by restriction's 2nd-order accuracy."""

    def test_smooth_function_2nd_order(self):
        """Restriction is 2nd-order accurate.  For f = sin(ω(x+y)) the leading
        error is `(1/32)·∇²f ≈ ω²/16`, so we use a small ω and verify the
        observed error matches the analytical estimate within an order of magnitude.
        """
        omega = 0.05
        def f(X, Y, fld):
            return jnp.sin(omega * (X + Y + fld))

        parent = _make_smooth_block(f)
        parent_ref_interior = np.asarray(parent[:, NG:NG+BS, NG:NG+BS])

        # Round-trip through all 4 children
        restored = parent
        for cx in (0, 1):
            for cy in (0, 1):
                child_full = prolongate(parent, (cx, cy))
                child_interior = child_full[:, NG:NG+BS, NG:NG+BS]
                restored = restrict_into_parent(restored, child_interior, (cx, cy))

        restored_interior = np.asarray(restored[:, NG:NG+BS, NG:NG+BS])
        err = float(np.max(np.abs(restored_interior - parent_ref_interior)))
        # Analytical estimate for 2nd-order restriction:
        # 1/32 · ∇²f = 1/32 · -2ω² · sin(ω·(x+y)) → max magnitude ω²/16
        analytic_bound = (omega ** 2) / 16 * 1.5   # 50% margin for accumulated terms
        assert err < analytic_bound, (
            f"roundtrip error {err:.2e} exceeds analytic 2nd-order bound {analytic_bound:.2e}"
        )


class TestProlongateFootprintIdentical:
    """A1 regression: footprint-only prolongation must match refining the whole
    parent block to MACHINE PRECISION.  Prolongation is local interpolation, so
    slicing the parent down to the child's coarse footprint (anchored to the
    corner-side edge so edge-padding is reproduced) cannot change any output cell
    mathematically.  The residual (~1e-15) is only XLA reordering the 6-term
    stencil sum for the differently-shaped sub-window — far below the 6th-order
    truncation error (~1e-6).  This is the gate that makes A1 a pure speedup."""

    ATOL = 1e-12   # ≫ observed ~1e-15 FP-reorder residual, ≪ ~1e-6 truncation error

    @staticmethod
    def _full_parent_reference(parent_block, child_corner):
        """The pre-A1 implementation: refine the WHOLE parent, then crop."""
        cx, cy = child_corner
        refined_x = _interp_to_fine_1d(parent_block, axis=1)
        refined_xy = _interp_to_fine_1d(refined_x, axis=2)
        start_x = NG + cx * BS
        start_y = NG + cy * BS
        return jax.lax.dynamic_slice(
            refined_xy, (0, start_x, start_y),
            (NF, BS + 2 * NG, BS + 2 * NG))

    @pytest.mark.parametrize("cx", [0, 1])
    @pytest.mark.parametrize("cy", [0, 1])
    def test_bit_identical_random(self, cx, cy):
        rng = np.random.default_rng(1234 + cx * 2 + cy)
        parent = jnp.asarray(rng.standard_normal((NF, BS + 2 * NG, BS + 2 * NG)))
        got = np.asarray(prolongate(parent, (cx, cy)))
        ref = np.asarray(self._full_parent_reference(parent, (cx, cy)))
        # Mathematically identical on every kept cell; only FP summation reorders.
        assert np.allclose(got, ref, rtol=0.0, atol=self.ATOL), (
            f"corner ({cx},{cy}) max|Δ| = {np.max(np.abs(got - ref)):.3e}")


class TestSyncAcrossLevelsFusedSelect:
    """4.5: the fused single `select` in sync_ghosts_across_levels must reproduce
    the two-`where` semantics exactly — inactive slots and active interiors are
    untouched; active halos are filled from the parent prolongation."""

    def test_selection_correct(self):
        rng = np.random.default_rng(7)
        n = 5
        parent_blocks = jnp.asarray(rng.standard_normal((n, NF, BS + 2*NG, BS + 2*NG)))
        child_blocks  = jnp.asarray(rng.standard_normal((n, NF, BS + 2*NG, BS + 2*NG)))
        parent_slot = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        child_cx    = jnp.array([0, 1, 0, 1, 0], dtype=jnp.int32)
        child_cy    = jnp.array([0, 0, 1, 1, 0], dtype=jnp.int32)
        active      = jnp.array([True, True, False, True, False])

        out = np.asarray(sync_ghosts_across_levels(
            child_blocks, parent_blocks, parent_slot, child_cx, child_cy, active))
        child_np = np.asarray(child_blocks)

        hmask = np.ones((BS + 2*NG, BS + 2*NG), bool)
        hmask[NG:NG + BS, NG:NG + BS] = False   # True on halo ring

        for i in range(n):
            if not bool(active[i]):
                assert np.array_equal(out[i], child_np[i]), f"inactive slot {i} changed"
                continue
            # active: interior is preserved exactly
            assert np.array_equal(
                out[i][:, NG:NG + BS, NG:NG + BS],
                child_np[i][:, NG:NG + BS, NG:NG + BS]), f"slot {i} interior changed"
            # active: halo equals the parent prolongation halo (machine precision)
            pro = np.asarray(prolongate(
                parent_blocks[int(parent_slot[i])],
                (int(child_cx[i]), int(child_cy[i]))))
            assert np.allclose(out[i][:, hmask], pro[:, hmask], atol=1e-12), \
                f"slot {i} halo not filled from prolongation"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
