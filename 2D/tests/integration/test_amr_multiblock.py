"""
Phase 5.1 / 5.2 — multi-block fine levels & within-level ghost sync.

The correctness gap this closes: before 5.1, only the root had within-level
(periodic) sync; where two same-level FINE blocks met, the shared-face halo was
filled by prolongation from the coarse parent (approximate).  Now adjacent
same-level blocks copy each other's exact interior into the shared-face halo.

Coverage (deliberately exhaustive — this is correctness-critical infrastructure):

  TestWithinLevelKernel       — the sync_ghosts_within_level kernel in isolation:
                                exact face copy on all 4 faces, boundary faces and
                                corner halos untouched, invalid/inactive faces ignored.
  TestWithinLevelRHSEquiv     — THE rigorous proof: two adjacent fine blocks, after
                                within-level sync, produce byte-for-byte the same RHS
                                on their interiors as one single grid over both blocks
                                (in x and in y).  Includes a "teeth" check that WITHOUT
                                the sync the result is wrong.
  TestNeighborTopology        — rebuild_neighbors / to_jax_arrays produce the correct
                                neighbour slots & validity (incl. patch-boundary = none).
  TestProperNesting           — check_proper_nesting passes on a valid hierarchy and
                                flags an orphaned/mis-nested block.
  TestMultiBlockNoRecompile   — the N-level step traces exactly once on a genuine
                                multi-block hierarchy (shape stability preserved).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import (
    BS, NG, NF, LEVELS, MAX_BLOCKS, REFINE_RATIO,
    AMRState, AMRTopology, make_root_state,
)
from mcs2d.amr.kernels import sync_ghosts_within_level
from mcs2d.amr.regrid import apply_flags, REFINE
from mcs2d.amr.evolve import make_n_level_step, make_subcycled_n_level_step
from mcs2d.schemes.fused_rhs_fp import _make_kernel_fn

# Face order used everywhere: 0=-x, 1=+x, 2=-y, 3=+y.
HALO = NG
H = BS + 2 * NG          # full block extent per axis
I0, I1 = NG, NG + BS     # interior slice


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_with_interior(interior, halo_fill):
    """(NF,BS,BS) interior → (NF,H,H) block with halos set to `halo_fill`."""
    blk = np.full((NF, H, H), halo_fill, dtype=np.float64)
    blk[:, I0:I1, I0:I1] = interior
    return blk


# ── 1. Kernel in isolation ──────────────────────────────────────────────────

class TestWithinLevelKernel:
    """Exhaustive unit tests of sync_ghosts_within_level on a 2×2 block patch."""

    def _setup_2x2(self, halo_fill=-999.0):
        """Four mutually-adjacent fine blocks tiling a (2BS, 2BS) global interior.
        Slot layout: A=(0,0)=0, B=(1,0)=1, C=(0,1)=2, D=(1,1)=3 in block units."""
        rng = np.random.default_rng(42)
        f = rng.standard_normal((NF, 2 * BS, 2 * BS))   # global interior field

        def interior(bx, by):
            return f[:, bx*BS:(bx+1)*BS, by*BS:(by+1)*BS]

        blocks = np.stack([
            _block_with_interior(interior(0, 0), halo_fill),   # A slot0 (0,0)
            _block_with_interior(interior(1, 0), halo_fill),   # B slot1 (1,0)
            _block_with_interior(interior(0, 1), halo_fill),   # C slot2 (0,1)
            _block_with_interior(interior(1, 1), halo_fill),   # D slot3 (1,1)
        ])

        # neighbour_slot[s, face], face = -x,+x,-y,+y; -1→invalid (filler 0, valid False)
        ns = np.zeros((4, 4), np.int32)
        nv = np.zeros((4, 4), bool)
        # A(0,0): +x→B(1), +y→C(2)
        ns[0, 1], nv[0, 1] = 1, True
        ns[0, 3], nv[0, 3] = 2, True
        # B(1,0): -x→A(0), +y→D(3)
        ns[1, 0], nv[1, 0] = 0, True
        ns[1, 3], nv[1, 3] = 3, True
        # C(0,1): +x→D(3), -y→A(0)
        ns[2, 1], nv[2, 1] = 3, True
        ns[2, 2], nv[2, 2] = 0, True
        # D(1,1): -x→C(2), -y→B(1)
        ns[3, 0], nv[3, 0] = 2, True
        ns[3, 2], nv[3, 2] = 1, True

        out = np.asarray(sync_ghosts_within_level(
            jnp.asarray(blocks), jnp.asarray(ns), jnp.asarray(nv)))
        return f, blocks, out

    def test_shared_faces_exact_all_directions(self):
        f, _, out = self._setup_2x2()
        # A(slot0): +x halo == B interior's first NG rows == f[BS:BS+NG, 0:BS]
        assert np.array_equal(out[0][:, I1:H, I0:I1], f[:, BS:BS+NG, 0:BS])
        # A: +y halo == C interior's first NG cols == f[0:BS, BS:BS+NG]
        assert np.array_equal(out[0][:, I0:I1, I1:H], f[:, 0:BS, BS:BS+NG])
        # B(slot1): -x halo == A interior's last NG rows == f[BS-NG:BS, 0:BS]
        assert np.array_equal(out[1][:, 0:NG, I0:I1], f[:, BS-NG:BS, 0:BS])
        # D(slot3): -y halo == B interior's last NG cols == f[BS:2BS, BS-NG:BS]
        assert np.array_equal(out[3][:, I0:I1, 0:NG], f[:, BS:2*BS, BS-NG:BS])

    def test_boundary_faces_untouched(self):
        _, blk, out = self._setup_2x2(halo_fill=-999.0)
        # A has NO -x and NO -y neighbour → those halos stay at the sentinel.
        assert np.all(out[0][:, 0:NG, I0:I1] == -999.0), "A -x halo was modified"
        assert np.all(out[0][:, I0:I1, 0:NG] == -999.0), "A -y halo was modified"
        # D has NO +x and NO +y neighbour.
        assert np.all(out[3][:, I1:H, I0:I1] == -999.0), "D +x halo was modified"
        assert np.all(out[3][:, I0:I1, I1:H] == -999.0), "D +y halo was modified"

    def test_corner_halos_never_written(self):
        _, _, out = self._setup_2x2(halo_fill=-999.0)
        # All four corner halo squares of every block stay at the sentinel —
        # the separable stencils never read them, so the kernel must not touch them.
        for s in range(4):
            for ri in (slice(0, NG), slice(I1, H)):
                for ci in (slice(0, NG), slice(I1, H)):
                    assert np.all(out[s][:, ri, ci] == -999.0), \
                        f"corner halo written on slot {s}"

    def test_interiors_unchanged(self):
        _, blk, out = self._setup_2x2()
        for s in range(4):
            assert np.array_equal(out[s][:, I0:I1, I0:I1], blk[s][:, I0:I1, I0:I1])

    def test_invalid_face_ignored(self):
        # A single block whose neighbour_slot points somewhere but valid=False:
        # output must be identical to input (no copy performed).
        rng = np.random.default_rng(7)
        blocks = jnp.asarray(rng.standard_normal((3, NF, H, H)))
        ns = jnp.array([[1, 2, 1, 2]] * 3, dtype=jnp.int32)   # arbitrary slots
        nv = jnp.zeros((3, 4), bool)                          # but all invalid
        out = sync_ghosts_within_level(blocks, ns, nv)
        assert jnp.array_equal(out, blocks)


# ── 2. The rigorous correctness proof: RHS equivalence vs a single grid ───────

class TestWithinLevelRHSEquiv:
    """Two adjacent fine blocks + within-level sync must reproduce the RHS of a
    single grid spanning both — exactly where the shared face matters."""

    PHYS = dict(dx=0.01, dy=0.01, cs=1.0, L=2.0, K1=1.0, K2=1.0, ko_sigma=0.05)

    def _kernel(self):
        p = self.PHYS
        return _make_kernel_fn(p["dx"], p["dy"], p["cs"], p["L"],
                               p["K1"], p["K2"], p["ko_sigma"])

    def test_two_blocks_match_single_grid_x(self):
        """Blocks adjacent along x (B is A's +x neighbour)."""
        rng = np.random.default_rng(1)
        kern = self._kernel()
        # Single reference grid: (NF, 2BS+2NG, BS+2NG) → RHS (NF, 2BS, BS).
        G = jnp.asarray(rng.standard_normal((NF, 2*BS + 2*NG, BS + 2*NG)))
        ref = np.asarray(kern(G))                       # (NF, 2BS, BS)

        # Split into A = G rows [0:H], B = G rows [BS:BS+H]; ZERO the shared face
        # halos so only within-level sync can restore them.
        A = np.asarray(G[:, 0:H, :]).copy()
        B = np.asarray(G[:, BS:BS+H, :]).copy()
        A[:, I1:H, :] = 0.0      # A's +x halo wiped
        B[:, 0:NG, :] = 0.0      # B's -x halo wiped
        blocks = jnp.asarray(np.stack([A, B]))

        ns = np.zeros((2, 4), np.int32); nv = np.zeros((2, 4), bool)
        ns[0, 1], nv[0, 1] = 1, True     # A +x → B
        ns[1, 0], nv[1, 0] = 0, True     # B -x → A
        synced = sync_ghosts_within_level(blocks, jnp.asarray(ns), jnp.asarray(nv))

        rhs = np.asarray(jax.vmap(kern)(synced))        # (2, NF, BS, BS)
        assert np.allclose(rhs[0], ref[:, 0:BS, :], rtol=1e-9, atol=1e-12), \
            "block A RHS != single-grid RHS over its region"
        assert np.allclose(rhs[1], ref[:, BS:2*BS, :], rtol=1e-9, atol=1e-12), \
            "block B RHS != single-grid RHS over its region"

    def test_two_blocks_match_single_grid_y(self):
        """Blocks adjacent along y (B is A's +y neighbour)."""
        rng = np.random.default_rng(2)
        kern = self._kernel()
        G = jnp.asarray(rng.standard_normal((NF, BS + 2*NG, 2*BS + 2*NG)))
        ref = np.asarray(kern(G))                       # (NF, BS, 2BS)

        A = np.asarray(G[:, :, 0:H]).copy()
        B = np.asarray(G[:, :, BS:BS+H]).copy()
        A[:, :, I1:H] = 0.0      # A's +y halo wiped
        B[:, :, 0:NG] = 0.0      # B's -y halo wiped
        blocks = jnp.asarray(np.stack([A, B]))

        ns = np.zeros((2, 4), np.int32); nv = np.zeros((2, 4), bool)
        ns[0, 3], nv[0, 3] = 1, True     # A +y → B
        ns[1, 2], nv[1, 2] = 0, True     # B -y → A
        synced = sync_ghosts_within_level(blocks, jnp.asarray(ns), jnp.asarray(nv))

        rhs = np.asarray(jax.vmap(kern)(synced))
        assert np.allclose(rhs[0], ref[:, :, 0:BS], rtol=1e-9, atol=1e-12)
        assert np.allclose(rhs[1], ref[:, :, BS:2*BS], rtol=1e-9, atol=1e-12)

    def test_without_sync_is_wrong(self):
        """Teeth: the SAME setup WITHOUT within-level sync must NOT match the
        single-grid RHS near the shared face — proving the test can fail."""
        rng = np.random.default_rng(1)
        kern = self._kernel()
        G = jnp.asarray(rng.standard_normal((NF, 2*BS + 2*NG, BS + 2*NG)))
        ref = np.asarray(kern(G))
        A = np.asarray(G[:, 0:H, :]).copy(); A[:, I1:H, :] = 0.0
        rhs_A = np.asarray(kern(jnp.asarray(A)))        # NO sync applied
        # The last interior rows (which read the wiped +x halo) must differ.
        assert not np.allclose(rhs_A[:, BS-1, :], ref[:, BS-1, :], atol=1e-8)


# ── 3. Neighbour topology ─────────────────────────────────────────────────────

class TestNeighborTopology:

    def _refined_root_block(self):
        """Root 2×2; refine root slot 0 → four mutually-adjacent L1 children."""
        data = jnp.zeros((NF, 2*BS + 2*NG, 2*BS + 2*NG))
        state, topo = make_root_state(data, nbx_root=2, nby_root=2)
        flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        return state, topo

    def test_four_children_are_mutually_adjacent(self):
        _, topo = self._refined_root_block()
        topo.rebuild_neighbors()
        assert topo.n_active(1) == 4, "refining one root block should make 4 L1 children"
        # Each of the 4 children must have exactly TWO same-level neighbours
        # (it's a corner of the 2×2 fine patch) and two patch-boundary faces.
        l1_slots = [s for s in range(topo.caps[1]) if topo.active[1, s]]
        for s in l1_slots:
            valid_faces = [topo.neighbors[(1, s, f)] for f in range(4)]
            n_real = sum(1 for v in valid_faces if v is not None)
            assert n_real == 2, f"L1 slot {s} has {n_real} neighbours, expected 2"

    def test_neighbor_links_are_reciprocal(self):
        _, topo = self._refined_root_block()
        topo.rebuild_neighbors()
        opp = {0: 1, 1: 0, 2: 3, 3: 2}    # -x↔+x, -y↔+y
        for (L, s, f), nb in topo.neighbors.items():
            if nb is None:
                continue
            _, ns = nb
            assert topo.neighbors[(L, ns, opp[f])] == (L, s), \
                f"non-reciprocal link {(L,s,f)}→{nb}"

    def test_to_jax_arrays_neighbor_fields(self):
        _, topo = self._refined_root_block()
        ta = topo.to_jax_arrays()
        # Shapes: per level (caps[L], 4).
        for L in range(LEVELS):
            assert ta.neighbor_slot[L].shape == (topo.caps[L], 4)
            assert ta.neighbor_valid[L].shape == (topo.caps[L], 4)
        # L1 has 4 active children, each with 2 valid faces → 8 valid entries.
        assert int(np.asarray(ta.neighbor_valid[1]).sum()) == 8
        # Every valid neighbour slot must itself be an active L1 block.
        nv = np.asarray(ta.neighbor_valid[1]); ns = np.asarray(ta.neighbor_slot[1])
        act1 = np.asarray(topo.active[1, :topo.caps[1]])
        for s in range(topo.caps[1]):
            for f in range(4):
                if nv[s, f]:
                    assert act1[ns[s, f]], f"neighbour slot {ns[s,f]} not active"

    def test_single_child_has_no_neighbors(self):
        """Refining only ONE quadrant of a parent (manually) → an L1 block with
        no same-level neighbours; all faces fall back to prolongation."""
        data = jnp.zeros((NF, BS + 2*NG, BS + 2*NG))
        state, topo = make_root_state(data, nbx_root=1, nby_root=1)
        # Manually add a single L1 child covering the (0,0) quadrant of root 0.
        topo.add_block(level=1, slot=0, bbox_ij=(0, 0), parent=(0, 0))
        topo.rebuild_neighbors()
        assert all(topo.neighbors[(1, 0, f)] is None for f in range(4))


# ── 4. Proper nesting ─────────────────────────────────────────────────────────

class TestProperNesting:

    def test_valid_multiblock_hierarchy(self):
        data = jnp.zeros((NF, 2*BS + 2*NG, 2*BS + 2*NG))
        state, topo = make_root_state(data, nbx_root=2, nby_root=2)
        flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        assert topo.check_proper_nesting() == [], \
            "freshly-refined hierarchy should be properly nested"

    def test_detects_inactive_parent(self):
        data = jnp.zeros((NF, 2*BS + 2*NG, 2*BS + 2*NG))
        state, topo = make_root_state(data, nbx_root=2, nby_root=2)
        flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)
        # Forcibly deactivate the parent without removing the children → violation.
        topo.active[0, 0] = False
        violations = topo.check_proper_nesting()
        assert len(violations) == 4, f"expected 4 orphaned children, got {violations}"

    def test_detects_orphan(self):
        data = jnp.zeros((NF, BS + 2*NG, BS + 2*NG))
        state, topo = make_root_state(data, nbx_root=1, nby_root=1)
        # Add an L1 block with NO parent recorded → "no parent" violation.
        topo.active[1, 0] = True
        topo.bbox_ijk[(1, 0)] = (0, 0)
        violations = topo.check_proper_nesting()
        assert any(v[:2] == (1, 0) and "no parent" in v[2] for v in violations)


# ── 5. Shape stability under genuine multi-block ──────────────────────────────

class TestMultiBlockNoRecompile:

    @staticmethod
    def _trace_counter(fn):
        """(counted_fn, counter) — counter[0] increments on each real trace."""
        counter = [0]

        def counted(*a, **k):
            counter[0] += 1
            return fn(*a, **k)
        return counted, counter

    def test_n_level_step_single_trace_multiblock(self):
        nbx = nby = 2
        dx = dy = 10.0 / (nbx * BS)
        data = jnp.asarray(
            np.random.default_rng(0).standard_normal((NF, 2*BS + 2*NG, 2*BS + 2*NG)))
        state, topo = make_root_state(data, nbx_root=nbx, nby_root=nby)
        flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)   # 4 adjacent L1 children
        assert topo.n_active(1) == 4

        step = make_n_level_step(
            dx_root=dx, dy_root=dy, dt=0.01 * dx,
            cs=1.0, L_coupling=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=nbx, nby_root=nby)
        counted, counter = self._trace_counter(step.__wrapped__)
        rejitted = jax.jit(counted)

        ta = topo.to_jax_arrays()
        for _ in range(4):
            state = rejitted(state, ta)
            state.blocks[0].block_until_ready()
        assert counter[0] == 1, f"multi-block step re-traced {counter[0]} times"


# ── 6. End-to-end wiring: the step actually applies within-level sync ─────────

class TestMultiBlockStepWiring:
    """Kernel-level tests prove the sync is correct; these prove the evolve steps
    actually CALL it — by checking that after a real step, a fine block's shared
    face halo equals its neighbour's interior edge (only within-level sync can
    make that true; prolongation from the coarse parent would not)."""

    def _multiblock(self, nbx=2, nby=2):
        dx = dy = 10.0 / (nbx * BS)
        data = jnp.asarray(np.random.default_rng(3).standard_normal(
            (NF, nbx*BS + 2*NG, nby*BS + 2*NG)))
        state, topo = make_root_state(data, nbx_root=nbx, nby_root=nby)
        flags = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
        flags[0, 0] = REFINE
        state, topo = apply_flags(state, topo, flags)   # 4 adjacent L1 children
        topo.rebuild_neighbors()
        return state, topo, dx

    @staticmethod
    def _find_plus_x_pair(topo):
        for s in range(topo.caps[1]):
            if topo.active[1, s] and topo.neighbors.get((1, s, 1)) is not None:
                return s, topo.neighbors[(1, s, 1)][1]
        raise AssertionError("no +x neighbour pair found among L1 children")

    def test_n_level_step_synced_halos(self):
        state, topo, dx = self._multiblock()
        A, B = self._find_plus_x_pair(topo)
        step = make_n_level_step(
            dx_root=dx, dy_root=dx, dt=0.01 * dx,
            cs=1.0, L_coupling=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=2, nby_root=2)
        out = step(state, topo.to_jax_arrays())
        l1 = np.asarray(out.blocks[1])
        assert np.isfinite(l1).all(), "non-finite after multi-block step"
        # A's +x halo must equal B's first NG interior rows (within-level sync).
        assert np.allclose(l1[A][:, I1:H, I0:I1], l1[B][:, NG:2*NG, I0:I1],
                           rtol=1e-9, atol=1e-12), \
            "n-level step did not within-level-sync the shared face"

    def test_subcycled_step_runs_and_synced_halos(self):
        state, topo, dx = self._multiblock()
        A, B = self._find_plus_x_pair(topo)
        step = make_subcycled_n_level_step(
            dx_root=dx, dy_root=dx, dt_root=0.01 * dx,
            cs=1.0, L_coupling=2.0, K1=1.0, K2=1.0, ko_sigma=0.05,
            nbx_root=2, nby_root=2)
        out = step(state, topo.to_jax_arrays())
        l1 = np.asarray(out.blocks[1])
        assert np.isfinite(l1).all(), "non-finite after sub-cycled multi-block step"
        assert np.allclose(l1[A][:, I1:H, I0:I1], l1[B][:, NG:2*NG, I0:I1],
                           rtol=1e-9, atol=1e-12), \
            "sub-cycled step did not within-level-sync the shared face"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
