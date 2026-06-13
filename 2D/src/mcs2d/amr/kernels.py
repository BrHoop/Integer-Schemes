"""
Jitted per-block kernels for block-structured AMR in 2D.

These functions take FIXED-SHAPE inputs and never recompile.  They're the
heavy-compute building blocks; the Python driver in amr.py orchestrates them.

Convention
----------
Each block has shape (NF, BS+2*NG, BS+2*NG) — interior (BS, BS) surrounded by
an NG-cell halo.  Spatial axes are last; the first axis is the field index.

Cell-centered AMR
-----------------
A coarse cell of width Δ centered at position x refines into TWO fine cells of
width Δ/2 centered at x - Δ/4 and x + Δ/4.  Both fine cell centers are at half-
quarter offsets from the coarse cell center, so we need different Lagrange
weights for each:

  W_LEFT  = weights at offset -0.25 using 6-point stencil {-3,-2,-1,0,1,2}
  W_RIGHT = weights at offset +0.25 using 6-point stencil {-2,-1,0,1,2,3}

By symmetry W_LEFT = reverse(W_RIGHT).  Each set is 6th-order accurate.

Prolongation
------------
Parent → 1 child FULL block (interior + halo).  Each of the 4 children (2D)
tiles a 16×16 region of the parent's 32×32 interior, refining to 32×32 of the
child's interior.  The child's halo cells are also produced by interpolating
the parent (using its own halo as the source) — this doubles as cross-level
ghost-zone fill.  Implementation (A1): slice the parent down to the child's
coarse footprint window, refine only that, then crop to the child's window —
bit-identical to refining the whole parent (see `_prolong_window`).

Restriction
-----------
4 child interiors → 1 parent interior, by 2×2 cell averaging (volume-weighted
in the trivial uniform-cell case).  Parent halo cells are not modified.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mcs2d.amr.state import BS, NG, NF, REFINE_RATIO


# ── Lagrange interpolation weights ─────────────────────────────────────────────

def _lagrange_weights(stencil_offsets, target_offset):
    """Compute Lagrange interpolation weights for evaluating a polynomial at
    `target_offset` given values at integer offsets `stencil_offsets`.

    For 6 stencil points the interpolant is 5th-degree → 6th-order accurate.
    """
    weights = []
    for j, x_j in enumerate(stencil_offsets):
        w = 1.0
        for m, x_m in enumerate(stencil_offsets):
            if m != j:
                w *= (target_offset - x_m) / (x_j - x_m)
        weights.append(w)
    return np.array(weights, dtype=np.float64)


# Cell-centered AMR convention: coarse cell k has center at position k and
# spans [k-0.5, k+0.5].  Refining gives 2 fine cells, the LEFT half centered
# at k-0.25 and the RIGHT half at k+0.25.  Both halves of the same anchor
# share a 6-cell stencil; only the weights differ.
_STENCIL_OFFSETS = np.array([-2, -1, 0, 1, 2, 3], dtype=np.int32)
_W_LEFT  = jnp.array(_lagrange_weights(_STENCIL_OFFSETS, -0.25))   # LEFT half of anchor
_W_RIGHT = jnp.array(_lagrange_weights(_STENCIL_OFFSETS, +0.25))   # RIGHT half of anchor


# Cell-centered restriction (fine → coarse).  A coarse cell of center 0 is
# surrounded by 6 fine cells at coarse-units offsets {-1.25, -0.75, -0.25,
# +0.25, +0.75, +1.25} (the 2 fine cells inside it plus the 2 nearest in each
# neighbour).  Lagrange weights that evaluate at offset 0 from these 6 nodes
# give a 6th-order-accurate fine→coarse transfer — the high-order replacement
# for plain 2×2 averaging (which is only 2nd-order).  By symmetry the weights
# are even.  The fine-index offsets relative to 2*c are {-2,-1,0,1,2,3}.
_RST_FINE_OFFSETS = np.array([-2, -1, 0, 1, 2, 3], dtype=np.int32)
_RST_NODE_POS     = np.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25], dtype=np.float64)
_W_RESTRICT = jnp.array(_lagrange_weights(_RST_NODE_POS, 0.0))


# ── 1-D cell-centered restriction (6th order) ─────────────────────────────────

def _restrict_to_coarse_1d(fine_full: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Map the fine cells of a full block (size BS+2*NG along `axis`) down to
    BS//2 coarse cells (the restriction of the fine INTERIOR), 6th-order.

    Coarse output cell c (c = 0..BS//2-1) sits at fine interior-index 2c; its
    stencil is fine interior-indices {2c-2 .. 2c+3} = full-block indices
    {NG+2c-2 .. NG+2c+3}.  Those reach 2 cells into the fine halo at the
    interior edges (NG=3 ≥ 2 covers it), so the full block (with halo) must be
    passed in.
    """
    ncoarse = BS // REFINE_RATIO
    base = NG                       # interior start in full-block coords
    out = 0.0
    for k, w in zip(_RST_FINE_OFFSETS.tolist(), _W_RESTRICT):
        start = base + k            # full-block index of the c=0 stencil point
        # Strided gather: one fine cell per coarse output, stride = REFINE_RATIO.
        sl = jax.lax.slice_in_dim(
            fine_full, start, start + REFINE_RATIO * ncoarse, stride=REFINE_RATIO,
            axis=axis,
        )
        out = out + w * sl
    return out                      # size ncoarse along `axis`


# ── 1-D cell-centered prolongation ────────────────────────────────────────────

def _interp_to_fine_1d(coarse: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Map N coarse cells along `axis` to 2N fine cells.

    For each coarse cell k, output two fine cells positioned at +0.25 and +0.75
    inside the coarse cell's extent (i.e. fine[2k] is the LEFT half-cell of
    coarse cell k, fine[2k+1] is the RIGHT half-cell).  Both use the same
    6-cell stencil {coarse[k-2], ..., coarse[k+3]} with the appropriate weights.

    Boundary cells use edge-mode padding for stencil halo; their results are
    only valid for cells with a full 6-point stencil, so the caller typically
    crops out the outermost fine cells.
    """
    N = coarse.shape[axis]

    # Pad to allow stencil offsets {-2..+3} for every anchor.  3 cells right is
    # the deepest reach; 2 cells left is the leftmost.  Use (3, 3) for safety
    # (so wrapper-level cropping has consistent slack).
    pw = [(0, 0)] * coarse.ndim
    pw[axis] = (3, 3)
    padded = jnp.pad(coarse, pw, mode='edge')

    # In padded coords, coarse cell k is at padded index k+3.
    # Stencil offsets {-2..+3} → padded indices {k+1, k+2, ..., k+6}.
    stencils = [jax.lax.dynamic_slice_in_dim(padded, s, N, axis=axis)
                for s in range(1, 7)]

    fine_left  = sum(w * s for w, s in zip(_W_LEFT,  stencils))   # (..., N, ...)
    fine_right = sum(w * s for w, s in zip(_W_RIGHT, stencils))

    # Interleave: fine[2k] = LEFT half of anchor k, fine[2k+1] = RIGHT half.
    fl = jnp.expand_dims(fine_left,  axis=axis + 1)
    fr = jnp.expand_dims(fine_right, axis=axis + 1)
    interleaved = jnp.concatenate([fl, fr], axis=axis + 1)

    new_shape = list(coarse.shape)
    new_shape[axis] = 2 * N
    return interleaved.reshape(new_shape)


# ── Footprint-only prolongation window (A1) ───────────────────────────────────
# A child quadrant only draws from its own coarse footprint (BS//2 interior cells)
# plus the interpolation stencil reach, so refining the WHOLE parent wastes ~3/4 of
# the work.  We slice the parent down to a coarse window of width `_PROLONG_W`
# before the two `_interp_to_fine_1d` calls.  The window is anchored to the
# corner-side parent edge so `_interp_to_fine_1d`'s edge-padding is reproduced
# bit-for-bit ⇒ identical output to refining the full parent (gated by
# test_prolongate_footprint_identical).
_PARENT_W = BS + 2 * NG
_INTERP_L, _INTERP_R = 2, 3            # stencil reach of _STENCIL_OFFSETS = [-2..+3]


def _prolong_window() -> int:
    # cx=0 quadrant (low side, anchored at coarse 0): rightmost coarse cell its
    # fine window [NG, NG+_PARENT_W) touches, + right stencil reach.
    w_lo = (NG + _PARENT_W - 1) // 2 + _INTERP_R + 1
    # cx=1 quadrant (high side, anchored at the parent's right edge): width needed
    # to reach the leftmost coarse cell its fine window touches, - left reach.
    w_hi = _PARENT_W - ((NG + BS) // 2 - _INTERP_L)
    return max(w_lo, w_hi)


_PROLONG_W = _prolong_window()              # coarse window width fed to refinement
_PROLONG_OFF = _PARENT_W - _PROLONG_W       # window start for the high (cx=1) quadrant
_PROLONG_SUBSTEP = BS - 2 * _PROLONG_OFF    # child-window offset step per corner index
assert 2 * _PROLONG_W >= (NG + _PROLONG_SUBSTEP) + _PARENT_W, "prolong window too small"


# ── Prolongation: parent → full child block (interior + halo) ─────────────────

@partial(jax.jit, static_argnames=('child_corner',))
def prolongate(parent_block: jnp.ndarray, child_corner: tuple) -> jnp.ndarray:
    """Generate a child's FULL block (interior + halo) from its parent.

    Args:
      parent_block: (NF, BS+2*NG, BS+2*NG) — parent's data including halo.
      child_corner: (cx, cy) ∈ {0,1}² — which quadrant of the parent's interior
                    this child covers.

    Returns:
      child_block: (NF, BS+2*NG, BS+2*NG) — full fine block with halo.

    Halo cells of the child are also interpolated from the parent; they cover
    fine positions that extend NG=3 fine cells (= 1.5 coarse cells) past the
    child's interior region.  Stencil cells that fall outside the parent block
    are filled by edge replication inside `_interp_to_fine_1d` — boundary cells
    of the halo may carry small interpolation error from this, but they get
    overwritten anyway by same-level neighbor sync if other fine blocks exist.

    For Phase 1 (single refined block per parent), this gives valid halo data
    on all 4 faces of the fine block.
    """
    cx, cy = child_corner

    # A1: slice the parent down to this child's coarse footprint window before
    # refining (see _prolong_window).  c0 anchors the window to the corner-side
    # parent edge; refining it gives the same fine values as refining the whole
    # parent, for every cell in the child window.
    c0x = cx * _PROLONG_OFF
    c0y = cy * _PROLONG_OFF
    sub = jax.lax.dynamic_slice(
        parent_block, (0, c0x, c0y), (NF, _PROLONG_W, _PROLONG_W))

    refined_x  = _interp_to_fine_1d(sub,        axis=1)
    refined_xy = _interp_to_fine_1d(refined_x,  axis=2)

    # Child window offset within the (2*_PROLONG_W)² sub-refined grid.  In the
    # full-parent grid the child starts at fine NG + cx*BS; subtract the window's
    # fine origin (2*c0) ⇒ NG + cx*(BS - 2*_PROLONG_OFF).
    start_x = NG + cx * _PROLONG_SUBSTEP
    start_y = NG + cy * _PROLONG_SUBSTEP

    return jax.lax.dynamic_slice(
        refined_xy,
        (0, start_x, start_y),
        (NF, BS + 2 * NG, BS + 2 * NG),
    )


# ── Ghost-zone sync at the root level (periodic BC) ──────────────────────────

@partial(jax.jit, static_argnames=('nbx', 'nby'))
def sync_ghosts_within_level_root_periodic(
    blocks_level0: jnp.ndarray,     # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)
    nbx: int,                       # static — root tiling along x
    nby: int,                       # static — root tiling along y
) -> jnp.ndarray:
    """Fill halo cells of every root-level block by stitching, wrap-padding, and
    re-extracting.  Assumes a regular nbx × nby root tiling (slot = bi*nby + bj)
    occupies the first nbx*nby slots of blocks_level0.  Periodic BC on the global
    domain.  Other slots are left untouched.

    Implementation: gather interiors → (NF, nbx*BS, nby*BS) global grid → pad
    with `mode='wrap'` → dynamic_slice back into per-block tiles with NG halo.
    """
    n_root = nbx * nby

    # Gather interiors from the first n_root slots.  Static slice [0:n_root] is
    # fine since the index is a Python int.
    interiors = blocks_level0[:n_root, :, NG:NG + BS, NG:NG + BS]
    # interiors: (n_root, NF, BS, BS)
    interiors = interiors.reshape(nbx, nby, NF, BS, BS)
    # Reorder & flatten into the global interior grid (NF, nbx*BS, nby*BS).
    # axis order: (NF, bi, bs_x, bj, bs_y) → (NF, bi*bs_x, bj*bs_y).
    global_int = jnp.transpose(interiors, (2, 0, 3, 1, 4))
    global_int = global_int.reshape(NF, nbx * BS, nby * BS)

    # Periodic-wrap padding by NG cells.
    global_pad = jnp.pad(global_int, ((0, 0), (NG, NG), (NG, NG)), mode='wrap')
    # global_pad: (NF, nbx*BS + 2*NG, nby*BS + 2*NG)

    # Re-extract each tile of shape (NF, BS+2*NG, BS+2*NG) starting at (bi*BS, bj*BS).
    bi_idx = jnp.arange(nbx)
    bj_idx = jnp.arange(nby)

    def extract(bi, bj):
        return jax.lax.dynamic_slice(
            global_pad,
            (0, bi * BS, bj * BS),
            (NF, BS + 2 * NG, BS + 2 * NG),
        )

    # vmap over the Cartesian product.
    extract_row = jax.vmap(extract, in_axes=(None, 0))   # (nby, NF, ...)
    extract_grid = jax.vmap(extract_row, in_axes=(0, None))  # (nbx, nby, NF, ...)
    tiles = extract_grid(bi_idx, bj_idx)                 # (nbx, nby, NF, BS+2*NG, BS+2*NG)
    tiles = tiles.reshape(n_root, NF, BS + 2 * NG, BS + 2 * NG)

    # Write the n_root tiles back into blocks_level0[:n_root].
    return blocks_level0.at[:n_root].set(tiles)


# ── Cross-level halo fill (parent interpolated into child's halo) ─────────────

def _prolongate_dynamic(parent_block: jnp.ndarray, cx, cy) -> jnp.ndarray:
    """Same as `prolongate` but with traced (cx, cy) — needed for vmap over slots.
    Uses the A1 footprint-only window (see `prolongate`)."""
    cx32 = cx.astype(jnp.int32)
    cy32 = cy.astype(jnp.int32)

    c0x = cx32 * jnp.int32(_PROLONG_OFF)
    c0y = cy32 * jnp.int32(_PROLONG_OFF)
    sub = jax.lax.dynamic_slice(
        parent_block, (jnp.int32(0), c0x, c0y), (NF, _PROLONG_W, _PROLONG_W))

    refined_x  = _interp_to_fine_1d(sub,        axis=1)
    refined_xy = _interp_to_fine_1d(refined_x,  axis=2)

    start_x = jnp.int32(NG) + cx32 * jnp.int32(_PROLONG_SUBSTEP)
    start_y = jnp.int32(NG) + cy32 * jnp.int32(_PROLONG_SUBSTEP)
    return jax.lax.dynamic_slice(
        refined_xy,
        (jnp.int32(0), start_x, start_y),
        (NF, BS + 2 * NG, BS + 2 * NG),
    )


@jax.jit
def sync_ghosts_across_levels(
    child_blocks:  jnp.ndarray,    # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG) — fine level
    parent_blocks: jnp.ndarray,    # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG) — coarse level
    parent_slot:   jnp.ndarray,    # (MAX_BLOCKS,) int32 — parent slot for each child
    child_cx:      jnp.ndarray,    # (MAX_BLOCKS,) int32 — corner index ∈ {0,1}
    child_cy:      jnp.ndarray,    # (MAX_BLOCKS,) int32
    active:        jnp.ndarray,    # (MAX_BLOCKS,) bool — whether slot is active
) -> jnp.ndarray:
    """For each active child block, fill its HALO cells from a prolongation of
    its parent block.  Interior cells are preserved unchanged.  Inactive slots
    are returned unchanged.

    `parent_slot[i]` should be a valid index into `parent_blocks` whenever
    `active[i]` is True; for inactive slots the value is ignored (filled by
    convention with 0 so the gather doesn't fault).
    """
    parents_for_children = parent_blocks[parent_slot]
    prolongated = jax.vmap(_prolongate_dynamic)(
        parents_for_children, child_cx, child_cy
    )

    # A2 (4.5): take the prolongated value only where BOTH the cell is a halo
    # cell AND the slot is active; everything else keeps child_blocks.  This is a
    # single fused `select` instead of the previous two (halo-vs-interior, then
    # active-vs-inactive) — bit-identical, and `select` is the top GPU bucket.
    halo_mask = jnp.ones((BS + 2 * NG, BS + 2 * NG), dtype=bool)
    halo_mask = halo_mask.at[NG:NG + BS, NG:NG + BS].set(False)
    write_mask = active[:, None, None, None] & halo_mask[None, None]
    return jnp.where(write_mask, prolongated, child_blocks)


def prolong_all(parent_blocks, parent_slot, child_cx, child_cy):
    """Raw prolongation of each child's parent block — full child blocks
    (interior+halo), NO masking.  The active/halo masking is applied separately by
    `apply_prolonged_halo`, so for M1a this raw prolong can be precomputed once per
    `advance` and Hermite-combined per RK substep without paying a redundant `where`
    on every bracket.  (This is `sync_ghosts_across_levels` minus its masking tail.)
    """
    parents = parent_blocks[parent_slot]
    return jax.vmap(_prolongate_dynamic)(parents, child_cx, child_cy)


def apply_prolonged_halo(child_blocks, prolonged_halo, active):
    """Write a PRE-PROLONGED halo into each active child's halo ring, keeping
    interiors.  This is the masking tail of `sync_ghosts_across_levels`, split out
    so the (expensive) prolongation can be done ONCE per `advance` and the per-RK-
    substep halos formed by a cheap Hermite/linear combine (M1a — prolongation is a
    linear operator, so `prolong(hermite(brackets, s)) == hermite(prolong(brackets), s)`).

    `prolonged_halo` is a child-shaped block whose halo holds the prolonged values
    (and whose interior is 0 — it comes from prolongating into a zero child), so for
    active slots the halo is overwritten and the interior kept; inactive slots are
    unchanged.  Equivalent (to machine precision) to `sync_ghosts_across_levels`.
    """
    halo_mask = jnp.ones((BS + 2 * NG, BS + 2 * NG), dtype=bool)
    halo_mask = halo_mask.at[NG:NG + BS, NG:NG + BS].set(False)
    write_mask = active[:, None, None, None] & halo_mask[None, None]
    return jnp.where(write_mask, prolonged_halo, child_blocks)


# ── Within-level ghost sync between adjacent same-level blocks (non-root) ──────
#
# Two same-level blocks that share a face hold REAL fine data on both sides, so
# the shared-face halo should be copied from the neighbour's interior (exact),
# not prolongated from the coarse parent (approximate).  This kernel does that
# copy, per face, for every block that has a same-level neighbour on that face.
#
# Faces are indexed 0..3 = (-x, +x, -y, +y), matching AMRTopology.neighbors.
# Per face we copy the neighbour's NG-deep edge interior strip into this block's
# NG-deep face halo, over the INTERIOR extent only — the corner halos (where both
# axes are in the halo) are never read by the separable 6th-order x/y stencils
# (verified in fused_rhs_fp `sx`/`sy`: each reads halo cells on its own axis
# but interior cells on the other), so face strips are sufficient and complete.
#
# Geometry (block = (NF, BS+2NG, BS+2NG); interior = [NG:NG+BS, NG:NG+BS]):
#   -x: my halo rows [0:NG]            ← neighbour interior rows [BS:BS+NG]
#   +x: my halo rows [NG+BS:2NG+BS]    ← neighbour interior rows [NG:2NG]
#   -y: my halo cols [0:NG]            ← neighbour interior cols [BS:BS+NG]
#   +y: my halo cols [NG+BS:2NG+BS]    ← neighbour interior cols [NG:2NG]

# (dst_row_lo, dst_row_hi, dst_col_lo, dst_col_hi, src_row_lo, src_row_hi, src_col_lo, src_col_hi)
_WL_FACE_SPECS = (
    (0,      NG,        NG, NG + BS,  BS,     BS + NG,   NG, NG + BS),   # -x
    (NG+BS,  2*NG+BS,   NG, NG + BS,  NG,     2*NG,      NG, NG + BS),   # +x
    (NG,     NG + BS,   0,  NG,       NG, NG + BS,       BS, BS + NG),   # -y
    (NG,     NG + BS,   NG+BS, 2*NG+BS, NG, NG + BS,     NG, 2*NG),      # +y
)


@jax.jit
def sync_ghosts_within_level(
    blocks:         jnp.ndarray,   # (mb, NF, BS+2*NG, BS+2*NG)
    neighbor_slot:  jnp.ndarray,   # (mb, 4) int32 — same-level neighbour slot per face
    neighbor_valid: jnp.ndarray,   # (mb, 4) bool  — True where that face has a neighbour
) -> jnp.ndarray:
    """Fill each block's FACE halos from its same-level neighbours' interiors.

    Faces with no same-level neighbour (`neighbor_valid` False — patch boundary)
    are left unchanged, so a preceding `sync_ghosts_across_levels` (parent
    prolongation) still provides those.  `neighbor_slot` for invalid faces is
    ignored (gathers a filler block that the `where` discards).

    Corner halos are intentionally NOT written — the separable stencils never
    read them.  Shape-stable: all slice extents are static; only the gather index
    and the validity flag are data-dependent.
    """
    out = blocks
    for f, (drl, drh, dcl, dch, srl, srh, scl, sch) in enumerate(_WL_FACE_SPECS):
        nb = blocks[neighbor_slot[:, f]]                     # (mb, NF, H, W) gather
        src = nb[:, :, srl:srh, scl:sch]                     # neighbour edge strip
        cur = out[:, :, drl:drh, dcl:dch]
        valid = neighbor_valid[:, f][:, None, None, None]
        out = out.at[:, :, drl:drh, dcl:dch].set(jnp.where(valid, src, cur))
    return out


# ── Restriction: 1 child interior → 1 quadrant of parent interior ─────────────

@partial(jax.jit, static_argnames=('child_corner',))
def restrict_into_parent(
    parent_block:   jnp.ndarray,           # (NF, BS+2*NG, BS+2*NG)
    child_interior: jnp.ndarray,           # (NF, BS, BS)
    child_corner:   tuple,                 # (cx, cy) ∈ {0,1}²
) -> jnp.ndarray:
    """Restrict one child's interior into the corresponding quadrant of parent.

    Each parent cell ← mean of the 4 child cells it covers.  Returns a NEW
    parent_block (immutable) with the restricted region updated.

    Repeating for all 4 children fully restricts the parent's interior.
    """
    cx, cy = child_corner
    half_bs = BS // REFINE_RATIO

    # Reshape child (NF, BS, BS) → (NF, half_bs, 2, half_bs, 2)
    # and average the 2-axes to get (NF, half_bs, half_bs).
    averaged = child_interior.reshape(
        NF, half_bs, REFINE_RATIO, half_bs, REFINE_RATIO
    ).mean(axis=(2, 4))

    start_x = NG + cx * half_bs
    start_y = NG + cy * half_bs
    return jax.lax.dynamic_update_slice(
        parent_block,
        averaged,
        (0, start_x, start_y),
    )


# ── Multi-slot restriction: fine blocks → coarse parents ──────────────────────

def _scatter_quadrants_into_parents(coarse_blocks, quad_data,
                                    parent_slot, child_cx, child_cy, active):
    """Scatter each fine slot's restricted (NF, half_bs, half_bs) interior into its
    parent's matching quadrant — in ONE batched scatter, replacing the old per-slot
    `fori_loop` (which launched ~4 tiny kernels PER slot → thousands of latency-
    bound launches).

    The 4 children of a parent occupy DISJOINT quadrants, so a single scatter has no
    write conflicts and is bit-identical to applying the quadrants one slot at a
    time.  Inactive slots are redirected to a trash row (index n_coarse) which is
    then sliced off, so they never touch a real parent — matching the
    `where(active, …)` no-op of the sequential version.
    """
    n_coarse = coarse_blocks.shape[0]
    hb = BS // REFINE_RATIO
    cx = child_cx.astype(jnp.int32)
    cy = child_cy.astype(jnp.int32)
    ar = jnp.arange(hb, dtype=jnp.int32)
    rows = (jnp.int32(NG) + cx * jnp.int32(hb))[:, None] + ar[None, :]   # (n_fine, hb)
    cols = (jnp.int32(NG) + cy * jnp.int32(hb))[:, None] + ar[None, :]   # (n_fine, hb)
    # Inactive slots → trash row n_coarse (dropped below) so they write nothing real.
    p_safe = jnp.where(active, parent_slot.astype(jnp.int32), jnp.int32(n_coarse))

    padded = jnp.concatenate(
        [coarse_blocks,
         jnp.zeros((1,) + coarse_blocks.shape[1:], coarse_blocks.dtype)], axis=0)
    # Broadcast the per-(slot,field,row,col) target indices and scatter in one op.
    p_idx = p_safe[:, None, None, None]               # (n_fine,1,1,1)
    f_idx = jnp.arange(NF)[None, :, None, None]        # (1,NF,1,1)
    r_idx = rows[:, None, :, None]                     # (n_fine,1,hb,1)
    c_idx = cols[:, None, None, :]                     # (n_fine,1,1,hb)
    padded = padded.at[p_idx, f_idx, r_idx, c_idx].set(quad_data)
    return padded[:n_coarse]


@jax.jit
def restrict_all_into_parents(
    coarse_blocks: jnp.ndarray,    # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)
    fine_blocks:   jnp.ndarray,    # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG) — fine level
    parent_slot:   jnp.ndarray,    # (MAX_BLOCKS,) int32 — parent slot for each fine slot
    child_cx:      jnp.ndarray,    # (MAX_BLOCKS,) int32
    child_cy:      jnp.ndarray,    # (MAX_BLOCKS,) int32
    active:        jnp.ndarray,    # (MAX_BLOCKS,) bool — whether fine slot is active
) -> jnp.ndarray:
    """For each active fine slot, 2×2-average its interior into the matching
    quadrant of its coarse parent.  Returns the updated coarse_blocks; only the
    written quadrants change.

    Multiple fine slots writing to the same coarse parent occupy disjoint
    quadrants, so a single batched scatter (`_scatter_quadrants_into_parents`)
    composes them into a full parent restriction with no write conflicts.
    """
    half_bs = BS // REFINE_RATIO
    n_fine = fine_blocks.shape[0]   # fine-level slot capacity (static)

    # Pre-compute the 2×2 averages of every fine slot's interior.
    fine_int = fine_blocks[:, :, NG:NG + BS, NG:NG + BS]
    averaged_all = fine_int.reshape(
        n_fine, NF, half_bs, REFINE_RATIO, half_bs, REFINE_RATIO
    ).mean(axis=(3, 5))   # (n_fine, NF, half_bs, half_bs)

    return _scatter_quadrants_into_parents(
        coarse_blocks, averaged_all, parent_slot, child_cx, child_cy, active)


# ── High-order (6th) restriction ──────────────────────────────────────────────

@partial(jax.jit, static_argnames=('child_corner',))
def restrict_into_parent_highorder(
    parent_block: jnp.ndarray,             # (NF, BS+2*NG, BS+2*NG)
    child_full:   jnp.ndarray,             # (NF, BS+2*NG, BS+2*NG) — FULL block (halo needed)
    child_corner: tuple,                   # (cx, cy) ∈ {0,1}²
) -> jnp.ndarray:
    """6th-order restriction of one child into the matching parent quadrant.

    Unlike `restrict_into_parent` (2×2 averaging, 2nd-order), this evaluates the
    fine field at each coarse-cell centre with a 6-point Lagrange stencil, so it
    no longer caps the coarse-fine interface at 2nd order.  Requires the FULL
    child block — the stencil reaches 2 fine cells into the halo at the quadrant
    edges.

    NOTE: not exactly mean-conservative (a high-order pointwise transfer trades
    exact-average for accuracy).  Correct choice for a finite-DIFFERENCE scheme
    (we track point values); a finite-volume scheme would want a conservative
    high-order variant instead.
    """
    cx, cy = child_corner
    half_bs = BS // REFINE_RATIO
    r = _restrict_to_coarse_1d(child_full, axis=1)   # (NF, half_bs, BS+2*NG)
    r = _restrict_to_coarse_1d(r,          axis=2)   # (NF, half_bs, half_bs)
    start_x = NG + cx * half_bs
    start_y = NG + cy * half_bs
    return jax.lax.dynamic_update_slice(parent_block, r, (0, start_x, start_y))


@jax.jit
def restrict_all_into_parents_highorder(
    coarse_blocks: jnp.ndarray,    # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)
    fine_blocks:   jnp.ndarray,    # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG) — FULL fine blocks
    parent_slot:   jnp.ndarray,    # (MAX_BLOCKS,) int32
    child_cx:      jnp.ndarray,    # (MAX_BLOCKS,) int32
    child_cy:      jnp.ndarray,    # (MAX_BLOCKS,) int32
    active:        jnp.ndarray,    # (MAX_BLOCKS,) bool
) -> jnp.ndarray:
    """Multi-slot 6th-order restriction (the high-order counterpart of
    `restrict_all_into_parents`).  Uses the full fine blocks (halo included)."""
    # 6th-order restrict every fine slot (vectorised over slots): axes 2 and 3.
    restricted_all = _restrict_to_coarse_1d(
        _restrict_to_coarse_1d(fine_blocks, axis=2), axis=3
    )   # (n_fine, NF, half_bs, half_bs)

    return _scatter_quadrants_into_parents(
        coarse_blocks, restricted_all, parent_slot, child_cx, child_cy, active)


# ── Refinement indicator: per-block max |∇field| ──────────────────────────────

@partial(jax.jit, static_argnames=('field_idx',))
def compute_indicator_gradient(
    blocks_one_level: jnp.ndarray,    # (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)
    dx: float,                        # cell size at this level
    field_idx: int = 2,               # default: EZ (field index 2)
) -> jnp.ndarray:
    """Per-block max |∇field| in physical units (returns shape (MAX_BLOCKS,)).

    Uses simple 2-point central differences on the interior cells (reads into
    the halo for the stencil).  Returns gradient magnitude:
        max over interior of sqrt((du/dx)² + (du/dy)²)
    measured in [field-units / length-units].

    Inactive blocks (zero halo + zero interior) return 0 — the caller is
    responsible for masking against the active mask before using this as a
    refinement criterion.
    """
    u = blocks_one_level[:, field_idx]   # (MAX_BLOCKS, BS+2*NG, BS+2*NG)
    # Centered 2-point differences over the interior.  At the interior cell
    # (NG+i, NG+j), the stencil reads (NG+i±1, NG+j) etc., which still falls
    # within the (BS+2*NG)-wide block as long as NG >= 1.
    inv_2dx = jnp.asarray(1.0 / (2.0 * dx))
    dudx = (u[:, NG+1:NG+1+BS, NG:NG+BS] - u[:, NG-1:NG-1+BS, NG:NG+BS]) * inv_2dx
    dudy = (u[:, NG:NG+BS, NG+1:NG+1+BS] - u[:, NG:NG+BS, NG-1:NG-1+BS]) * inv_2dx
    grad_mag = jnp.sqrt(dudx ** 2 + dudy ** 2)
    return grad_mag.max(axis=(-2, -1))   # (MAX_BLOCKS,)
