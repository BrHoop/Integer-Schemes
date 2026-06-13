"""
Block-structured AMR for MCS 2D — data structures and Python-side bookkeeping.

The architecture is in AMR_PLAN.md.  Quick summary:

  * JAX side: `AMRState` is a pytree with FIXED shapes (LEVELS, MAX_BLOCKS, ...).
    Refinement creates/destroys blocks by flipping bits in an `active` mask,
    not by resizing arrays.  No mid-evolution recompilation.

  * Python side: `AMRTopology` carries parent/child links, neighbor maps,
    bbox info, hysteresis counters — all small, all on host, all NumPy dict/list.

  * Per-block kernels (prolongation, restriction, ghost-zone sync, RHS advance)
    are jitted in `amr_kernels.py`.  They take fixed-shape inputs and never
    recompile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


# ── AMR shape constants (set per run; default for the 2D MCS prototype) ────────

LEVELS     = int(os.environ.get("MCS_AMR_LEVELS",     4))
MAX_BLOCKS = int(os.environ.get("MCS_AMR_MAX_BLOCKS", 64))
BS         = int(os.environ.get("MCS_AMR_BS",         32))   # matches fused_*
NG         = 3                                                # 6th-order halo
NF         = 10                                               # MCS field count


# Per-level slot capacity.  AMR storage is RAGGED: level L holds MAX_BLOCKS_PER_LEVEL[L]
# slots, so memory and compute scale with what each level can actually hold rather
# than a single global MAX_BLOCKS.  Each entry is a compile-time constant, so the
# arrays stay shape-stable (no recompile on regrid).
#
# Phase 1 default is UNIFORM (every level = MAX_BLOCKS) so behaviour is bit-identical
# to the old single-stacked-array layout; Phase 2 shrinks these to per-level sizes.
# Override with MCS_AMR_MAX_BLOCKS_PER_LEVEL="16,32,64,64" (comma-separated, length LEVELS).
def _parse_mb_per_level() -> tuple:
    raw = os.environ.get("MCS_AMR_MAX_BLOCKS_PER_LEVEL")
    if raw is None:
        return (MAX_BLOCKS,) * LEVELS
    vals = tuple(int(x) for x in raw.split(","))
    if len(vals) != LEVELS:
        raise ValueError(
            f"MCS_AMR_MAX_BLOCKS_PER_LEVEL has {len(vals)} entries, expected LEVELS={LEVELS}"
        )
    return vals

MAX_BLOCKS_PER_LEVEL = _parse_mb_per_level()

# Host-side bookkeeping arrays are padded to this width (≥ every per-level cap);
# phantom columns [mb(L), MAX_BLOCKS) are never allocated.
assert max(MAX_BLOCKS_PER_LEVEL) <= MAX_BLOCKS, (
    f"MAX_BLOCKS_PER_LEVEL {MAX_BLOCKS_PER_LEVEL} exceeds MAX_BLOCKS={MAX_BLOCKS}; "
    f"raise MCS_AMR_MAX_BLOCKS."
)


def mb(level: int) -> int:
    """Slot capacity at `level`."""
    return MAX_BLOCKS_PER_LEVEL[level]


# Refinement ratio (per axis).  Standard Berger-Oliger: 2.  Children are 2× finer.
REFINE_RATIO = 2


# ── AMRState — the JAX pytree ─────────────────────────────────────────────────

@register_pytree_node_class
class AMRState:
    """Block storage + active mask — RAGGED per-level, fixed shapes for the run.

    Attributes:
      blocks: tuple of LEVELS arrays; blocks[L] has shape
              (MAX_BLOCKS_PER_LEVEL[L], NF, BS+2*NG, BS+2*NG) float64.
              Each (NF, BS+2*NG, BS+2*NG) slice is one block (interior + halo).
      active: tuple of LEVELS bool arrays; active[L] has shape
              (MAX_BLOCKS_PER_LEVEL[L],).  True where a slot holds a real block.

    Each per-level shape is a compile-time constant, so the layout is still
    fully shape-stable — refinement flips `active` bits and writes into slots,
    never resizes.  (A tuple of differently-shaped arrays is a valid JAX pytree:
    every array is a leaf, traced at its own shape.)
    """

    def __init__(self, blocks: tuple, active: tuple):
        # Accept any sequence; normalise to tuple so the pytree structure is stable.
        self.blocks = tuple(blocks)
        self.active = tuple(active)

    @classmethod
    def empty(cls, dtype=jnp.float64, caps=None) -> "AMRState":
        """An AMRState with everything zero / inactive.  `caps` is the per-level
        capacity tuple (defaults to the global MAX_BLOCKS_PER_LEVEL)."""
        if caps is None:
            caps = MAX_BLOCKS_PER_LEVEL
        return cls(
            blocks=tuple(
                jnp.zeros((caps[L], NF, BS + 2*NG, BS + 2*NG), dtype) for L in range(LEVELS)
            ),
            active=tuple(jnp.zeros((caps[L],), bool) for L in range(LEVELS)),
        )

    # Pytree interface ────────────────────────────────────────────────────────
    # children are the two per-level tuples; JAX flattens them recursively into
    # the individual per-level arrays (leaves) and rebuilds the tuples on unflatten.
    def tree_flatten(self):
        return ((self.blocks, self.active), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    def __repr__(self):
        n_active_per_level = [int(self.active[L].sum()) for L in range(LEVELS)]
        return (f"AMRState(LEVELS={LEVELS}, MAX_BLOCKS_PER_LEVEL={MAX_BLOCKS_PER_LEVEL}, "
                f"BS={BS}, active_per_level={n_active_per_level})")


# ── AMRTopology — host-side bookkeeping ────────────────────────────────────────

@dataclass
class AMRTopology:
    """Python-side bookkeeping for the AMR hierarchy.

    Mirror of the `active` mask plus the relationships JAX never sees
    (parent/child links, neighbor maps, bbox info).  All small NumPy / dict —
    cheap to manipulate, easy to debug, never causes a JAX recompile.

    bbox_ijk: each block has integer corner coords in the LEVEL-L cell grid.
              i.e., bbox_ijk[(L, i)] = (i0, j0) means the block's interior
              starts at coarse-cell coords (i0, j0) at level L.
              Used to compute physical positions and to derive child/parent
              bboxes during refinement.
    """

    # Mirror of AMRState.active for host queries — keep in sync.
    active:    np.ndarray = field(default_factory=lambda: np.zeros((LEVELS, MAX_BLOCKS), bool))

    # parent[(level, idx)] = (level-1, parent_idx) or None for root-level blocks
    parent:    dict = field(default_factory=dict)

    # children[(level, idx)] = list of (level+1, child_idx) for refined parents
    children:  dict = field(default_factory=dict)

    # bbox_ijk[(level, idx)] = (i0, j0) integer corner in level-L cell units
    bbox_ijk:  dict = field(default_factory=dict)

    # neighbors[(level, idx, face)] = (level, neighbor_idx) or None
    # face = 0..3 for (-x, +x, -y, +y) in 2D
    neighbors: dict = field(default_factory=dict)

    # Per-block streak counter for refinement hysteresis (positive = consecutive
    # cycles flagged; negative = consecutive cycles unflagged).
    streaks:   np.ndarray = field(default_factory=lambda: np.zeros((LEVELS, MAX_BLOCKS), np.int32))

    # Per-level slot CAPACITY — the live, mutable bound on how many blocks each
    # level can hold (the JAX arrays are sized to this).  Seeded from the global
    # MAX_BLOCKS_PER_LEVEL default; the auto-grow calibration may raise entries
    # at runtime (up to the hard ceiling MAX_BLOCKS).
    caps:      list = field(default_factory=lambda: list(MAX_BLOCKS_PER_LEVEL))

    # Per-level peak simultaneous occupancy seen so far (calibration data).
    peak_active: list = field(default_factory=lambda: [0] * LEVELS)

    # Log of auto-grow events: list of (level, old_cap, new_cap).  Each entry is
    # one capacity bump (and one step recompile) — a signal the default cap for
    # that level was too low.
    grow_history: list = field(default_factory=list)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def n_active(self, level: int) -> int:
        return int(self.active[level].sum())

    def record_occupancy(self):
        """Update peak simultaneous occupancy per level (calibration data)."""
        for L in range(LEVELS):
            self.peak_active[L] = max(self.peak_active[L], self.n_active(L))

    def recommended_caps(self, margin: float = 1.5) -> tuple:
        """Suggested MAX_BLOCKS_PER_LEVEL from observed peak occupancy × margin
        (level 0 is exact — the fixed root tiling).  Clamped to the hard ceiling."""
        rec = []
        for L in range(LEVELS):
            want = self.peak_active[L] if L == 0 else int(np.ceil(self.peak_active[L] * margin))
            rec.append(min(max(want, 1), MAX_BLOCKS))
        return tuple(rec)

    def find_empty_slot(self, level: int) -> int:
        """Return the first inactive slot index at `level`, within that level's
        current capacity caps[level].  Raises if the level's budget is exhausted."""
        cap = self.caps[level]
        empty = np.flatnonzero(~self.active[level, :cap])
        if len(empty) == 0:
            raise RuntimeError(
                f"AMR budget exhausted at level {level}: all {cap} slots in use. "
                f"Increase MCS_AMR_MAX_BLOCKS_PER_LEVEL[{level}]."
            )
        return int(empty[0])

    def add_block(self, level: int, slot: int, bbox_ij: tuple[int, int],
                  parent: Optional[tuple[int, int]] = None):
        """Mark a slot active and record its bookkeeping.  Does NOT touch JAX data;
        caller is responsible for writing into AMRState.blocks at this slot."""
        self.active[level, slot] = True
        self.bbox_ijk[(level, slot)] = bbox_ij
        if parent is not None:
            self.parent[(level, slot)] = parent
            self.children.setdefault(parent, []).append((level, slot))
        # neighbors[] will be filled by rebuild_neighbors().

    def remove_block(self, level: int, slot: int):
        """Deactivate a block and drop its bookkeeping.  Caller may want to
        delete from self.children[parent] too."""
        key = (level, slot)
        self.active[level, slot] = False
        self.parent.pop(key, None)
        self.children.pop(key, None)
        self.bbox_ijk.pop(key, None)
        # Strip from neighbor map (clean up dangling references).
        self.neighbors = {
            k: v for k, v in self.neighbors.items()
            if k[:2] != key and v != key
        }

    def rebuild_neighbors(self):
        """Recompute same-level face-neighbour links from `bbox_ijk` adjacency.

        Two same-level blocks are face neighbours when their interiors abut in the
        level-L cell grid — i.e. their corners differ by exactly BS along one axis
        and 0 along the other.  Face index 0..3 = (-x, +x, -y, +y).

        Fine levels are NOT periodic: only actual adjacencies are recorded; a face
        with no same-level block is set to None (its halo is filled by cross-level
        prolongation from the parent instead).  Sets `neighbors[(L, s, f)]`.
        """
        self.neighbors = {}
        for L in range(LEVELS):
            pos = {}                       # (i0, j0) → slot, for active blocks at L
            for s in range(self.caps[L]):
                if self.active[L, s]:
                    pos[self.bbox_ijk[(L, s)]] = s
            for (i0, j0), s in pos.items():
                face_pos = {0: (i0 - BS, j0), 1: (i0 + BS, j0),
                            2: (i0, j0 - BS), 3: (i0, j0 + BS)}
                for f, p in face_pos.items():
                    nb = pos.get(p)
                    self.neighbors[(L, s, f)] = (L, nb) if nb is not None else None

    def check_proper_nesting(self) -> list:
        """Return a list of proper-nesting violations (empty ⇒ properly nested).

        Berger-Oliger proper nesting (as we need it): every active block at level
        L≥1 must have an ACTIVE parent at L-1 whose interior fully contains the
        child's footprint.  That guarantees the child's interior data and its
        cross-level (prolongation) halo are backed by valid coarse data from a
        single parent.  Each violation is `(level, slot, reason)`.
        """
        violations = []
        half = BS // REFINE_RATIO          # child footprint in parent cells
        for L in range(1, LEVELS):
            for s in range(self.caps[L]):
                if not self.active[L, s]:
                    continue
                par = self.parent.get((L, s))
                if par is None:
                    violations.append((L, s, "no parent"))
                    continue
                pL, ps = par
                if not self.active[pL, ps]:
                    violations.append((L, s, f"parent {par} inactive"))
                    continue
                cb = self.bbox_ijk[(L, s)]
                pb = self.bbox_ijk[(pL, ps)]
                fx0, fy0 = cb[0] // REFINE_RATIO, cb[1] // REFINE_RATIO
                if not (pb[0] <= fx0 and fx0 + half <= pb[0] + BS and
                        pb[1] <= fy0 and fy0 + half <= pb[1] + BS):
                    violations.append(
                        (L, s, f"footprint ({fx0},{fy0})+{half} not inside "
                               f"parent interior {pb}+{BS}"))
        return violations

    # ── JAX-side snapshot ─────────────────────────────────────────────────────

    def to_jax_arrays(self) -> "AMRTopologyArrays":
        """Snapshot the host-side topology into per-level JAX arrays suitable
        for passing into the step functions.  Inactive slots get filler values
        (parent_slot=0, child_cx/cy=0) — the active mask in AMRState gates
        whether they're consumed.
        """
        parent_slot = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        child_cx    = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        child_cy    = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        half_bs = BS // REFINE_RATIO
        for (L, s), (pL, ps) in self.parent.items():
            if not self.active[L, s]:
                continue
            parent_slot[L, s] = ps
            cb = self.bbox_ijk[(L, s)]
            pb = self.bbox_ijk[(pL, ps)]
            rel_x = (cb[0] // REFINE_RATIO) - pb[0]
            rel_y = (cb[1] // REFINE_RATIO) - pb[1]
            child_cx[L, s] = rel_x // half_bs
            child_cy[L, s] = rel_y // half_bs

        # Same-level face-neighbour links for within-level ghost sync (faces
        # 0..3 = -x,+x,-y,+y).  Invalid faces keep slot 0 (a filler the kernel's
        # validity mask discards).
        neighbor_slot  = np.zeros((LEVELS, MAX_BLOCKS, 4), dtype=np.int32)
        neighbor_valid = np.zeros((LEVELS, MAX_BLOCKS, 4), dtype=bool)
        self.rebuild_neighbors()
        for (L, s, f), nb in self.neighbors.items():
            if nb is not None and self.active[L, s]:
                neighbor_slot[L, s, f]  = nb[1]
                neighbor_valid[L, s, f] = True

        # Slice each level to its current capacity → ragged per-level tuples.
        return AMRTopologyArrays(
            parent_slot=tuple(jnp.asarray(parent_slot[L, :self.caps[L]]) for L in range(LEVELS)),
            child_cx   =tuple(jnp.asarray(child_cx[L, :self.caps[L]])    for L in range(LEVELS)),
            child_cy   =tuple(jnp.asarray(child_cy[L, :self.caps[L]])    for L in range(LEVELS)),
            neighbor_slot =tuple(jnp.asarray(neighbor_slot[L, :self.caps[L]])  for L in range(LEVELS)),
            neighbor_valid=tuple(jnp.asarray(neighbor_valid[L, :self.caps[L]]) for L in range(LEVELS)),
        )


# ── JAX-friendly topology bundle ──────────────────────────────────────────────

class AMRTopologyArrays(NamedTuple):
    """Snapshot of the host-side topology in a form JAX can trace over.

    Each field is a tuple of LEVELS arrays; field[L] has shape
    (MAX_BLOCKS_PER_LEVEL[L],).  Level-0 entries are unused (the root has no
    parent); entries for inactive slots pass through unread because the active
    mask (on AMRState) gates every consumer.

    Treat instances as immutable — rebuild via `AMRTopology.to_jax_arrays()`
    after each regrid event.  Per-level shapes are constant, so feeding new
    instances into a jitted step does NOT trigger recompilation.  (A NamedTuple
    of tuples-of-arrays is a valid pytree.)
    """
    parent_slot: tuple   # tuple of (mb(L),) int32
    child_cx:    tuple    # tuple of (mb(L),) int32
    child_cy:    tuple    # tuple of (mb(L),) int32
    neighbor_slot:  tuple  # tuple of (mb(L), 4) int32 — same-level neighbour per face
    neighbor_valid: tuple  # tuple of (mb(L), 4) bool  — face has a same-level neighbour


# ── Initial-state factory ─────────────────────────────────────────────────────

def make_root_state(
    initial_data: jnp.ndarray,           # (NF, Nx_tot, Ny_tot) float64 — already padded with halo
    nbx_root:     int,                   # root tiles along x
    nby_root:     int,                   # root tiles along y
    caps=None,                           # per-level capacity (default: global MAX_BLOCKS_PER_LEVEL)
) -> tuple[AMRState, AMRTopology]:
    """Build an AMR state at level 0 only, tiling the input data into root-level blocks.

    The input `initial_data` is the SAME shape as the existing scheme's state
    (Nx_tot, Ny_tot include the NG halo on each side).  We chop it into
    nbx_root × nby_root tiles each of shape (NF, BS+2*NG, BS+2*NG).

    `caps` seeds the per-level slot capacity (e.g. loaded from a calibration
    sidecar); defaults to the global MAX_BLOCKS_PER_LEVEL.

    Returns (state, topo) with no children at any deeper level.
    """
    if caps is None:
        caps = list(MAX_BLOCKS_PER_LEVEL)
    caps = list(caps)
    Nx_tot, Ny_tot = initial_data.shape[1], initial_data.shape[2]
    expected_nx = nbx_root * BS + 2 * NG
    expected_ny = nby_root * BS + 2 * NG
    assert Nx_tot == expected_nx, f"input Nx={Nx_tot} ≠ nbx_root*BS + 2*NG = {expected_nx}"
    assert Ny_tot == expected_ny, f"input Ny={Ny_tot} ≠ nby_root*BS + 2*NG = {expected_ny}"

    if nbx_root * nby_root > caps[0]:
        raise ValueError(
            f"Root grid {nbx_root}×{nby_root} = {nbx_root*nby_root} blocks "
            f"exceeds level-0 capacity caps[0]={caps[0]}.  Increase "
            f"MCS_AMR_MAX_BLOCKS_PER_LEVEL[0]."
        )

    topo = AMRTopology(caps=caps)

    # Level-0 block storage (only the root level is populated here).
    blocks_l0 = np.zeros((caps[0], NF, BS + 2*NG, BS + 2*NG), dtype=np.float64)
    for bi in range(nbx_root):
        for bj in range(nby_root):
            slot = bi * nby_root + bj
            i0 = bi * BS
            j0 = bj * BS
            blocks_l0[slot] = np.asarray(
                initial_data[:, i0 : i0 + BS + 2*NG, j0 : j0 + BS + 2*NG]
            )
            topo.add_block(level=0, slot=slot, bbox_ij=(i0, j0), parent=None)

    # Per-level block tuple: level 0 filled, deeper levels empty.
    blocks = tuple(
        jnp.asarray(blocks_l0) if L == 0
        else jnp.zeros((caps[L], NF, BS + 2*NG, BS + 2*NG))
        for L in range(LEVELS)
    )
    active = tuple(jnp.asarray(topo.active[L, :caps[L]]) for L in range(LEVELS))
    state = AMRState(blocks=blocks, active=active)
    return state, topo
