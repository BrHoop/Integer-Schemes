"""
Block-structured AMR for the 3D Llama multipatch grid — data structures.

A 3D, node-centered port of ``mcs2d/amr/state.py``.  The design is unchanged:

  * JAX side: ``AMRState`` is a pytree with FIXED shapes (ragged per-level
    ``(slots, NF, BS+2NG, BS+2NG, BS+2NG)`` block arrays + an ``active`` mask).
    Refinement flips ``active`` bits and writes into slots — never resizes, never
    recompiles.

  * Python side: ``AMRTopology`` carries parent/child links, 6-face neighbour
    maps, integer bbox corners, hysteresis streaks — all small NumPy/dict on host.

Node-centered convention (vs the 2D cell-centered original): a block owns ``BS``
*disjoint* nodes of the patch's logical node grid (no duplicated shared vertex —
the seam vertex is reached through the ``NG`` ghost layer).  Block *tiling* is
therefore identical to the 2D layout (disjoint chunks + halo); only the
prolong/restrict stencils differ (see ``kernels.py``).  Refinement is 2:1 in the
patch's logical coordinates; a child octant covers ``BS//2`` of its parent's
nodes and refines them to ``BS`` fine nodes (copy-at-coincident + midpoint).

Faces: 0..5 = (-x, +x, -y, +y, -z, +z).  Children: 8 octants (cx, cy, cz) ∈ {0,1}³.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


# ── AMR shape constants (env-overridable, like the 2D module) ──────────────────

LEVELS     = int(os.environ.get("MP_AMR_LEVELS",     3))
MAX_BLOCKS = int(os.environ.get("MP_AMR_MAX_BLOCKS", 64))
BS         = int(os.environ.get("MP_AMR_BS",         8))    # 3D: small (memory)
NG         = int(os.environ.get("MP_AMR_NG",         3))    # 6th-order halo
NF         = int(os.environ.get("MP_AMR_NF",         5))    # wave=5, MCS=10

REFINE_RATIO = 2
NFACE = 6


def _parse_mb_per_level() -> tuple:
    raw = os.environ.get("MP_AMR_MAX_BLOCKS_PER_LEVEL")
    if raw is None:
        return (MAX_BLOCKS,) * LEVELS
    vals = tuple(int(x) for x in raw.split(","))
    if len(vals) != LEVELS:
        raise ValueError(
            f"MP_AMR_MAX_BLOCKS_PER_LEVEL has {len(vals)} entries, expected LEVELS={LEVELS}")
    return vals


MAX_BLOCKS_PER_LEVEL = _parse_mb_per_level()
assert max(MAX_BLOCKS_PER_LEVEL) <= MAX_BLOCKS, (
    f"MAX_BLOCKS_PER_LEVEL {MAX_BLOCKS_PER_LEVEL} exceeds MAX_BLOCKS={MAX_BLOCKS}")


def mb(level: int) -> int:
    """Slot capacity at ``level``."""
    return MAX_BLOCKS_PER_LEVEL[level]


# ── AMRState — the JAX pytree ──────────────────────────────────────────────────

@register_pytree_node_class
class AMRState:
    """Block storage + active mask — ragged per-level, fixed shapes for the run.

    Attributes:
      blocks: tuple of LEVELS arrays; blocks[L] has shape
              (caps[L], NF, BS, BS, BS) float64 — INTERIORS ONLY (no persistent
              halo). The NG-layer halo is a transient working buffer rebuilt
              from interiors each RHS substage (amr/evolve.build_haloed).
      active: tuple of LEVELS bool arrays; active[L] has shape (caps[L],).
    """

    def __init__(self, blocks: tuple, active: tuple):
        self.blocks = tuple(blocks)
        self.active = tuple(active)

    @classmethod
    def empty(cls, dtype=jnp.float64, caps=None) -> "AMRState":
        if caps is None:
            caps = MAX_BLOCKS_PER_LEVEL
        # interiors-only storage (BS^3, no persistent halo); the haloed working
        # buffer is rebuilt transiently inside the RHS (see amr/evolve.build_haloed).
        return cls(
            blocks=tuple(jnp.zeros((caps[L], NF, BS, BS, BS), dtype) for L in range(LEVELS)),
            active=tuple(jnp.zeros((caps[L],), bool) for L in range(LEVELS)),
        )

    def tree_flatten(self):
        return ((self.blocks, self.active), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    def __repr__(self):
        n = [int(self.active[L].sum()) for L in range(LEVELS)]
        return (f"AMRState(LEVELS={LEVELS}, caps={MAX_BLOCKS_PER_LEVEL}, "
                f"BS={BS}, NF={NF}, active_per_level={n})")


# ── AMRTopology — host-side bookkeeping ─────────────────────────────────────────

@dataclass
class AMRTopology:
    """Host-side mirror + the relationships JAX never sees.

    bbox_ijk[(L, s)] = (i0, j0, k0): the block's interior starts at level-L node
    coords (i0, j0, k0).  Children/parent/neighbours derived from these.
    """

    active:    np.ndarray = field(default_factory=lambda: np.zeros((LEVELS, MAX_BLOCKS), bool))
    parent:    dict = field(default_factory=dict)   # (L,s) -> (L-1, ps) or None
    children:  dict = field(default_factory=dict)   # (L,s) -> list[(L+1, cs)]
    bbox_ijk:  dict = field(default_factory=dict)   # (L,s) -> (i0,j0,k0)
    neighbors: dict = field(default_factory=dict)   # (L,s,face) -> (L, nb) or None
    streaks:   np.ndarray = field(default_factory=lambda: np.zeros((LEVELS, MAX_BLOCKS), np.int32))
    caps:      list = field(default_factory=lambda: list(MAX_BLOCKS_PER_LEVEL))
    peak_active: list = field(default_factory=lambda: [0] * LEVELS)
    grow_history: list = field(default_factory=list)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def n_active(self, level: int) -> int:
        return int(self.active[level].sum())

    def record_occupancy(self):
        for L in range(LEVELS):
            self.peak_active[L] = max(self.peak_active[L], self.n_active(L))

    def recommended_caps(self, margin: float = 1.5) -> tuple:
        rec = []
        for L in range(LEVELS):
            want = self.peak_active[L] if L == 0 else int(np.ceil(self.peak_active[L] * margin))
            rec.append(min(max(want, 1), MAX_BLOCKS))
        return tuple(rec)

    def find_empty_slot(self, level: int) -> int:
        cap = self.caps[level]
        empty = np.flatnonzero(~self.active[level, :cap])
        if len(empty) == 0:
            raise RuntimeError(
                f"AMR budget exhausted at level {level}: all {cap} slots in use. "
                f"Increase MP_AMR_MAX_BLOCKS_PER_LEVEL[{level}].")
        return int(empty[0])

    def add_block(self, level: int, slot: int, bbox_ijk: tuple,
                  parent: Optional[tuple] = None):
        """Mark a slot active + record bookkeeping.  Does NOT touch JAX data."""
        self.active[level, slot] = True
        self.bbox_ijk[(level, slot)] = bbox_ijk
        if parent is not None:
            self.parent[(level, slot)] = parent
            self.children.setdefault(parent, []).append((level, slot))

    def remove_block(self, level: int, slot: int):
        key = (level, slot)
        self.active[level, slot] = False
        self.parent.pop(key, None)
        self.children.pop(key, None)
        self.bbox_ijk.pop(key, None)
        self.neighbors = {
            k: v for k, v in self.neighbors.items() if k[:2] != key and v != key}

    def rebuild_neighbors(self):
        """Recompute same-level 6-face neighbour links from bbox adjacency.

        Two same-level blocks are face neighbours when their interiors abut in the
        level-L node grid — corners differ by exactly BS along one axis, 0 on the
        others.  Faces 0..5 = (-x,+x,-y,+y,-z,+z).  No periodicity at fine levels;
        a missing face is None (its halo is filled by cross-level prolongation).
        """
        self.neighbors = {}
        for L in range(LEVELS):
            pos = {}
            for s in range(self.caps[L]):
                if self.active[L, s]:
                    pos[self.bbox_ijk[(L, s)]] = s
            for (i0, j0, k0), s in pos.items():
                face_pos = {
                    0: (i0 - BS, j0, k0), 1: (i0 + BS, j0, k0),
                    2: (i0, j0 - BS, k0), 3: (i0, j0 + BS, k0),
                    4: (i0, j0, k0 - BS), 5: (i0, j0, k0 + BS),
                }
                for f, p in face_pos.items():
                    nb = pos.get(p)
                    self.neighbors[(L, s, f)] = (L, nb) if nb is not None else None

    def check_proper_nesting(self) -> list:
        """Berger-Oliger proper nesting: every active L≥1 block has an active
        parent at L-1 whose interior fully contains the child's footprint."""
        violations = []
        half = BS // REFINE_RATIO
        for L in range(1, LEVELS):
            for s in range(self.caps[L]):
                if not self.active[L, s]:
                    continue
                par = self.parent.get((L, s))
                if par is None:
                    violations.append((L, s, "no parent")); continue
                pL, ps = par
                if not self.active[pL, ps]:
                    violations.append((L, s, f"parent {par} inactive")); continue
                cb = self.bbox_ijk[(L, s)]
                pb = self.bbox_ijk[(pL, ps)]
                f0 = [cb[a] // REFINE_RATIO for a in range(3)]
                inside = all(pb[a] <= f0[a] and f0[a] + half <= pb[a] + BS for a in range(3))
                if not inside:
                    violations.append(
                        (L, s, f"footprint {tuple(f0)}+{half} not inside parent {pb}+{BS}"))
        return violations

    # ── JAX-side snapshot ────────────────────────────────────────────────────

    def to_jax_arrays(self) -> "AMRTopologyArrays":
        """Snapshot host topology into ragged per-level JAX arrays.  Inactive
        slots get filler; the active mask gates every consumer."""
        parent_slot = np.zeros((LEVELS, MAX_BLOCKS), np.int32)
        child_c     = np.zeros((LEVELS, MAX_BLOCKS, 3), np.int32)   # (cx,cy,cz)
        half_bs = BS // REFINE_RATIO
        for (L, s), (pL, ps) in self.parent.items():
            if not self.active[L, s]:
                continue
            parent_slot[L, s] = ps
            cb = self.bbox_ijk[(L, s)]
            pb = self.bbox_ijk[(pL, ps)]
            for a in range(3):
                rel = (cb[a] // REFINE_RATIO) - pb[a]
                child_c[L, s, a] = rel // half_bs

        neighbor_slot  = np.zeros((LEVELS, MAX_BLOCKS, NFACE), np.int32)
        neighbor_valid = np.zeros((LEVELS, MAX_BLOCKS, NFACE), bool)
        self.rebuild_neighbors()
        for (L, s, f), nb in self.neighbors.items():
            if nb is not None and self.active[L, s]:
                neighbor_slot[L, s, f] = nb[1]
                neighbor_valid[L, s, f] = True

        return AMRTopologyArrays(
            parent_slot=tuple(jnp.asarray(parent_slot[L, :self.caps[L]]) for L in range(LEVELS)),
            child_c    =tuple(jnp.asarray(child_c[L, :self.caps[L]])     for L in range(LEVELS)),
            neighbor_slot =tuple(jnp.asarray(neighbor_slot[L, :self.caps[L]])  for L in range(LEVELS)),
            neighbor_valid=tuple(jnp.asarray(neighbor_valid[L, :self.caps[L]]) for L in range(LEVELS)),
        )


class AMRTopologyArrays(NamedTuple):
    """JAX-traceable snapshot.  Each field is a tuple of LEVELS arrays sized to
    caps[L].  Per-level shapes are constant ⇒ feeding new instances into a jitted
    step does NOT retrace."""
    parent_slot: tuple    # tuple of (caps[L],) int32
    child_c:     tuple     # tuple of (caps[L], 3) int32  — (cx,cy,cz) octant
    neighbor_slot:  tuple  # tuple of (caps[L], NFACE) int32
    neighbor_valid: tuple  # tuple of (caps[L], NFACE) bool


# ── Initial-state factory ──────────────────────────────────────────────────────

def make_root_state(
    initial_data: jnp.ndarray,   # (NF, Nx_tot, Ny_tot, Nz_tot) — padded with NG halo
    nb_root: tuple,              # (nbx, nby, nbz) root tiles per axis
    caps=None,
) -> tuple[AMRState, AMRTopology]:
    """Build a level-0-only AMR state tiling ``initial_data`` into root blocks.

    ``initial_data`` is the SAME node-grid layout as the single-level patch state
    (interior + NG halo each side).  We extract the nbx×nby×nbz **interiors** of
    shape (NF, BS, BS, BS) — the halo is NOT stored (it is rebuilt transiently in
    the RHS; see amr/evolve.build_haloed).
    """
    if caps is None:
        caps = list(MAX_BLOCKS_PER_LEVEL)
    caps = list(caps)
    nbx, nby, nbz = nb_root
    Nx, Ny, Nz = initial_data.shape[1:]
    for n, nb, name in zip((Nx, Ny, Nz), (nbx, nby, nbz), "xyz"):
        exp = nb * BS + 2 * NG
        assert n == exp, f"input N{name}={n} ≠ nb*BS+2NG={exp}"

    if nbx * nby * nbz > caps[0]:
        raise ValueError(
            f"Root grid {nbx}×{nby}×{nbz} = {nbx*nby*nbz} blocks exceeds level-0 "
            f"capacity caps[0]={caps[0]}.  Increase MP_AMR_MAX_BLOCKS_PER_LEVEL[0].")

    topo = AMRTopology(caps=caps)
    blocks_l0 = np.zeros((caps[0], NF, BS, BS, BS), np.float64)
    idata = np.asarray(initial_data)
    for bi in range(nbx):
        for bj in range(nby):
            for bk in range(nbz):
                slot = (bi * nby + bj) * nbz + bk
                # interior start in the padded input is NG + b*BS
                i0, j0, k0 = NG + bi * BS, NG + bj * BS, NG + bk * BS
                blocks_l0[slot] = idata[:, i0:i0+BS, j0:j0+BS, k0:k0+BS]
                topo.add_block(0, slot, (bi * BS, bj * BS, bk * BS), parent=None)

    blocks = tuple(
        jnp.asarray(blocks_l0) if L == 0 else jnp.zeros((caps[L], NF, BS, BS, BS))
        for L in range(LEVELS))
    active = tuple(jnp.asarray(topo.active[L, :caps[L]]) for L in range(LEVELS))
    return AMRState(blocks=blocks, active=active), topo
