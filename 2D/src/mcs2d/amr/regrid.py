"""
Host-side regridding driver for the 2D MCS AMR.

The JAX side stays shape-stable: AMRState carries fixed (LEVELS, MAX_BLOCKS)
arrays, regridding only flips bits in the active mask and writes new data
into the corresponding slots.  This file orchestrates that — pure Python +
NumPy, calling into the jitted prolongate/restrict kernels to populate fresh
slots.

Phase 2 scope (this module):
  * `compute_flags`          — threshold + hysteresis → per-block decisions
  * `enforce_nesting_buffer` — proper nesting + buffer dilation of REFINE flags
  * `apply_flags`            — realise flags into AMRState + AMRTopology
  * `regrid`                 — top-level entry point
  * `evolve_with_regrid`     — time loop that regrids every K steps

Phase 3 will add:
  * Multi-block within-level ghost sync for non-root levels
  * Conservative cross-level flux correction
  * Berger-Oliger sub-cycling
"""

from __future__ import annotations

import json
import os
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .state import (
    AMRState, AMRTopology, LEVELS, MAX_BLOCKS, MAX_BLOCKS_PER_LEVEL,
    BS, NG, NF, REFINE_RATIO, make_root_state,
)
from .kernels import (
    prolongate, restrict_into_parent, compute_indicator_gradient,
)


# ── Calibration sidecar (persist discovered per-level capacities) ─────────────

def write_caps_sidecar(path: str, topo: AMRTopology) -> None:
    """Write the discovered per-level caps + peak occupancy to a JSON sidecar,
    so a later run can start pre-sized (skipping the auto-grow recompiles)."""
    data = {
        "levels": LEVELS,
        "caps": [int(c) for c in topo.caps],
        "peak_active": [int(p) for p in topo.peak_active],
        "recommended": list(topo.recommended_caps()),
        "grow_history": [list(g) for g in topo.grow_history],
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def read_caps_sidecar(path: str, default=None):
    """Read per-level caps from a sidecar written by `write_caps_sidecar`.
    Returns the `recommended` caps (a tuple) if present and length-LEVELS, else
    `default` (or MAX_BLOCKS_PER_LEVEL).  Safe to call when the file is absent."""
    if default is None:
        default = MAX_BLOCKS_PER_LEVEL
    if not os.path.exists(path):
        return tuple(default)
    with open(path) as fh:
        data = json.load(fh)
    rec = data.get("recommended") or data.get("caps")
    if rec is None or len(rec) != LEVELS:
        return tuple(default)
    return tuple(int(c) for c in rec)


def make_calibrated_root_state(
    initial_data: jnp.ndarray,
    nbx_root: int,
    nby_root: int,
    *,
    caps_sidecar: Optional[str] = None,
    caps=None,
) -> tuple[AMRState, AMRTopology]:
    """Build the root AMR state pre-sized from a calibration sidecar — the
    PRODUCTION DEFAULT so a run starts at its discovered per-level capacities
    instead of the worst-case uniform MAX_BLOCKS.

    Why this is the default: the GPU profile showed a uniform-caps run sits at
    ~9% slot occupancy, and the per-slot compute + active-mask `select` is paid
    on every dead slot.  Starting at calibrated caps was the single biggest
    measured lever (~8.8× fewer slots → ~13× less GPU work per step combined with
    A1).

    Capacity precedence:  explicit `caps`  >  sidecar `recommended`  >  uniform
    MAX_BLOCKS_PER_LEVEL.  Safe when the sidecar is absent (falls back to the
    uniform default), so the FIRST run self-calibrates and writes the sidecar
    (via `evolve_with_regrid(..., caps_sidecar=path)`), and every later run with
    the same path starts pre-sized — no auto-grow recompiles.
    """
    if caps is None and caps_sidecar is not None:
        caps = read_caps_sidecar(caps_sidecar)
    return make_root_state(initial_data, nbx_root, nby_root, caps=caps)


# Refinement decision encoded per (level, slot):
#   +1  → flagged for refinement (create children)
#    0  → keep as-is
#   -1  → flagged for coarsening (remove this block, restore parent's interior)
REFINE  =  1
KEEP    =  0
COARSEN = -1


# ── Indicator → flag decisions (with hysteresis) ──────────────────────────────

def compute_flags(
    indicators_per_level: list[np.ndarray],   # [(MAX_BLOCKS,) per level]
    topo: AMRTopology,
    *,
    refine_threshold:  float,
    coarsen_threshold: float,
    hysteresis_K: int = 3,
    max_level: Optional[int] = None,
) -> np.ndarray:
    """Convert raw per-block indicators into per-block refinement decisions
    using a streak-counter hysteresis.

    Args:
      indicators_per_level: list of (MAX_BLOCKS,) arrays, one per level.
        Each entry is the indicator magnitude for that block (0 if inactive).
      topo: the current AMR topology.  `topo.streaks` is read AND updated.
      refine_threshold: a block at level L is *flagged* for refinement when
        its indicator > refine_threshold AND has been so for `hysteresis_K`
        consecutive cycles.  Inactive blocks never get flagged.
      coarsen_threshold: a block whose indicator < coarsen_threshold for
        `hysteresis_K` consecutive cycles is flagged for coarsening.  Only
        applies to blocks that HAVE a parent (level > 0).
      hysteresis_K: cycles of agreement required before a flag fires.
      max_level: deepest level allowed to EXIST.  Blocks at level >= max_level
        are never flagged REFINE (they can't create children deeper than
        max_level).  None ⇒ LEVELS-1 (refine all the way down).  Capping depth
        bounds the slot budget and is the usual BBH control — refinement of a
        resolution-independent indicator (like |∇field|) otherwise cascades to
        the deepest level wherever the feature is.

    Returns:
      flags: (LEVELS, MAX_BLOCKS) int32 — one of REFINE, KEEP, COARSEN.

    Side effects:
      Updates `topo.streaks` in place.

    Hysteresis semantics
    --------------------
    `topo.streaks[L, s]` tracks the consensus over recent cycles.  Each cycle:
      * Indicator > refine_threshold   → streak += 1 (cap at +hysteresis_K)
      * Indicator < coarsen_threshold  → streak -= 1 (floor at -hysteresis_K)
      * Otherwise                      → streak decays toward zero by 1
    When streak >= +hysteresis_K, the block is REFINE-flagged.
    When streak <= -hysteresis_K, the block is COARSEN-flagged.
    """
    assert refine_threshold > coarsen_threshold, \
        "refine_threshold must be > coarsen_threshold (otherwise blocks oscillate)"

    deepest_refinable = (LEVELS - 1) if max_level is None else min(max_level, LEVELS - 1)

    flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
    for L in range(LEVELS):
        ind = indicators_per_level[L]
        active_mask = topo.active[L]
        # Update streaks.
        s = topo.streaks[L]
        above = (ind > refine_threshold) & active_mask
        below = (ind < coarsen_threshold) & active_mask
        # Inc/dec/decay
        new_s = np.where(above, np.minimum(s + 1,  hysteresis_K),
                np.where(below, np.maximum(s - 1, -hysteresis_K),
                                np.where(s > 0, s - 1, np.where(s < 0, s + 1, 0))))
        topo.streaks[L] = new_s

        # Decide flags.
        flags[L] = np.where(new_s >= hysteresis_K, REFINE,
                   np.where(new_s <= -hysteresis_K, COARSEN, KEEP))

        # Level 0 blocks cannot be coarsened (they're the root tiling).
        if L == 0:
            flags[L] = np.where(flags[L] == COARSEN, KEEP, flags[L])
        # A block at or below the deepest refinable level cannot create children.
        if L >= deepest_refinable:
            flags[L] = np.where(flags[L] == REFINE, KEEP, flags[L])

    return flags


# ── Proper nesting + buffer dilation ──────────────────────────────────────────

def enforce_nesting_buffer(
    flags: np.ndarray,        # (LEVELS, MAX_BLOCKS) int32 — REFINE/KEEP/COARSEN
    topo:  AMRTopology,
    *,
    n_buffer: int = 1,
) -> np.ndarray:
    """Adjust raw refine/coarsen flags to keep the hierarchy properly nested
    and to surround refined features with a buffer.  Pure NumPy; returns a NEW
    flags array (the input is not mutated).

    Two corrections are applied:

    1. NESTING (no orphans).  A block must not be coarsened while it still has
       active children — that would orphan a finer level.  Such COARSEN flags
       are downgraded to KEEP.  (Coarsening proceeds finest-level-first across
       successive regrids, so a fully-flagged sub-tree still collapses over a
       few cycles, one level at a time.)

    2. BUFFER (dilation).  For every block flagged REFINE, its same-level active
       neighbours within `n_buffer` blocks (Chebyshev distance, in block units)
       are also promoted to REFINE.  This keeps a moving feature inside the
       refined region between regrids: by the time the feature drifts into a
       neighbouring block, that block is already refined.

    Blocks are BS-sized tiles at integer `bbox_ijk` positions, so two level-L
    blocks are k-block neighbours iff their bbox differs by ≤ k·BS on each axis.
    """
    flags = flags.copy()

    # 1. Nesting: don't coarsen a block that has active children.
    for (L, s), kids in list(topo.children.items()):
        if not topo.active[L, s]:
            continue
        has_active_child = any(topo.active[cl, cs] for (cl, cs) in kids)
        if has_active_child and flags[L, s] == COARSEN:
            flags[L, s] = KEEP

    # 2. Buffer dilation per level.
    reach = n_buffer * BS
    for L in range(LEVELS):
        refine_bboxes = [
            topo.bbox_ijk[(L, s)]
            for s in range(MAX_BLOCKS)
            if topo.active[L, s] and flags[L, s] == REFINE
        ]
        if not refine_bboxes:
            continue
        for s in range(MAX_BLOCKS):
            if not topo.active[L, s] or flags[L, s] == REFINE:
                continue
            bx, by = topo.bbox_ijk[(L, s)]
            in_buffer = any(
                abs(bx - rbx) <= reach and abs(by - rby) <= reach
                for (rbx, rby) in refine_bboxes
            )
            if in_buffer:
                # Promote into the refined region (overrides KEEP and COARSEN).
                flags[L, s] = REFINE

    return flags


# ── Apply flags: create / coarsen ─────────────────────────────────────────────

def apply_flags(
    state: AMRState,
    topo:  AMRTopology,
    flags: np.ndarray,    # (LEVELS, MAX_BLOCKS) int32
    *,
    auto_grow: bool = True,
) -> tuple[AMRState, AMRTopology]:
    """Realise the REFINE / COARSEN flags into the AMR state and topology.

    Order of operations (top-down for refine, bottom-up for coarsen):
      * Refine: for each REFINE-flagged block at level L, allocate up to 4
        (2D) new slots at level L+1, prolongate the parent block into each,
        update topology.
      * Coarsen: for each COARSEN-flagged block at level L, restrict back
        into its parent's matching quadrant, mark slot inactive, drop from
        topology dicts.

    Refinement creates children at the immediate parent (no skip-levels).
    The 4 children correspond to corners (0,0), (0,1), (1,0), (1,1).

    `auto_grow`: when a level's slot capacity is exhausted, grow it (sticky,
    up to the hard ceiling MAX_BLOCKS) and record the event in
    `topo.grow_history` — the per-level JAX arrays enlarge, so the next jitted
    step retraces ONCE for the new shape (rare calibration cost, not per-regrid).
    With `auto_grow=False`, exhaustion raises immediately.

    Returns the new (state, topo).  AMRState is rebuilt because JAX arrays
    are immutable; topo is mutated in place AND returned for clarity.
    """
    # Per-level writable numpy copies (storage is ragged → list, not one array).
    blocks_np = [np.array(state.blocks[L], copy=True) for L in range(LEVELS)]
    active_np = [np.array(state.active[L], copy=True) for L in range(LEVELS)]

    def _grow_level(child_L: int):
        """Enlarge level child_L's capacity (sticky), padding its numpy arrays."""
        old = topo.caps[child_L]
        new = min(max(old + max(8, old // 2), old + 1), MAX_BLOCKS)
        if new <= old:
            raise RuntimeError(
                f"AMR hard ceiling reached at level {child_L}: capacity {old} "
                f"== MAX_BLOCKS={MAX_BLOCKS} and more blocks are needed.  Raise "
                f"MCS_AMR_MAX_BLOCKS (the absolute slot ceiling / host array width)."
            )
        pad = new - old
        blocks_np[child_L] = np.concatenate(
            [blocks_np[child_L],
             np.zeros((pad, NF, BS + 2*NG, BS + 2*NG), dtype=blocks_np[child_L].dtype)],
            axis=0,
        )
        active_np[child_L] = np.concatenate(
            [active_np[child_L], np.zeros(pad, dtype=bool)]
        )
        topo.caps[child_L] = new
        topo.grow_history.append((child_L, old, new))
        print(f"[AMR] level {child_L} capacity {old}→{new} (auto-grow; "
              f"recompiles the step once)")

    # ── Coarsen pass ──────────────────────────────────────────────────────────
    # For each COARSEN-flagged fine block: restrict back into parent.  Sweep
    # from finest level upward so a coarsen cascade is possible later (Phase 3).
    for L in range(LEVELS - 1, 0, -1):
        for s in range(topo.caps[L]):
            if flags[L, s] != COARSEN or not active_np[L][s]:
                continue
            key = (L, s)
            if key not in topo.parent:
                continue   # orphan — shouldn't happen if topology is consistent
            (pL, ps) = topo.parent[key]
            # Determine which corner this child covers (from bbox or stored info).
            child_bbox = topo.bbox_ijk[key]
            parent_bbox = topo.bbox_ijk[(pL, ps)]
            # child_bbox is in level-L coords; parent_bbox in level-(L-1).
            # Child's relative position inside parent (in parent-cell units):
            rel_x = (child_bbox[0] // REFINE_RATIO) - parent_bbox[0]
            rel_y = (child_bbox[1] // REFINE_RATIO) - parent_bbox[1]
            # Each child spans BS/2 parent cells; corner index is rel / (BS/2).
            cx = rel_x // (BS // REFINE_RATIO)
            cy = rel_y // (BS // REFINE_RATIO)

            # Restrict the fine interior into the parent's quadrant.
            child_int = blocks_np[L][s][:, NG:NG+BS, NG:NG+BS]
            new_parent = np.asarray(restrict_into_parent(
                jnp.asarray(blocks_np[pL][ps]), jnp.asarray(child_int), (int(cx), int(cy))
            ))
            blocks_np[pL][ps] = new_parent

            # Deactivate the fine block.
            active_np[L][s] = False
            topo.remove_block(L, s)

    # ── Refine pass ───────────────────────────────────────────────────────────
    # For each REFINE-flagged parent: allocate 4 children, prolongate.
    for L in range(LEVELS - 1):
        child_L = L + 1
        for s in range(topo.caps[L]):
            if flags[L, s] != REFINE or not active_np[L][s]:
                continue

            parent_block = blocks_np[L][s]
            parent_bbox  = topo.bbox_ijk[(L, s)]

            for cx in (0, 1):
                for cy in (0, 1):
                    # Allocate a fresh slot at the child level.
                    # Skip if this child already exists (e.g., partial refine).
                    already_exists = any(
                        topo.bbox_ijk.get((child_L, ss)) ==
                        (parent_bbox[0] * REFINE_RATIO + cx * BS,
                         parent_bbox[1] * REFINE_RATIO + cy * BS)
                        for ss in range(topo.caps[child_L])
                        if active_np[child_L][ss]
                    )
                    if already_exists:
                        continue

                    free = np.flatnonzero(~active_np[child_L])
                    if len(free) == 0:
                        if not auto_grow:
                            raise RuntimeError(
                                f"AMR budget exhausted at level {child_L}: all "
                                f"{topo.caps[child_L]} slots in use.  Increase "
                                f"MCS_AMR_MAX_BLOCKS_PER_LEVEL[{child_L}] or enable auto_grow."
                            )
                        _grow_level(child_L)
                        free = np.flatnonzero(~active_np[child_L])
                    slot = int(free[0])

                    child_block = np.asarray(prolongate(
                        jnp.asarray(parent_block), (cx, cy)
                    ))
                    blocks_np[child_L][slot] = child_block
                    active_np[child_L][slot] = True
                    # Child bbox in level-(L+1) cell units:
                    child_bbox = (parent_bbox[0] * REFINE_RATIO + cx * BS,
                                  parent_bbox[1] * REFINE_RATIO + cy * BS)
                    topo.add_block(level=child_L, slot=slot,
                                   bbox_ij=child_bbox, parent=(L, s))

    new_state = AMRState(
        blocks=tuple(jnp.asarray(b) for b in blocks_np),
        active=tuple(jnp.asarray(a) for a in active_np),
    )
    return new_state, topo


# ── Top-level entry point ─────────────────────────────────────────────────────

def regrid(
    state: AMRState,
    topo:  AMRTopology,
    dx_per_level: list[float] | np.ndarray,
    *,
    field_idx: int = 2,           # default EZ
    refine_threshold:  float,
    coarsen_threshold: float,
    hysteresis_K: int = 3,
    n_buffer: int = 1,
    max_level: Optional[int] = None,
) -> tuple[AMRState, AMRTopology]:
    """One regrid pass: indicators → hysteresis flags → nesting/buffer → apply.

    Convenience wrapper combining `compute_indicator_gradient` + `compute_flags`
    + `enforce_nesting_buffer` + `apply_flags`.  Call every K steps in the time
    loop (see `evolve_with_regrid`).  `max_level` caps refinement depth.
    """
    indicators_per_level = []
    for L in range(LEVELS):
        ind = np.asarray(compute_indicator_gradient(
            state.blocks[L], dx_per_level[L], field_idx
        ))
        # Mask inactive slots to zero so they never get refine-flagged.
        ind = np.where(np.asarray(state.active[L]), ind, 0.0)
        indicators_per_level.append(ind)

    flags = compute_flags(
        indicators_per_level, topo,
        refine_threshold=refine_threshold,
        coarsen_threshold=coarsen_threshold,
        hysteresis_K=hysteresis_K,
        max_level=max_level,
    )
    flags = enforce_nesting_buffer(flags, topo, n_buffer=n_buffer)
    new_state, new_topo = apply_flags(state, topo, flags)
    new_topo.record_occupancy()   # calibration data (peak active per level)
    return new_state, new_topo


# ── Time loop with periodic regridding ────────────────────────────────────────

def evolve_with_regrid(
    state: AMRState,
    topo:  AMRTopology,
    step,                              # jitted step(state, topology_arrays) → state
    dx_per_level: list[float] | np.ndarray,
    *,
    n_steps: int,
    regrid_every: int,
    refine_threshold:  float,
    coarsen_threshold: float,
    hysteresis_K: int = 3,
    n_buffer: int = 1,
    max_level: Optional[int] = None,
    field_idx: int = 2,
    on_regrid=None,                   # optional callback(step_index, state, topo)
    caps_sidecar: Optional[str] = None,   # path to read/write calibrated caps
    runaway_limit: int = 50,          # consecutive growth regrids → error
) -> tuple[AMRState, AMRTopology]:
    """Advance `state` by `n_steps`, regridding every `regrid_every` steps.

    `step` is a compiled N-level step taking `(state, topology_arrays)`.  Between
    regrid events the topology is fixed, so the inner `regrid_every` steps run
    under a single jitted `lax.scan`; rebuilding the topology arrays after each
    regrid does NOT re-trace as long as their per-level shapes are unchanged.

    Auto-calibration: if a regrid grows a level's capacity (`apply_flags`
    auto_grow), the per-level arrays enlarge and the next chunk's `run_chunk`
    retraces ONCE for the new shape — the rare, intended recompile.  Growth on
    more than `runaway_limit` consecutive regrids raises (runaway refinement).
    If `caps_sidecar` is set, the discovered caps are written there at the end
    (and the recommendation is logged) so the next run can start pre-sized.

    Returns the final (state, topo).  `on_regrid`, if given, is called after
    each regrid event with (steps_completed, state, topo).
    """
    import jax

    # Compiled once; reused for every full chunk (topology SHAPE is constant, so
    # passing a fresh AMRTopologyArrays each call does not re-trace).
    @jax.jit
    def run_chunk(st, topology_arrays):
        def body(s, _):
            return step(s, topology_arrays), None
        return jax.lax.scan(body, st, None, length=regrid_every)[0]

    steps_done = 0
    consecutive_grow = [0]   # consecutive regrids that grew a capacity (runaway guard)
    while steps_done < n_steps:
        chunk = min(regrid_every, n_steps - steps_done)
        topology_arrays = topo.to_jax_arrays()
        if chunk == regrid_every:
            state = run_chunk(state, topology_arrays)
        else:
            # Final short chunk: plain python loop (rare; avoids a 2nd compile).
            for _ in range(chunk):
                state = step(state, topology_arrays)
        steps_done += chunk

        if steps_done < n_steps:   # don't regrid after the very last chunk
            n_grows_before = len(topo.grow_history)
            state, topo = regrid(
                state, topo, dx_per_level,
                field_idx=field_idx,
                refine_threshold=refine_threshold,
                coarsen_threshold=coarsen_threshold,
                hysteresis_K=hysteresis_K,
                n_buffer=n_buffer,
                max_level=max_level,
            )
            # Runaway guard: capacity that keeps growing every regrid signals
            # pathological refinement (a bug or instability), not calibration.
            if len(topo.grow_history) > n_grows_before:
                consecutive_grow[0] += 1
                if consecutive_grow[0] > runaway_limit:
                    raise RuntimeError(
                        f"AMR auto-grow runaway: capacity grew on "
                        f"{consecutive_grow[0]} consecutive regrids "
                        f"(grow_history={topo.grow_history[-5:]}).  Likely "
                        f"runaway refinement — check the indicator/instability."
                    )
            else:
                consecutive_grow[0] = 0
            if on_regrid is not None:
                on_regrid(steps_done, state, topo)

    # Calibration summary + optional persistence.
    print(f"[AMR] peak occupancy per level = {tuple(topo.peak_active)}; "
          f"final caps = {tuple(topo.caps)}; "
          f"recommended MCS_AMR_MAX_BLOCKS_PER_LEVEL="
          f"{','.join(str(c) for c in topo.recommended_caps())}")
    if caps_sidecar is not None:
        write_caps_sidecar(caps_sidecar, topo)
    return state, topo
