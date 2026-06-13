"""Block-structured AMR for the 2D MCS solver."""

from .state import (
    LEVELS, MAX_BLOCKS, MAX_BLOCKS_PER_LEVEL, mb, BS, NG, NF, REFINE_RATIO,
    AMRState, AMRTopology, AMRTopologyArrays,
    make_root_state,
)
from .kernels import (
    prolongate, restrict_into_parent, restrict_all_into_parents,
    restrict_into_parent_highorder, restrict_all_into_parents_highorder,
    sync_ghosts_within_level_root_periodic,
    sync_ghosts_within_level,
    sync_ghosts_across_levels,
    compute_indicator_gradient,
)
from .evolve import (
    make_root_step, make_two_level_step, make_subcycled_two_level_step,
    make_n_level_step, make_subcycled_n_level_step,
    make_subcycled_n_level_step_unrolled,
    amr_state_from_global, amr_state_to_global,
)
from .regrid import (
    regrid, compute_flags, enforce_nesting_buffer, apply_flags,
    evolve_with_regrid, write_caps_sidecar, read_caps_sidecar,
    make_calibrated_root_state,
    REFINE, KEEP, COARSEN,
)

# Public API (these names are re-exported deliberately).
__all__ = [
    # state
    "LEVELS", "MAX_BLOCKS", "MAX_BLOCKS_PER_LEVEL", "mb",
    "BS", "NG", "NF", "REFINE_RATIO",
    "AMRState", "AMRTopology", "AMRTopologyArrays", "make_root_state",
    # kernels
    "prolongate", "restrict_into_parent", "restrict_all_into_parents",
    "restrict_into_parent_highorder", "restrict_all_into_parents_highorder",
    "sync_ghosts_within_level_root_periodic", "sync_ghosts_within_level",
    "sync_ghosts_across_levels",
    "compute_indicator_gradient",
    # evolve
    "make_root_step", "make_two_level_step", "make_subcycled_two_level_step",
    "make_n_level_step", "make_subcycled_n_level_step",
    "make_subcycled_n_level_step_unrolled",
    "amr_state_from_global", "amr_state_to_global",
    # regrid
    "regrid", "compute_flags", "enforce_nesting_buffer", "apply_flags",
    "evolve_with_regrid", "write_caps_sidecar", "read_caps_sidecar",
    "make_calibrated_root_state",
    "REFINE", "KEEP", "COARSEN",
]
