"""Hamiltonian + momentum constraints (transliterated Dendro-GR physcon.cpp).

Fed by the Phase-1 centred FD (same derivative-bundle pattern as the RHS). The
analytic test data (gauge wave, Minkowski) is exactly constraint-satisfying
(H = M^i = 0), so the *discrete* constraint of the analytic ID is pure truncation
error → it converges at the FD order under refinement. That convergence both
validates this transliteration and is the Phase-2.3 constraint apples test
(momentum is the under-protected channel).
"""

from typing import Dict, Tuple

import jax.numpy as jnp

from mcs_common.derivatives import SpatialDerivative
from .state import BSSNState, NAME_TO_IDX
from .grid import Grid
from ._constraints_generated import (
    FIELD_INPUTS, GRAD1_INPUTS, GRAD2_INPUTS, constraints_algebra,
)


def _bundle(state: BSSNState, diff_op: SpatialDerivative, dx, dy, dz):
    d = (dx, dy, dz)
    F = {n: state.data[NAME_TO_IDX[n]] for n in FIELD_INPUTS}
    D: Dict[str, jnp.ndarray] = {}
    for a, f in GRAD1_INPUTS:
        D[f"grad_{a}_{f}"] = diff_op.compute_d1(F[f], d[a], a)
    for i, j, f in GRAD2_INPUTS:
        if i == j:
            D[f"grad2_{i}_{j}_{f}"] = diff_op.compute_d2(F[f], d[i], i)
        else:
            t = diff_op.compute_d1(F[f], d[i], i)
            D[f"grad2_{i}_{j}_{f}"] = diff_op.compute_d1(t, d[j], j)
    return F, D


class ConstraintSolver:
    def __init__(self, grid: Grid, order: int = 6):
        self.grid = grid
        self.diff_op = SpatialDerivative(order=order)
        # The grid must carry AT LEAST the stencil's ghosts; extra ghosts (e.g. ng=4 from a
        # production 8th-order-KO grid serving this 6th-order, reach-3 constraint stencil) are
        # fine — the operator edge-pads to its own reach and leaves the interior unchanged.
        if self.diff_op.ng > grid.ng:
            raise ValueError(f"order {order} needs ng>={self.diff_op.ng}, grid ng={grid.ng}")

    def constraints(self, state: BSSNState) -> Dict[str, jnp.ndarray]:
        """``{'H','M0','M1','M2'}`` over the full padded grid."""
        F, D = _bundle(state, self.diff_op, self.grid.dx, self.grid.dy, self.grid.dz)
        return constraints_algebra(F, D)

    def l2(self, state: BSSNState) -> Tuple[float, float]:
        """Interior L2 norms (H, |M|) with |M| = sqrt(mean(M0^2+M1^2+M2^2))."""
        out = self.constraints(state)
        ng = self.grid.ng
        sl = (slice(ng, -ng),) * 3
        H = out["H"][sl]
        M0, M1, M2 = out["M0"][sl], out["M1"][sl], out["M2"][sl]
        H_l2 = float(jnp.sqrt(jnp.mean(H ** 2)))
        M_l2 = float(jnp.sqrt(jnp.mean(M0 ** 2 + M1 ** 2 + M2 ** 2)))
        return H_l2, M_l2
