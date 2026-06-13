"""Build the 138 derivative-input arrays the BSSN algebra consumes.

The Dendro CSE RHS takes derivatives as inputs (``grad_i_f`` and ``grad2_i_j_f``).
This builds exactly the 72 first + 66 second derivatives it references, from the
shared Phase-1 FD operator (``mcs_common.derivatives.SpatialDerivative``):

* ``grad_i_f``        -> ``compute_d1`` along axis i.
* ``grad2_i_i_f``     -> ``compute_d2`` along axis i (diagonal stencil).
* ``grad2_i_j_f``, i<j (mixed) -> composed ``compute_d1`` along i then j.

Dendro indexes mixed second derivatives as (min, max), so every ``grad2`` token
has i <= j; the composition order (i then j) matches that convention. Exact
bit-agreement with Dendro's own mixed-derivative association is settled by the
Phase-2.2 bit-compare.
"""

from typing import Dict

import jax.numpy as jnp

from mcs_common.derivatives import SpatialDerivative
from .state import BSSNState, NAME_TO_IDX
from ._bssn_rhs_generated import FIELD_INPUTS, GRAD1_INPUTS, GRAD2_INPUTS


def field_dict(state: BSSNState) -> Dict[str, jnp.ndarray]:
    """``{field_name: array}`` for the 24 inputs the algebra binds."""
    return {name: state.data[NAME_TO_IDX[name]] for name in FIELD_INPUTS}


def derivative_bundle(state: BSSNState, diff_op: SpatialDerivative,
                      dx: float, dy: float, dz: float) -> Dict[str, jnp.ndarray]:
    """Return ``{grad_i_f / grad2_i_j_f: array}`` for all 138 referenced inputs."""
    d = (dx, dy, dz)
    F = field_dict(state)
    D: Dict[str, jnp.ndarray] = {}

    for axis, f in GRAD1_INPUTS:
        D[f"grad_{axis}_{f}"] = diff_op.compute_d1(F[f], d[axis], axis)

    for i, j, f in GRAD2_INPUTS:
        if i == j:
            D[f"grad2_{i}_{j}_{f}"] = diff_op.compute_d2(F[f], d[i], i)
        else:
            tmp = diff_op.compute_d1(F[f], d[i], i)
            D[f"grad2_{i}_{j}_{f}"] = diff_op.compute_d1(tmp, d[j], j)

    return D
