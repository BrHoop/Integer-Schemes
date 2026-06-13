"""BSSN 24-variable state container and physics/gauge parameters.

Mirrors the ``WaveState`` pytree pattern (``mcs_common.wave_state``) but for the
24 evolved BSSN fields, in the exact variable order Dendro-GR's generated RHS
emits (``bssneqs_sympy_cse_wo_derivs.cpp``):

    alpha, chi, K,
    gt0..gt5   (conformal metric  ~g_ij, symmetric, upper-triangular)
    beta0..2   (shift)
    At0..At5   (conformal traceless A~_ij, symmetric, upper-triangular)
    Gt0..2     (conformal connection functions ~Gamma^i)
    B0..2      (Gamma-driver auxiliary)

Symmetric 3x3 tensors are stored as 6 components with the upper-triangular,
row-major index map  (0,0)->0 (0,1)->1 (0,2)->2 (1,1)->3 (1,2)->4 (2,2)->5,
matching Dendro's ``gt0..gt5`` / ``At0..At5``.
"""

from dataclasses import dataclass, field
from typing import Tuple

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

# ---- field index layout (must match the Dendro RHS output order) -------------
ALPHA = 0
CHI = 1
K = 2
GT0, GT1, GT2, GT3, GT4, GT5 = 3, 4, 5, 6, 7, 8
BETA0, BETA1, BETA2 = 9, 10, 11
AT0, AT1, AT2, AT3, AT4, AT5 = 12, 13, 14, 15, 16, 17
GTILDE0, GTILDE1, GTILDE2 = 18, 19, 20
B0, B1, B2 = 21, 22, 23

NUM_VARS = 24

VAR_NAMES = [
    "alpha", "chi", "K",
    "gt0", "gt1", "gt2", "gt3", "gt4", "gt5",
    "beta0", "beta1", "beta2",
    "At0", "At1", "At2", "At3", "At4", "At5",
    "Gt0", "Gt1", "Gt2",
    "B0", "B1", "B2",
]
assert len(VAR_NAMES) == NUM_VARS

# name -> row index in the (24, ...) data array. The Dendro grad tokens use these
# exact field names (alpha, chi, K, gt0..5, beta0..2, At0..5, Gt0..2, B0..2).
NAME_TO_IDX = {name: i for i, name in enumerate(VAR_NAMES)}

# Upper-triangular (row-major) symmetric-tensor component index, both orderings.
SYM_IDX = {
    (0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5,
    (1, 0): 1, (2, 0): 2, (2, 1): 4,
}


@register_pytree_node_class
class BSSNState:
    """24-field BSSN state as a single ``(24, Nx, Ny, Nz)`` array pytree."""

    def __init__(self, arr: jnp.ndarray):
        self.data = arr

    # --- scalar accessors ---
    @property
    def alpha(self): return self.data[ALPHA]
    @property
    def chi(self): return self.data[CHI]
    @property
    def K(self): return self.data[K]

    # --- vector accessors (return the 3 components stacked on axis 0) ---
    @property
    def beta(self): return self.data[BETA0:BETA0 + 3]
    @property
    def Gt(self): return self.data[GTILDE0:GTILDE0 + 3]
    @property
    def B(self): return self.data[B0:B0 + 3]

    # --- symmetric-tensor accessors (6 components, Dendro ordering) ---
    @property
    def gt(self): return self.data[GT0:GT0 + 6]
    @property
    def At(self): return self.data[AT0:AT0 + 6]

    def gt_ij(self, i: int, j: int): return self.data[GT0 + SYM_IDX[(i, j)]]
    def At_ij(self, i: int, j: int): return self.data[AT0 + SYM_IDX[(i, j)]]

    @property
    def shape(self): return self.data.shape[1:]

    def tree_flatten(self):
        return ((self.data,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    # --- convenience builders ---
    @classmethod
    def minkowski(cls, shape: Tuple[int, ...], dtype=jnp.float64) -> "BSSNState":
        """Flat space: alpha=chi=1, ~g_ij=delta_ij, all else 0."""
        data = jnp.zeros((NUM_VARS,) + tuple(shape), dtype=dtype)
        data = data.at[ALPHA].set(1.0)
        data = data.at[CHI].set(1.0)
        data = data.at[GT0].set(1.0)   # gt_xx
        data = data.at[GT3].set(1.0)   # gt_yy
        data = data.at[GT5].set(1.0)   # gt_zz
        return cls(data)


@dataclass(frozen=True)
class PhysicsParams:
    """BSSN gauge / damping parameters.

    The values *consumed* by the production CAHD+SSL CSE RHS
    (``bssneqs_SSL_HD_dxsq.cpp``) are: the gauge knobs ``eta``, ``lmbda``
    (``lambda[0..3]``) and ``lambda_f`` (``lambda_f[0..1]``); the
    Hamiltonian-constraint-damping strength ``cahd_c`` (``BSSN_CAHD_C``); and the
    spatial-slice-locking amplitude/width ``ssl_h`` (``h_ssl``) / ``ssl_sigma``
    (``sig_ssl``). The runtime scalars the RHS also needs — the time step ``dt``,
    the grid spacing ``dx_i`` and the current time ``t`` — are NOT physics params;
    they are supplied at evaluation (``BSSNSolver.rhs_dict``/``BSSNEvolution``).

    Defaults are Dendro-GR's standard moving-puncture values (``q1.par.toml``:
    ``BSSN_CAHD_C=0.06``, ``BSSN_SSL_H=0.6``, ``BSSN_SSL_SIGMA=20.0``); the *exact*
    numbers used for the Dendro-GR bit-compare are pinned in the oracle so both
    sides share identical params.
    """

    # --- gauge knobs consumed by the RHS ---
    eta: float = 2.0                                   # Gamma-driver shift damping
    lmbda: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # lambda[0..3] advection switches
    lambda_f: Tuple[float, float] = (1.0, 0.0)         # lambda_f[0..1] lapse-driver coefficients

    # --- constraint-damping / slice-locking consumed by the CAHD+SSL RHS ---
    cahd_c: float = 0.06                               # BSSN_CAHD_C  (Hamiltonian-constraint damping on chi)
    ssl_h: float = 0.6                                 # h_ssl        (SSL lapse-locking amplitude)
    ssl_sigma: float = 20.0                            # sig_ssl      (SSL Gaussian time-width)

    # --- operational (not consumed by the RHS; used by enforcement / BC) ---
    chi_floor: float = 1.0e-4
    alpha_floor: float = 1.0e-4

    # --- asymptotic ("background") values for outer/radiative BC ---
    alpha_inf: float = 1.0
    chi_inf: float = 1.0
    # ~g_ij -> delta_ij, K -> 0, beta -> 0, At -> 0, Gt -> 0, B -> 0 at infinity.

    def lam(self, i: int) -> float:
        """``lambda[i]`` (the name ``lambda`` is a Python keyword)."""
        return self.lmbda[i]
