"""Test-only BSSN initial data (NOT BBH ID — that is Phase 6).

Three generators, each returning a ``BSSNState`` on a ``Grid``:

* ``minkowski``            — flat space (sanity / base state).
* ``robust_stability``     — Minkowski + low-amplitude random noise on every
                             evolved variable (the standard "robust stability"
                             apples-with-apples test; correct evolution keeps it
                             bounded).
* ``gauge_wave``           — the 1D harmonic gauge-wave apples test: a traveling
                             sinusoidal lapse that is Minkowski in disguise, used
                             for FD-order convergence.

All formulas are analytic (no finite differencing needed for the ID itself).
"""

import jax
import jax.numpy as jnp

from .grid import Grid
from .state import (
    BSSNState, NUM_VARS,
    ALPHA, CHI, K, GT0, GT3, GT5, GTILDE0,
    AT0, AT3, AT5,
)


def minkowski(grid: Grid) -> BSSNState:
    """Flat space."""
    return BSSNState.minkowski(grid.shape)


def robust_stability(grid: Grid, amp: float = 1.0e-10, seed: int = 0) -> BSSNState:
    """Minkowski + uniform random noise in ``[-amp, amp]`` on all 24 fields.

    The classic robustness test: a correct, stable BSSN system must not amplify
    this round-off-scale noise into exponential growth over many crossing times.
    """
    base = BSSNState.minkowski(grid.shape).data
    key = jax.random.PRNGKey(seed)
    noise = jax.random.uniform(
        key, (NUM_VARS,) + grid.shape, minval=-amp, maxval=amp, dtype=base.dtype
    )
    return BSSNState(base + noise)


def gauge_wave(grid: Grid, amplitude: float = 0.01, wavelength: float = None,
               axis: int = 0) -> BSSNState:
    """1D harmonic gauge wave (apples-with-apples), traveling along ``axis``.

    Physical metric (Minkowski in disguise):
        ds^2 = -H dt^2 + H dx^2 + dy^2 + dz^2,
        H(x, t) = 1 - A sin(2 pi (x - t) / d).

    Converted to BSSN variables at t = 0 (zero shift, harmonic lapse):
        alpha = sqrt(H),                 beta^i = 0,  B^i = 0
        det(gamma) = H  ->  chi = H^(-1/3)
        ~g_xx = H^(2/3), ~g_yy = ~g_zz = H^(-1/3)  (det ~g = 1)
        K_xx = -d_t H / (2 sqrt(H)),     K = K_xx / H
        ~A_ij = chi (K_ij - 1/3 gamma_ij K)
        ~Gamma^x = (2/3) H^(-5/3) d_x H,  ~Gamma^{y,z} = 0
    """
    if wavelength is None:
        # default: one wavelength across the (cubic) domain on the chosen axis
        coords = (grid.x, grid.y, grid.z)[axis]
        wavelength = float(coords[-grid.ng - 1] - coords[grid.ng]) + \
            (grid.dx, grid.dy, grid.dz)[axis]

    coord = (grid.X, grid.Y, grid.Z)[axis]
    k = 2.0 * jnp.pi / wavelength
    phase = k * coord

    A = amplitude
    H = 1.0 - A * jnp.sin(phase)
    dH_dx = -A * k * jnp.cos(phase)        # spatial derivative of H at t=0
    dH_dt = -dH_dx                         # H depends on (x - t)

    sqrtH = jnp.sqrt(H)

    # ADM extrinsic curvature: only the (axis, axis) component is nonzero.
    Kxx = -dH_dt / (2.0 * sqrtH)
    Ktrace = Kxx / H                       # K = gamma^{xx} K_xx

    chi = H ** (-1.0 / 3.0)

    # conformal metric diagonal: the wave axis carries H^(2/3); transverse H^(-1/3)
    gt_wave = H ** (2.0 / 3.0)
    gt_trans = chi                         # = H^(-1/3)

    # ~A_ij = chi (K_ij - 1/3 gamma_ij K); gamma = diag(H, 1, 1) on (axis|trans)
    At_wave = chi * (Kxx - (1.0 / 3.0) * H * Ktrace)
    At_trans = chi * (0.0 - (1.0 / 3.0) * 1.0 * Ktrace)

    # ~Gamma^axis = -d_j ~g^{axis j} = (2/3) H^(-5/3) d_x H ; transverse = 0
    Gt_wave = (2.0 / 3.0) * H ** (-5.0 / 3.0) * dH_dx

    # The diagonal slots for (gt, At) depend on which axis the wave is on.
    gt_diag = (GT0, GT3, GT5)
    at_diag = (AT0, AT3, AT5)

    data = jnp.zeros((NUM_VARS,) + grid.shape, dtype=jnp.float64)
    data = data.at[ALPHA].set(sqrtH)
    data = data.at[CHI].set(chi)
    data = data.at[K].set(Ktrace)

    # ~g and ~A: wave component on `axis`, transverse on the other two
    for a in range(3):
        data = data.at[gt_diag[a]].set(gt_wave if a == axis else gt_trans)
        data = data.at[at_diag[a]].set(At_wave if a == axis else At_trans)

    data = data.at[GTILDE0 + axis].set(Gt_wave)
    return BSSNState(data)
