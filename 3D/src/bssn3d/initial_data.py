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


def hamiltonian_bump(grid: Grid, amplitude: float = 0.01) -> BSSNState:
    """Minkowski + a smooth, COHERENT conformal-factor perturbation seeding a clean
    Hamiltonian-constraint violation (and ~zero momentum violation) for the A6 CAHD
    damping-rate certificate.

    Unlike ``robust_stability`` (incoherent grid-scale noise, which KO dissipation
    damps and which carries no coherent H for CAHD to act on -- masking the very
    effect A6 wants to measure), this perturbs the conformal factor by a single
    smooth periodic mode::

        chi  ->  chi + A cos(2 pi x) cos(2 pi y) cos(2 pi z)

    on the periodic domain [-1/2, 1/2]^3 (the mode is exactly periodic). The
    conformal metric, K, and Atilde are left flat, so the momentum constraint stays
    ~0 while the curved conformal factor sources a coherent Hamiltonian constraint
    H != 0 -- precisely the structured violation the CAHD term
    (``chi_rhs += cahd_c (dx^2/dt) chi H``) is designed to damp.
    """
    base = BSSNState.minkowski(grid.shape).data
    twopi = 2.0 * jnp.pi
    pert = amplitude * jnp.cos(twopi * grid.X) * jnp.cos(twopi * grid.Y) \
        * jnp.cos(twopi * grid.Z)
    return BSSNState(base.at[CHI].add(pert))


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


def gauge_wave_solution(grid: Grid, t: float = 0.0, amplitude: float = 0.01,
                        wavelength: float = None, axis: int = 0) -> BSSNState:
    """The EXACT harmonic gauge-wave solution at time ``t`` (Minkowski in disguise).

    Identical construction to :func:`gauge_wave`, but with the traveling phase
    ``k*(x - t)`` instead of ``k*x`` -- the profile translates along ``axis`` at
    speed 1.  ``gauge_wave_solution(grid, t=0.0, ...)`` reproduces ``gauge_wave`` to
    round-off (asserted in the validation suite).

    This is an exact solution of the Einstein equations at every ``t`` (so its
    continuum Hamiltonian/momentum constraints vanish), under HARMONIC slicing.  It
    is the analytic reference for the convergence-to-exact guards.

    NOTE (see ``docs/BSSN_VALIDATION_PLAN.md`` A2): the production CAHD+SSL RHS does
    NOT preserve this solution -- it uses 1+log slicing, plus an SSL term that drives
    alpha -> 1 and CAHD damping on chi -- so an *evolved* state will diverge from this
    reference by an O(amplitude) gauge term that does not converge away.  Use this for
    constraint-of-exact-solution convergence (gauge-independent), NOT for
    evolve-then-compare-to-exact.
    """
    if wavelength is None:
        coords = (grid.x, grid.y, grid.z)[axis]
        wavelength = float(coords[-grid.ng - 1] - coords[grid.ng]) + \
            (grid.dx, grid.dy, grid.dz)[axis]

    coord = (grid.X, grid.Y, grid.Z)[axis]
    k = 2.0 * jnp.pi / wavelength
    phase = k * (coord - t)                # the only change vs gauge_wave: travels

    A = amplitude
    H = 1.0 - A * jnp.sin(phase)
    dH_dx = -A * k * jnp.cos(phase)
    dH_dt = -dH_dx

    sqrtH = jnp.sqrt(H)
    Kxx = -dH_dt / (2.0 * sqrtH)
    Ktrace = Kxx / H

    chi = H ** (-1.0 / 3.0)
    gt_wave = H ** (2.0 / 3.0)
    gt_trans = chi

    At_wave = chi * (Kxx - (1.0 / 3.0) * H * Ktrace)
    At_trans = chi * (0.0 - (1.0 / 3.0) * 1.0 * Ktrace)
    Gt_wave = (2.0 / 3.0) * H ** (-5.0 / 3.0) * dH_dx

    gt_diag = (GT0, GT3, GT5)
    at_diag = (AT0, AT3, AT5)

    data = jnp.zeros((NUM_VARS,) + grid.shape, dtype=jnp.float64)
    data = data.at[ALPHA].set(sqrtH)
    data = data.at[CHI].set(chi)
    data = data.at[K].set(Ktrace)
    for a in range(3):
        data = data.at[gt_diag[a]].set(gt_wave if a == axis else gt_trans)
        data = data.at[at_diag[a]].set(At_wave if a == axis else At_trans)
    data = data.at[GTILDE0 + axis].set(Gt_wave)
    return BSSNState(data)


def gowdy_solution(grid: Grid, t: float = 1.0) -> BSSNState:
    """Polarized Gowdy T^3 cosmology -- an exact, CURVED, nonlinear vacuum solution --
    as BSSN variables at time ``t`` (Babiuc et al. 2008 / New et al. 1998).

    Unlike the gauge waves (flat space in disguise), this is genuinely curved and
    dynamical, so it exercises the nonlinear RHS the gauge waves leave thin. The
    inhomogeneous direction is ``z``; ``x``/``y`` are the two polarization directions.
    Periodic on [-1/2, 1/2]^3 (cos(2 pi z) has period 1).

    4-metric::

        ds^2 = t^{-1/2} e^{L/2}(-dt^2 + dz^2) + t(e^P dx^2 + e^{-P} dy^2)
        P  = J0(2 pi t) cos(2 pi z)
        L  = -2 pi t J0 J1 cos^2(2 pi z) + 2 pi^2 t^2 (J0^2 + J1^2) - C

    with ``Jn = Jn(2 pi t)`` the Bessel functions (evaluated on the HOST at the scalar
    argument 2 pi t; only cos/sin(2 pi z) live on the grid -> no JAX Bessel needed) and
    ``C`` the t=1 normalization constant.

    ADM (diagonal, zero shift)::

        alpha = t^{-1/4} e^{L/4},   beta^i = 0
        gamma = diag(t e^P, t e^{-P}, t^{-1/2} e^{L/2})
        K_ij  = -(1/2 alpha) d_t gamma_ij

    NOTE (same gauge caveat as ``gauge_wave_solution`` / plan A2,B1,B2): the production
    1+log + Gamma-driver + SSL gauge does NOT preserve this solution. Use it for
    constraint-of-exact convergence (gauge-independent) and evolution self-convergence,
    NOT evolve-then-compare-to-exact.
    """
    from scipy.special import j0, j1               # host-side scalar Bessel evals

    tw = 2.0 * float(jnp.pi)
    pi2 = float(jnp.pi) ** 2
    b0, b1 = float(j0(tw * t)), float(j1(tw * t))
    c0, c1 = float(j0(tw)), float(j1(tw))           # t=1 normalization constant
    C = 0.5 * (tw * tw * (c0 * c0 + c1 * c1) - tw * c0 * c1)

    z = grid.Z
    cz = jnp.cos(tw * z)
    s2z = jnp.sin(2.0 * tw * z)                      # sin(4 pi z)

    # P, lambda and the derivatives needed for K_ij and Gamma~ (Bessel id: J0'=-J1).
    P = b0 * cz
    P_t = -tw * b1 * cz
    lam = -tw * t * b0 * b1 * cz ** 2 + 2.0 * pi2 * t * t * (b0 * b0 + b1 * b1) - C
    lam_z = 4.0 * pi2 * t * b0 * b1 * s2z
    lam_t = 4.0 * pi2 * t * (b0 * b0 - (b0 * b0 - b1 * b1) * cz ** 2)

    # ADM variables
    eP = jnp.exp(P)
    g_xx = t * eP
    g_yy = t / eP
    g_zz = t ** (-0.5) * jnp.exp(0.5 * lam)
    alpha = t ** (-0.25) * jnp.exp(0.25 * lam)

    # K_ij = -(1/2 alpha) d_t gamma_ij  (zero shift)
    dt_gxx = eP * (1.0 + t * P_t)
    dt_gyy = (1.0 / eP) * (1.0 - t * P_t)
    dt_gzz = g_zz * (0.5 * lam_t - 0.5 / t)
    Kxx = -dt_gxx / (2.0 * alpha)
    Kyy = -dt_gyy / (2.0 * alpha)
    Kzz = -dt_gzz / (2.0 * alpha)
    Ktr = Kxx / g_xx + Kyy / g_yy + Kzz / g_zz       # K = gamma^{ij} K_ij

    # BSSN reduction (det gamma = t^{3/2} e^{L/2})
    chi = (g_xx * g_yy * g_zz) ** (-1.0 / 3.0)
    At_xx = chi * (Kxx - g_xx * Ktr / 3.0)
    At_yy = chi * (Kyy - g_yy * Ktr / 3.0)
    At_zz = chi * (Kzz - g_zz * Ktr / 3.0)
    # Gamma~^z = -d_z g~^{zz} = (t/3) e^{-L/3} L_z ;  Gamma~^{x,y} = 0 (no x,y dependence)
    Gt_z = (t / 3.0) * jnp.exp(-lam / 3.0) * lam_z

    data = jnp.zeros((NUM_VARS,) + grid.shape, dtype=jnp.float64)
    data = data.at[ALPHA].set(alpha)
    data = data.at[CHI].set(chi)
    data = data.at[K].set(Ktr)
    data = data.at[GT0].set(chi * g_xx)
    data = data.at[GT3].set(chi * g_yy)
    data = data.at[GT5].set(chi * g_zz)
    data = data.at[AT0].set(At_xx)
    data = data.at[AT3].set(At_yy)
    data = data.at[AT5].set(At_zz)
    data = data.at[GTILDE0 + 2].set(Gt_z)
    return BSSNState(data)


def gowdy(grid: Grid, t0: float = 1.0) -> BSSNState:
    """Polarized Gowdy initial data at ``t = t0`` (default 1.0 -- the smooth expanding
    slice where the metric is O(1) and well resolved). Thin wrapper over
    :func:`gowdy_solution`."""
    return gowdy_solution(grid, t=t0)
