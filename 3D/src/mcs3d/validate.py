"""
Baseline validation + characterization harness for the MCS 3D solver (Phase 1).

This is the 3D port of `2D/src/mcs2d/validate.py`.  The 2D design carries over
verbatim -- *one symbol, many views*, cross-checked to the exact AD Jacobian --
and the only change is the extra spatial axis: kx, ky -> kx, ky, kz and an
`axis=2` derivative.  The symbol stays 10x10 (NF = 10 fields are unchanged); the
Brillouin zone is now 3D.

Design
------
Almost every spectral diagnostic is post-processing of ONE object: the linearized
semi-discrete symbol M(kx, ky, kz) (a 10x10 complex matrix per wavenumber) and
its real-space counterpart, the Jacobian J = d(RHS)/d(state).  Once you have
those, stability, dispersion error, KO kernel shape, mode counting, the CFJ
tachyon, the RK4 amplification factor, and strong hyperbolicity are all cheap
functions of the same eigenvalues.

The MCS RHS is quadratic in the fields (terms like Pi*B, Ez*dxi).  Linearizing at
u = 0 therefore DROPS the Chern-Simons coupling entirely (every CS term is a
product of two perturbations).  To see the real MCS physics -- birefringence and
the Carroll-Field-Jackiw tachyon -- we must linearize at the homogeneous
background Pi = Pi_0 = m_cs / (2*cs*L), where cs*2L*Pi_0*B collapses to a LINEAR
mass term with coefficient exactly m_cs.  That is the background used throughout.

CPU/GPU
-------
Everything here runs on CPU (small-matrix linear algebra and short evolutions).
GPU profiling lives in `benchmark.py` (Step 1.4) and is not part of this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mcs3d.main import (
    MaxwellChernSimons3D, InitialData, load_parameters,
    get_physical, l2norm,
)
from mcs_common.wave_state import WaveState


# ── Field indices (match MaxwellChernSimons3D) ────────────────────────────────
EX, EY, EZ = 0, 1, 2
BX, BY, BZ = 3, 4, 5
XI, PI     = 6, 7
PSI, PHI   = 8, 9
NF         = 10
FIELD_NAMES = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "xi", "Pi", "Psi", "Phi"]

def _find_repo_paths():
    """Locate 3D/params.toml + the deliverables dir robustly across install
    layouts.  `__file__`-relative resolution only works for an in-tree editable
    install; a copying/site-packages install puts the package under
    `.venv/.../site-packages/mcs3d/`, so `parents[2]` lands in the venv, not the
    repo.  Try the in-tree path first, then CWD-relative candidates (run from the
    repo root or from 3D/).  The results dir is derived from the located
    params.toml so the two never disagree about where the repo is."""
    candidates = [
        Path(__file__).resolve().parents[2] / "params.toml",  # in-tree editable: 3D/
        Path.cwd() / "3D" / "params.toml",                    # run from repo root
        Path.cwd() / "params.toml",                           # run from 3D/
    ]
    params = next((c for c in candidates if c.exists()), candidates[0])
    repo_root = params.resolve().parent.parent                # .../3D -> repo root
    results = repo_root / "docs/phases/phase_1_3d_foundation/step_1.1_results"
    return str(params), str(results)


_PARAMS_FILE, _RESULTS_DIR = _find_repo_paths()


# ══════════════════════════════════════════════════════════════════════════════
#  Full 10-field analytic oracle (3D birefringent wave)
# ══════════════════════════════════════════════════════════════════════════════

class FullBirefringentOracle:
    """Exact 10-field solution of the 3D left-circularly-polarized birefringent
    wave (omega_plus branch), matching `InitialData._birefringent_wave_3d`.

    The wave propagates along k = (kx, ky, kz) with a flawless polarization triad
    (e1, e2) orthogonal to k.  The EM fields rotate in time at
    omega = sqrt(k^2 + m_cs*k); xi grows linearly (xi = -cs*Pi_0*t, spatially
    uniform); Pi is constant at Pi_0; Psi = Phi = 0.  This is the only
    configuration of the 3D MCS system with a closed-form solution, so it is the
    gold standard for every accuracy check.

    Note the triad singularity: norm_factor = sqrt(kx^2 + ky^2) must be nonzero,
    i.e. k must not be aligned with the z-axis.  The default fundamental k (all
    components 2*pi/L) is safe; tests that pick an axis-aligned k must guard it.
    """

    def __init__(self, params: Dict[str, Any]):
        Lx = params["xmax"] - params["xmin"]
        Ly = params["ymax"] - params["ymin"]
        Lz = params["zmax"] - params["zmin"]
        self.kx = 2.0 * np.pi / Lx
        self.ky = 2.0 * np.pi / Ly
        self.kz = 2.0 * np.pi / Lz
        self.k = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        self.cs = params.get("enable_cs", 1.0)
        self.L = params.get("Lambda", 1.0)
        self.m_cs = params.get("id_m_cs", self.L * 2.0)
        self.omega = np.sqrt(self.k**2 + self.m_cs * self.k)
        self.E0 = params.get("id_amp", 1.0)
        self.Pi0 = self.m_cs / (2.0 * self.cs * self.L)

        nf = np.sqrt(self.kx**2 + self.ky**2)
        if nf < 1e-14:
            raise ValueError("Birefringent triad is singular for k aligned with "
                             "the z-axis (kx = ky = 0). Pick a generic k.")
        # e1, e2 orthonormal and perpendicular to k (matches the solver triad).
        self.e1 = np.array([self.ky / nf, -self.kx / nf, 0.0])
        self.e2 = np.array([-self.kx * self.kz / (self.k * nf),
                            -self.ky * self.kz / (self.k * nf),
                            nf / self.k])

    def state(self, X, Y, Z, t) -> np.ndarray:
        """Return the exact (10, *X.shape) field stack at time t."""
        X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
        phase = self.kx * X + self.ky * Y + self.kz * Z - self.omega * t
        cosф = np.cos(phase)
        sinф = np.sin(phase)
        b_scale = self.k / self.omega
        out = np.zeros((NF,) + X.shape, dtype=np.float64)
        for i in range(3):
            out[EX + i] = self.E0 * (self.e1[i] * cosф - self.e2[i] * sinф)
            out[BX + i] = self.E0 * b_scale * (-self.e1[i] * sinф - self.e2[i] * cosф)
        out[XI] = -self.cs * self.Pi0 * t          # linear growth, spatially uniform
        out[PI] = self.Pi0                         # constant
        out[PSI] = 0.0
        out[PHI] = 0.0
        return out

    def Ez(self, X, Y, Z, t):
        phase = (self.kx * np.asarray(X) + self.ky * np.asarray(Y)
                 + self.kz * np.asarray(Z) - self.omega * t)
        return self.E0 * (self.e1[2] * np.cos(phase) - self.e2[2] * np.sin(phase))


# ══════════════════════════════════════════════════════════════════════════════
#  6th-order finite-difference Fourier symbols (identical to 2D -- per-axis)
# ══════════════════════════════════════════════════════════════════════════════
#  For a single Fourier mode exp(i k x), applying the centred stencil
#  sum_m c[m] u[i+m] gives the multiplier sum_m c[m] exp(i k m dx).  These are the
#  exact symbols of the operators used in `SpatialDerivative` (C1, C2, CKO).

def fd_symbol_d1(k: float, dx: float) -> complex:
    """Symbol of the 6th-order centred first derivative (purely imaginary).
    -> i*k as k*dx -> 0, with O((k*dx)^6) error."""
    t = k * dx
    return 1j / (30.0 * dx) * (45.0 * np.sin(t) - 9.0 * np.sin(2 * t) + np.sin(3 * t))


def fd_symbol_d2(k: float, dx: float) -> float:
    """Symbol of the 6th-order centred second derivative (real, <= 0).
    -> -k^2 as k*dx -> 0."""
    t = k * dx
    return (1.0 / (180.0 * dx * dx)) * (
        -490.0 + 540.0 * np.cos(t) - 54.0 * np.cos(2 * t) + 4.0 * np.cos(3 * t)
    )


def ko_symbol_1d(k: float, dx: float, sigma: float) -> float:
    """Symbol of the 6th-order Kreiss-Oliger operator (CKO * sigma/dx) on one axis.
    Equals -(sigma/dx) * sin^6(k*dx/2): real, <= 0 for sigma > 0 (dissipative).
    The sign here is the load-bearing one -- a positive value would be
    anti-dissipative and blow up the Nyquist mode."""
    return -(sigma / dx) * np.sin(0.5 * k * dx) ** 6


# ══════════════════════════════════════════════════════════════════════════════
#  Linearized semi-discrete symbol  M(kx, ky, kz)
# ══════════════════════════════════════════════════════════════════════════════

def symbol_params(params: Dict[str, Any], dx: float, dy: float, dz: float) -> Dict[str, float]:
    """Pack the scalar coefficients the symbol needs from a params dict."""
    cs = params.get("enable_cs", 1.0)
    L = params.get("Lambda", 1.0)
    m_cs = params.get("id_m_cs", L * 2.0)
    Pi0 = m_cs / (2.0 * cs * L)
    return {
        "dx": dx, "dy": dy, "dz": dz, "cs": cs, "L": L,
        "K1": params.get("K1", 1.0), "K2": params.get("K2", 1.0),
        "sigma": params.get("ko_sigma", 0.05),
        "mc": cs * 2.0 * L * Pi0,            # = m_cs (CS mass at the Pi_0 background)
        "Pi0": Pi0,
    }


def _symbol(kx: float, ky: float, kz: float, p: Dict[str, float],
            *, with_ko: bool = True, with_cs: bool = True,
            principal_only: bool = False) -> np.ndarray:
    """The 10x10 complex semi-discrete symbol of the 3D MCS operator, linearized
    at the homogeneous Pi = Pi_0 background.

    Derived term-by-term from `MaxwellChernSimons3D.rhs` (the full 3D curl, the
    constraint-cleaning gradients, the CS mass coupling, the scalar wave sector,
    and KO on the diagonal).  Cross-checked to the AD Jacobian to ~1e-8.

    Parameters
    ----------
    with_ko : include Kreiss-Oliger dissipation on the diagonal.
    with_cs : include the Chern-Simons mass coupling (mc).  with_cs=False gives
              pure Maxwell + scalar wave + constraint damping.
    principal_only : keep ONLY the first-derivative (principal) terms -- no mass,
              no damping, no KO.  Used for the strong-hyperbolicity test.  The
              continuum symbol (i*k) is used instead of the FD symbol here, since
              well-posedness is a property of the PDE, not the discretization.
    """
    dx, dy, dz = p["dx"], p["dy"], p["dz"]
    cs, K1, K2 = p["cs"], p["K1"], p["K2"]
    mc = p["mc"] if with_cs else 0.0

    if principal_only:
        sx = 1j * kx
        sy = 1j * ky
        sz = 1j * kz
        mc = 0.0
        K1 = K2 = 0.0
        ko = 0.0
    else:
        sx = fd_symbol_d1(kx, dx)
        sy = fd_symbol_d1(ky, dy)
        sz = fd_symbol_d1(kz, dz)
        qx = fd_symbol_d2(kx, dx)
        qy = fd_symbol_d2(ky, dy)
        qz = fd_symbol_d2(kz, dz)
        ko = (ko_symbol_1d(kx, dx, p["sigma"]) + ko_symbol_1d(ky, dy, p["sigma"])
              + ko_symbol_1d(kz, dz, p["sigma"]) if with_ko else 0.0)

    M = np.zeros((NF, NF), dtype=np.complex128)

    # Maxwell curl (full 3D) + constraint-cleaning gradients + CS mass (-mc on E<-B).
    #   dt_Ex = (dBz_dy - dBy_dz) - dPsi_dx - mc*Bx
    #   dt_Ey = (dBx_dz - dBz_dx) - dPsi_dy - mc*By
    #   dt_Ez = (dBy_dx - dBx_dy) - dPsi_dz - mc*Bz
    M[EX, BZ] += sy;  M[EX, BY] += -sz; M[EX, PSI] += -sx; M[EX, BX] += -mc
    M[EY, BX] += sz;  M[EY, BZ] += -sx; M[EY, PSI] += -sy; M[EY, BY] += -mc
    M[EZ, BY] += sx;  M[EZ, BX] += -sy; M[EZ, PSI] += -sz; M[EZ, BZ] += -mc
    #   dt_Bx = -dEz_dy + dEy_dz + dPhi_dx
    #   dt_By = -dEx_dz + dEz_dx + dPhi_dy
    #   dt_Bz = -dEy_dx + dEx_dy + dPhi_dz
    M[BX, EZ] += -sy; M[BX, EY] += sz;  M[BX, PHI] += sx
    M[BY, EX] += -sz; M[BY, EZ] += sx;  M[BY, PHI] += sy
    M[BZ, EY] += -sx; M[BZ, EX] += sy;  M[BZ, PHI] += sz

    # Scalar (xi, Pi) wave sector: xi_t = -cs Pi ; Pi_t = -cs (d2x+d2y+d2z) xi.
    if not principal_only:
        M[XI, PI] += -cs
        M[PI, XI] += -cs * (qx + qy + qz)

    # Constraint-damping (Psi cleans divE, Phi cleans divB).
    M[PSI, EX] += -sx; M[PSI, EY] += -sy; M[PSI, EZ] += -sz; M[PSI, PSI] += -K1
    M[PHI, BX] += sx;  M[PHI, BY] += sy;  M[PHI, BZ] += sz;  M[PHI, PHI] += -K2

    # Kreiss-Oliger acts on every field's own equation (diagonal).
    if ko != 0.0:
        M[np.arange(NF), np.arange(NF)] += ko

    return M


def _symbol_continuum(kx: float, ky: float, kz: float, p: Dict[str, float]) -> np.ndarray:
    """The exact-PDE symbol: identical structure to `_symbol` but with the
    continuum operators i*k and -k^2 instead of their FD approximations and no
    KO.  Used as the dispersion-error reference."""
    M = np.zeros((NF, NF), dtype=np.complex128)
    sx, sy, sz = 1j * kx, 1j * ky, 1j * kz
    qx, qy, qz = -kx * kx, -ky * ky, -kz * kz
    cs, K1, K2, mc = p["cs"], p["K1"], p["K2"], p["mc"]
    M[EX, BZ] += sy;  M[EX, BY] += -sz; M[EX, PSI] += -sx; M[EX, BX] += -mc
    M[EY, BX] += sz;  M[EY, BZ] += -sx; M[EY, PSI] += -sy; M[EY, BY] += -mc
    M[EZ, BY] += sx;  M[EZ, BX] += -sy; M[EZ, PSI] += -sz; M[EZ, BZ] += -mc
    M[BX, EZ] += -sy; M[BX, EY] += sz;  M[BX, PHI] += sx
    M[BY, EX] += -sz; M[BY, EZ] += sx;  M[BY, PHI] += sy
    M[BZ, EY] += -sx; M[BZ, EX] += sy;  M[BZ, PHI] += sz
    M[XI, PI] += -cs;  M[PI, XI] += -cs * (qx + qy + qz)
    M[PSI, EX] += -sx; M[PSI, EY] += -sy; M[PSI, EZ] += -sz; M[PSI, PSI] += -K1
    M[PHI, BX] += sx;  M[PHI, BY] += sy;  M[PHI, BZ] += sz;  M[PHI, PHI] += -K2
    return M


def grid_wavenumbers(N: int, length: float) -> np.ndarray:
    """Discrete periodic wavenumbers resolvable on N points over `length`."""
    return 2.0 * np.pi * np.fft.fftfreq(N, d=length / N)


# ── Real-space Jacobian via forward-mode AD (cross-check + global spectrum) ────

def _build_jacobian(nx: int = 8, ny: int = 8, nz: int = 8, *,
                    params_file: str = _PARAMS_FILE,
                    ko_sigma: float = 0.05,
                    lam: float = 0.4,
                    background: str = "pi0") -> Tuple[np.ndarray, "MaxwellChernSimons3D", Dict]:
    """Exact Jacobian of the (periodic) RHS over the interior DOFs.

    The MCS RHS is quadratic, so jacfwd at a fixed background is the EXACT linear
    operator (not an approximation).  `background='pi0'` linearizes at Pi = Pi_0
    (CS physics visible); `background='zero'` gives the Maxwell+scalar core.

    Returns (J, sim, params) with J of shape (10*nx*ny*nz, 10*nx*ny*nz).  Keep
    the grid tiny: the DOF count (and the Jacobian) grow as (10 N^3)^2.
    """
    params = load_parameters(params_file)
    params.update({
        "scheme": "floating_point", "Nx": nx, "Ny": ny, "Nz": nz, "Nt": 1,
        "id_type": "birefringent", "bc_type": "periodic",
        "sponge_strength": 0.0, "ko_sigma": ko_sigma, "Lambda": lam,
    })
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / ny
    dz = (params["zmax"] - params["zmin"]) / nz
    sim = MaxwellChernSimons3D(dx, dy, dz, lam, params)
    ng = sim.ng

    if background == "pi0":
        cs = params.get("enable_cs", 1.0)
        Pi0 = (lam * 2.0) / (2.0 * cs * lam)
        u0 = np.zeros((NF, nx, ny, nz)); u0[PI] = Pi0
    else:
        u0 = np.zeros((NF, nx, ny, nz))
    u0_flat = jnp.asarray(u0.reshape(-1))

    def rhs_interior(u_flat):
        interior = u_flat.reshape(NF, nx, ny, nz)
        data = jnp.zeros((NF, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng))
        data = data.at[:, ng:ng + nx, ng:ng + ny, ng:ng + nz].set(interior)
        out = sim.rhs(WaveState(data)).data
        return out[:, ng:ng + nx, ng:ng + ny, ng:ng + nz].reshape(-1)

    J = np.asarray(jax.jacfwd(rhs_interior)(u0_flat))
    return J, sim, params


# ══════════════════════════════════════════════════════════════════════════════
#  Shared analysis helpers (imported by the CI tests)
# ══════════════════════════════════════════════════════════════════════════════

def _is_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def run_sim(scheme: str, nx: int, ny: int, nz: int, n_steps: int,
            *, params_file: str = _PARAMS_FILE, **overrides):
    """Build a birefringent/periodic 3D sim and advance it n_steps under jit+scan.
    Returns (sim, final_state, params)."""
    params = load_parameters(params_file)
    params.update({
        "scheme": scheme, "Nx": nx, "Ny": ny, "Nz": nz, "Nt": 1,
        "id_type": "birefringent", "bc_type": "periodic",
        "sponge_strength": 0.0, **overrides,
    })
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / ny
    dz = (params["zmax"] - params["zmin"]) / nz
    sim = MaxwellChernSimons3D(dx, dy, dz, params["Lambda"], params)
    state = InitialData(sim, params).generate()
    if n_steps > 0:
        def body(carry, _):
            return sim.step_rk4(carry, sim.dt), None
        state = jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_steps)[0])(state)
    return sim, state, params


def field_l2_errors(sim, state, params, t) -> Dict[str, float]:
    """L2 error of every field against the full 10-field oracle at time t."""
    oracle = FullBirefringentOracle(params)
    x = np.asarray(sim.x[sim.ng:-sim.ng])
    y = np.asarray(sim.y[sim.ng:-sim.ng])
    z = np.asarray(sim.z[sim.ng:-sim.ng])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    exact = oracle.state(X, Y, Z, t)                      # (10, nx, ny, nz)
    num = np.asarray(get_physical(state.data, sim.ng))    # (10, nx, ny, nz)
    return {FIELD_NAMES[f]: float(np.sqrt(np.mean((num[f] - exact[f]) ** 2)))
            for f in range(NF)}


def rk4_amplification_radius(M: np.ndarray, dt: float) -> float:
    """Spectral radius of the RK4 one-step amplification matrix for symbol M.
    G = I + z + z^2/2 + z^3/6 + z^4/24, z = dt*M.  Von Neumann stability <=> <= 1."""
    z = dt * M
    n = M.shape[0]
    G = (np.eye(n) + z + z @ z / 2.0
         + z @ z @ z / 6.0 + z @ z @ z @ z / 24.0)
    return float(np.max(np.abs(np.linalg.eigvals(G))))


def rk4_stability_R(z: np.ndarray) -> np.ndarray:
    """Scalar RK4 stability function R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24."""
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def em_wave_omega(kx: float, p: Dict[str, float], *, fd: bool = True) -> float:
    """Largest |Im(eigenvalue)| of the symbol at (kx, 0, 0) = the birefringent EM
    wave frequency.  fd=True uses the discrete FD symbol; fd=False the continuum
    operator (i*k, -k^2), i.e. the exact PDE dispersion."""
    if fd:
        M = _symbol(kx, 0.0, 0.0, p, with_ko=False)
    else:
        M = _symbol_continuum(kx, 0.0, 0.0, p)
    return float(np.max(np.abs(np.linalg.eigvals(M).imag)))


def mode_classification(kx: float, ky: float, kz: float, p: Dict[str, float],
                        tol: float = 1e-9) -> Dict[str, int]:
    """Count symbol eigenvalues by type at (kx, ky, kz): oscillatory (Re~0),
    damped (Re<0), growing (Re>0).  Growing modes only appear in the CFJ regime
    (|k| < m_cs).  KO is excluded so the classification reflects the physics."""
    eig = np.linalg.eigvals(_symbol(kx, ky, kz, p, with_ko=False))
    re = eig.real
    return {
        "oscillatory": int(np.sum(np.abs(re) < tol)),
        "damped":      int(np.sum(re < -tol)),
        "growing":     int(np.sum(re > tol)),
    }


def _sphere_directions(n_dirs: int) -> np.ndarray:
    """`n_dirs` quasi-uniform unit directions on the sphere (golden spiral).
    The 3D analogue of the 2D semicircle sweep -- well-posedness must hold over
    every propagation direction n-hat, not just the axes."""
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_dirs)      # polar angle, uniform in cos
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i       # golden-angle azimuth
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def principal_symbol_condition(p: Dict[str, float], n_dirs: int = 64) -> float:
    """Worst-case eigenvector-matrix condition number of the first-order
    principal symbol over propagation directions n-hat on the unit sphere.
    Bounded <=> strongly hyperbolic (well-posed).  Evaluated on the 8 first-order
    fields {E, B, Psi, Phi}; the (xi, Pi) sector is a standard scalar wave
    equation and is handled separately."""
    idx = np.array([EX, EY, EZ, BX, BY, BZ, PSI, PHI])
    worst = 0.0
    for nx, ny, nz in _sphere_directions(n_dirs):
        # Principal symbol P(n) = i*(A^x nx + A^y ny + A^z nz); use unit |k| = 1.
        M = _symbol(nx, ny, nz, p, principal_only=True)[np.ix_(idx, idx)]
        w, V = np.linalg.eig(M)
        # Real wave speeds: eigenvalues of P should be purely imaginary.
        if np.max(np.abs(w.real)) > 1e-9:
            return np.inf
        worst = max(worst, float(np.linalg.cond(V)))
    return worst


def energy_spectrum_offmode(sim, state, params) -> float:
    """Ratio of off-mode to on-mode spectral power in Ez.  The birefringent IC is
    a single Fourier mode (kx0, ky0, kz0); because MCS is linear, ALL power must
    stay in that mode.  Power leaking to harmonics signals aliasing, ghost-cell
    contamination, or an accidental nonlinearity."""
    ez = np.asarray(get_physical(state.data[EZ], sim.ng))
    F = np.fft.fftn(ez)
    P = np.abs(F) ** 2
    nx, ny, nz = ez.shape
    # On-mode = fundamental (+/- 1 in each axis), the only modes the IC excites.
    on = 0.0
    for ix in (1, nx - 1):
        for iy in (1, ny - 1):
            for iz in (1, nz - 1):
                on += P[ix, iy, iz]
    total = float(np.sum(P)) - float(P[0, 0, 0])   # drop the DC (Pi/xi offset)
    off = total - on
    return float(off / (on + 1e-300))


# ══════════════════════════════════════════════════════════════════════════════
#  Figure-producing studies (research-meeting harness)
# ══════════════════════════════════════════════════════════════════════════════

def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _converge(resolutions, *, ko, dt=None, cfl=None, t_phys=None,
              fields=("Ex", "Ey", "Ez", "Bx", "By", "Bz")):
    """Fit convergence order per field on the floating_point scheme, holding
    PHYSICAL TIME fixed across resolutions (the honest protocol -- a fixed step
    count makes T ∝ dx and inflates the order by ~1).

    dt!=None: fixed dt (temporal error a negligible constant -> isolates spatial).
    cfl!=None: production stepping dt = cfl*dx (temporal error present at its true
               order); t_phys sets the fixed end time, n_steps scales with N.
    """
    Ns = list(resolutions)
    errs = {f: [] for f in fields}
    for N in Ns:
        sim, st0, params = run_sim("floating_point", N, N, N, 0, ko_sigma=ko)
        step_dt = dt if dt is not None else cfl * sim.dx
        n_steps = round(t_phys / step_dt) if t_phys is not None else 100
        def body(c, _):
            return sim.step_rk4(c, step_dt), None
        st = jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_steps)[0])(st0)
        e = field_l2_errors(sim, st, params, n_steps * step_dt)
        for f in fields:
            errs[f].append(e[f])
    logN = np.log(Ns)
    orders = {f: -float(np.polyfit(logN, np.log(errs[f]), 1)[0]) for f in fields}
    return Ns, errs, orders


def convergence_study(resolutions=(12, 16, 24),
                      outdir: str = "validation_plots") -> Dict[str, Any]:
    """Honest fixed-time convergence, as the 'three-line story':

      1. STENCIL  (KO off, dt fixed tiny): the 6th-order FD scheme -> ~6.
      2. PRODUCTION (KO on, CFL=0.05, fixed physical time): KO-limited to ~5
         because the σ/dx·6th-difference KO is an O(h⁵) dissipation.
      3. TEMPORAL FLOOR (KO on, CFL=0.4): RK4 is 4th-order in time -> ~4.

    Grids are kept small (3D cost is (N+2NG)^3); the orders are robust on
    (12, 16, 24).
    """
    os.makedirs(outdir, exist_ok=True)
    plt = _matplotlib()

    Ns, errs_st, ord_st = _converge(resolutions, ko=0.0, dt=0.002)               # stencil
    _,  errs_pr, ord_pr = _converge(resolutions, ko=0.05, cfl=0.05, t_phys=0.3)  # production
    _,  errs_tf, ord_tf = _converge(resolutions, ko=0.05, cfl=0.40, t_phys=0.3,
                                    fields=("Ez",))                              # temporal floor

    fig, ax = plt.subplots(figsize=(8, 6))
    series = [("stencil (KO off): p=%.2f" % ord_st["Ez"], errs_st["Ez"], 6, "C0"),
              ("production (KO, CFL=0.05): p=%.2f" % ord_pr["Ez"], errs_pr["Ez"], 5, "C1"),
              ("RK4 floor (CFL=0.40): p=%.2f" % ord_tf["Ez"], errs_tf["Ez"], 4, "C3")]
    for label, y, slope, c in series:
        ax.loglog(Ns, y, "o-", color=c, label=label)
        ax.loglog(Ns, [y[0] * (Ns[0] / N) ** slope for N in Ns], ":", color=c,
                  alpha=0.6, label=f"h^{slope} ref")
    ax.set_xlabel("grid points per side N"); ax.set_ylabel("L2(Ez - oracle)")
    ax.set_title("MCS 3D spatial convergence (fixed physical time):\n"
                 "stencil 6th · production KO-limited 5th · RK4 floor 4th")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/convergence_spatial.png", dpi=140)
    plt.close(fig)

    return {"resolutions": Ns,
            "stencil_order": ord_st, "production_order": ord_pr,
            "temporal_floor_order": ord_tf["Ez"],
            "errors_stencil": errs_st, "errors_production": errs_pr}


def spectral_analysis(outdir: str = "validation_plots",
                      lam: float = 0.4, n_k: int = 200,
                      params_file: str = _PARAMS_FILE) -> Dict[str, Any]:
    """The eigenvalue battery, as four figures off one symbol:
      1. spectrum_complex_plane.png  -- eigenvalues with/without KO
      2. spectrum_dispersion.png     -- omega(k) vs analytic birefringent branches
      3. spectrum_ko_kernel.png      -- KO damping vs the analytic sin^6 symbol
      4. spectrum_cfl_margin.png     -- dt*lambda inside the RK4 stability region
    """
    os.makedirs(outdir, exist_ok=True)
    plt = _matplotlib()
    params = load_parameters(params_file)
    params.update({"Lambda": lam, "ko_sigma": 0.05, "K1": 1.0, "K2": 1.0})
    nx = ny = nz = 12                 # 12^3 = 1728 symbol evals (kept cheap in 3D)
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / ny
    dz = (params["zmax"] - params["zmin"]) / nz
    dt = params.get("cfl", 0.02) * dx
    p = symbol_params(params, dx, dy, dz)
    kxs = grid_wavenumbers(nx, params["xmax"] - params["xmin"])
    kys = grid_wavenumbers(ny, params["ymax"] - params["ymin"])
    kzs = grid_wavenumbers(nz, params["zmax"] - params["zmin"])

    eig_ko, eig_noko = [], []
    for kx in kxs:
        for ky in kys:
            for kz in kzs:
                eig_ko.extend(np.linalg.eigvals(_symbol(kx, ky, kz, p, with_ko=True)))
                eig_noko.extend(np.linalg.eigvals(_symbol(kx, ky, kz, p, with_ko=False)))
    eig_ko = np.array(eig_ko); eig_noko = np.array(eig_noko)

    # 1. Complex-plane spectrum.
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(eig_noko.real, eig_noko.imag, s=8, alpha=0.4, label="no KO")
    ax.scatter(eig_ko.real, eig_ko.imag, s=8, alpha=0.4, label="with KO (sigma=0.05)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Re(lambda)"); ax.set_ylabel("Im(lambda)")
    ax.set_title(f"3D semi-discrete spectrum (Lambda={lam})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/spectrum_complex_plane.png", dpi=140)
    plt.close(fig)

    # 2. Dispersion: omega_FD(k) vs analytic sqrt(k^2 +/- m_cs k).
    kmax = np.pi / dx
    kline = np.linspace(1e-3, kmax, n_k)
    omega_fd = np.array([em_wave_omega(k, p, fd=True) for k in kline])
    m_cs = p["mc"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(kline, omega_fd, "b.", ms=3, label="FD symbol (EM branch)")
    ax.plot(kline, np.sqrt(kline**2 + m_cs * kline), "k-", lw=1, label="sqrt(k^2 + m_cs k)")
    ax.plot(kline, np.sqrt(np.abs(kline**2 - m_cs * kline)), "r--", lw=1,
            label="sqrt|k^2 - m_cs k| (CFJ branch)")
    ax.axvline(m_cs, color="g", ls=":", label=f"k = m_cs = {m_cs:.2f} (CFJ onset)")
    ax.set_xlabel("k"); ax.set_ylabel("omega")
    ax.set_title("Numerical vs analytic MCS dispersion (3D)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/spectrum_dispersion.png", dpi=140)
    plt.close(fig)

    # 3. KO kernel: measured left-shift vs the analytic sin^6 symbol.
    shift = []
    for k in kline:
        re_ko = np.linalg.eigvals(_symbol(k, 0.0, 0.0, p, with_ko=True)).real
        re_no = np.linalg.eigvals(_symbol(k, 0.0, 0.0, p, with_ko=False)).real
        shift.append(np.min(re_ko) - np.min(re_no))     # most-damped mode
    analytic = [ko_symbol_1d(k, dx, p["sigma"]) for k in kline]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(kline, shift, "b.", ms=3, label="measured Re-shift (most damped)")
    ax.plot(kline, analytic, "k-", lw=1, label="-(sigma/dx) sin^6(k dx/2)")
    ax.set_xlabel("k"); ax.set_ylabel("Re(lambda) shift from KO")
    ax.set_title("Kreiss-Oliger dissipation kernel (3D, one axis)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/spectrum_ko_kernel.png", dpi=140)
    plt.close(fig)

    # 4. CFL margin: dt*lambda vs the RK4 stability boundary |R(z)| = 1.
    fig, ax = plt.subplots(figsize=(7, 7))
    re = np.linspace(-3.5, 1.0, 400); im = np.linspace(-3.5, 3.5, 400)
    RE, IM = np.meshgrid(re, im)
    ax.contour(RE, IM, np.abs(rk4_stability_R(RE + 1j * IM)), levels=[1.0],
               colors="k", linewidths=1.2)
    z = dt * eig_ko
    ax.scatter(z.real, z.imag, s=8, alpha=0.5, color="C0",
               label=f"dt*lambda (CFL={params.get('cfl',0.02)})")
    ax.axvline(0, color="grey", lw=0.5); ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Re(dt*lambda)"); ax.set_ylabel("Im(dt*lambda)")
    ax.set_title("CFL margin: dt*lambda inside RK4 stability region (3D)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/spectrum_cfl_margin.png", dpi=140)
    plt.close(fig)

    max_amp = max(rk4_amplification_radius(_symbol(kx, ky, kz, p, with_ko=True), dt)
                  for kx in kxs for ky in kys for kz in kzs)
    return {"max_re_eig": float(np.max(eig_ko.real)),
            "max_rk4_amplification": float(max_amp),
            "cond_principal": principal_symbol_condition(p)}


def stability_monitor(scheme: str = "floating_point",
                      n_steps: int = 300, record_every: int = 25,
                      nx: int = 24, outdir: str = "validation_plots",
                      lam: float = 0.2) -> Dict[str, Any]:
    """Track accuracy + constraints + amplitude over a run, in light-crossing
    times.  Uses the CFJ-stable regime (Lambda=0.2) so growth is a real defect,
    not the physical tachyon."""
    os.makedirs(outdir, exist_ok=True)
    plt = _matplotlib()
    from mcs3d.main import calc_constraints
    sim, state0, params = run_sim(scheme, nx, nx, nx, 0, Lambda=lam)
    oracle = FullBirefringentOracle(params)
    x = np.asarray(sim.x[sim.ng:-sim.ng]); y = np.asarray(sim.y[sim.ng:-sim.ng])
    z = np.asarray(sim.z[sim.ng:-sim.ng])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Lx = params["xmax"] - params["xmin"]

    rec = {k: [] for k in ["t", "ez_err", "xi_err", "pi_dev", "divB", "divE", "eb_max"]}
    state = state0
    step = jax.jit(lambda s: sim.step_rk4(s, sim.dt))
    for n in range(0, n_steps + 1, record_every):
        t = n * sim.dt
        data = np.asarray(get_physical(state.data, sim.ng))
        ex = oracle.state(X, Y, Z, t)
        divE, divB = calc_constraints(sim, state)
        rec["t"].append(t * 1.0 / Lx)
        rec["ez_err"].append(float(np.sqrt(np.mean((data[EZ] - ex[EZ])**2))))
        rec["xi_err"].append(float(np.sqrt(np.mean((data[XI] - ex[XI])**2))))
        rec["pi_dev"].append(float(np.max(np.abs(data[PI] - oracle.Pi0))))
        rec["divB"].append(float(l2norm(get_physical(divB, sim.ng))))
        rec["divE"].append(float(l2norm(get_physical(divE, sim.ng))))
        rec["eb_max"].append(float(np.max(np.abs(data[:6]))))
        for _ in range(record_every):
            state = step(state)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].semilogy(rec["t"], rec["ez_err"], label="L2(Ez-oracle)")
    axes[0].semilogy(rec["t"], rec["xi_err"], label="L2(xi-oracle)")
    axes[0].set_title("Solution accuracy"); axes[0].legend()
    axes[1].semilogy(rec["t"], rec["divB"], label="L2(divB)")
    axes[1].semilogy(rec["t"], rec["divE"], label="L2(divE)")
    axes[1].semilogy(rec["t"], rec["pi_dev"], label="max|Pi-Pi0|")
    axes[1].set_title("Constraints / invariants"); axes[1].legend()
    axes[2].plot(rec["t"], rec["eb_max"], label="max|E,B|")
    axes[2].set_title("Amplitude (blow-up guard)"); axes[2].legend()
    for a in axes:
        a.set_xlabel("light-crossing times"); a.grid(True, alpha=0.3)
    fig.suptitle(f"3D stability monitor ({scheme}, Lambda={lam}, {n_steps} steps)")
    fig.tight_layout(); fig.savefig(f"{outdir}/stability_monitor.png", dpi=140)
    plt.close(fig)
    return rec


def spectrum_table(outdir: str = "validation_plots",
                   params_file: str = _PARAMS_FILE) -> Dict[str, Any]:
    """Text/CSV summary: mode counts at representative k, CFJ onset, K1/K2 sweep,
    and the strong-hyperbolicity condition number."""
    os.makedirs(outdir, exist_ok=True)
    params = load_parameters(params_file)
    lines = []
    d = 0.3125

    # CFJ onset: a growing mode must appear below |k| = m_cs and vanish above.
    pf = symbol_params({**params, "Lambda": 0.4}, d, d, d)
    m_cs = pf["mc"]
    below = mode_classification(0.5 * m_cs, 0.0, 0.0, pf)
    above = mode_classification(3.0 * m_cs, 0.0, 0.0, pf)
    lines.append(f"CFJ onset m_cs = {m_cs:.3f}")
    lines.append(f"  k=0.5*m_cs: {below}")
    lines.append(f"  k=3.0*m_cs: {above}")

    # K1/K2 sweep: the constraint-cleaning pair damps at rate K/2 (underdamped
    # Gundlach oscillator).  Evaluated at a CFJ-STABLE wavenumber (|k| > m_cs).
    lines.append("Constraint-damping eigenvalue vs K (Re of most-damped mode; "
                 "expect -K/2):")
    for K in (0.0, 0.5, 1.0, 2.0, 5.0):
        pk = symbol_params({**params, "Lambda": 0.4, "K1": K, "K2": K,
                            "ko_sigma": 0.0}, d, d, d)
        eig = np.linalg.eigvals(_symbol(2.0, 2.0, 2.0, pk, with_ko=False))
        lines.append(f"  K={K}: min Re(lambda) = {np.min(eig.real):.4f}  (-K/2 = {-K/2:.4f})")

    cond = principal_symbol_condition(pf)
    lines.append(f"Strong-hyperbolicity condition number (worst dir): {cond:.3f}")

    text = "\n".join(lines)
    with open(f"{outdir}/spectrum_table.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {"m_cs": m_cs, "cfj_below": below, "cfj_above": above, "cond": cond}


def main(outdir: str = _RESULTS_DIR):
    """Run the full CPU validation harness and write a consolidated summary.

    Produces, in `outdir`: convergence_spatial.png, spectrum_*.png,
    stability_monitor.png, spectrum_table.txt, and validation_summary.txt.
    """
    os.makedirs(outdir, exist_ok=True)
    print(">> Convergence study ...")
    conv = convergence_study(outdir=outdir)
    print(">> Spectral analysis ...")
    spec = spectral_analysis(outdir=outdir)
    print(">> Stability monitor ...")
    mon = stability_monitor(outdir=outdir)
    print(">> Spectrum table ...")
    tab = spectrum_table(outdir=outdir)

    st, pr = conv["stencil_order"], conv["production_order"]
    lines = ["MCS 3D — Phase 1 validation summary", "=" * 38, "",
             "Spatial convergence order vs 10-field oracle (fixed physical time):",
             "  [1] STENCIL (6th-order FD, KO off)        -- the scheme's true order",
             *[f"        {f:3s}: p = {st[f]:.2f}" for f in ("Ex", "Ey", "Ez", "Bx", "By", "Bz")],
             "  [2] PRODUCTION (KO on, CFL=0.05)          -- KO-limited (O(h^5) dissipation)",
             *[f"        {f:3s}: p = {pr[f]:.2f}" for f in ("Ex", "Ey", "Ez", "Bx", "By", "Bz")],
             f"  [3] RK4 TEMPORAL FLOOR (CFL=0.40)         -- binds only at large CFL",
             f"        Ez : p = {conv['temporal_floor_order']:.2f}",
             "",
             "Spectral certificates (CFJ-aware; see notes):",
             f"  max Re(lambda) [Lambda=0.4, CFJ band present] : {spec['max_re_eig']:.3e}",
             f"  max RK4 |G(k)| at CFL                         : {spec['max_rk4_amplification']:.6f}",
             f"  strong-hyperbolicity cond number (worst dir)  : {spec['cond_principal']:.2f}",
             f"  CFJ onset m_cs                                : {tab['m_cs']:.3f}",
             f"  growing modes below/above m_cs                : "
             f"{tab['cfj_below']['growing']} / {tab['cfj_above']['growing']}",
             "",
             "Long-run monitor (Lambda=0.2, stable regime):",
             f"  final L2(Ez - oracle)   : {mon['ez_err'][-1]:.3e}",
             f"  final L2(divB)          : {mon['divB'][-1]:.3e}",
             f"  final max|E,B|          : {mon['eb_max'][-1]:.3f}",
             ""]
    summary = "\n".join(lines)
    with open(f"{outdir}/validation_summary.txt", "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)
    print(f">> All deliverables written to {outdir}/")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else _RESULTS_DIR)
