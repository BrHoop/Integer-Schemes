"""Phase-4 (4.C) — BSSN linearized semi-discrete spectrum + MSRK CFL / ECF gate.

The canonical `validate.py` method, lifted to BSSN: linearize the pointwise RHS algebra by AD about
a constant (Minkowski) background, inject each derivative input's *stencil Fourier symbol*, assemble
the 24x24 per-mode symbol J(k) over the Brillouin zone, and feed its eigenvalues to
`mcs2d.msrk.max_stable_dt` — giving the effective CFL and Effective Cost Factor (ECF = stages/CFL)
for RK4 vs the three MSRK methods on the *actual* BSSN spectrum (gauge + CAHD + KO), not the
imaginary-axis idealization. This is the gate that picks the production integrator.

Background = Minkowski (alpha=chi=1, gt=I, all else 0); the RHS is ~0 there (fixed point), and its
Jacobian is the frozen-coefficient principal operator that sets the von Neumann CFL. Stencil symbols
are validated numerically against the actual SpatialDerivative operators.

Run:  python -m bssn3d.msrk_spectrum
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mcs_common.derivatives import SpatialDerivative
from ._bssn_rhs_generated import bssn_rhs_algebra, FIELD_INPUTS, GRAD1_INPUTS, GRAD2_INPUTS
from .state import PhysicsParams
from mcs2d.msrk import stability_radius, imag_axis_intercept, STAGES, PREV_BUFFERS

_FIDX = {f: i for i, f in enumerate(FIELD_INPUTS)}


# ── background + derivative-input metadata ───────────────────────────────────────────────────
def _minkowski_vec():
    F0 = {n: 0.0 for n in FIELD_INPUTS}
    F0["alpha"] = 1.0
    F0["chi"] = 1.0
    F0["gt0"] = F0["gt3"] = F0["gt5"] = 1.0
    return jnp.array([F0[f] for f in FIELD_INPUTS], dtype=jnp.float64)


def _deriv_meta():
    """(keys, meta) where meta[d] = (field, kind, axes) for each of the 138 derivative inputs."""
    keys, meta = [], []
    for a, f in GRAD1_INPUTS:
        keys.append(f"grad_{a}_{f}")
        meta.append((f, "d1", (a,)))
    for i, j, f in GRAD2_INPUTS:
        keys.append(f"grad2_{i}_{j}_{f}")
        meta.append((f, "d2diag" if i == j else "d2mix", (i, j)))
    return keys, meta


# ── algebra Jacobians at the background (AD) ─────────────────────────────────────────────────
def algebra_jacobians(params: PhysicsParams, dx: float, dt: float, t: float):
    keys, meta = _deriv_meta()

    def g(Fv, Dv):
        F = {f: Fv[i] for i, f in enumerate(FIELD_INPUTS)}
        D = {k: Dv[i] for i, k in enumerate(keys)}
        out = bssn_rhs_algebra(F, D, params.eta,
                               jnp.asarray(params.lmbda, dtype=jnp.float64),
                               jnp.asarray(params.lambda_f, dtype=jnp.float64),
                               params.cahd_c, dt, dx, params.ssl_h, params.ssl_sigma, t)
        return jnp.stack([out[f] for f in FIELD_INPUTS])   # rows in FIELD_INPUTS order

    Fv0 = _minkowski_vec()
    Dv0 = jnp.zeros(len(keys), dtype=jnp.float64)
    JF = np.asarray(jax.jacfwd(g, 0)(Fv0, Dv0))            # (24, 24)
    JD = np.asarray(jax.jacfwd(g, 1)(Fv0, Dv0))            # (24, 138)
    rhs0 = np.asarray(g(Fv0, Dv0))                          # ~0 (Minkowski is a fixed point)
    return JF, JD, rhs0, meta


# ── stencil Fourier symbols ──────────────────────────────────────────────────────────────────
def _stencils():
    d = SpatialDerivative(order=6, ko_order=8)
    return np.asarray(d.C1), np.asarray(d.C2), np.asarray(d.CKO)


def _sym1(theta, C1, dx):
    m = np.arange(7) - 3
    return np.sum(C1 * np.exp(1j * m * theta)) / dx          # ~ i*real (centred 1st deriv)


def _sym2(theta, C2, dx):
    m = np.arange(7) - 3
    return np.sum(C2 * np.exp(1j * m * theta)) / dx ** 2     # real <= 0


def _symko(theta, CKO, dx, sigma):
    m = np.arange(9) - 4
    return (sigma / dx) * np.sum(CKO * np.exp(1j * m * theta))   # real <= 0 (dissipative)


def validate_symbols(dx=0.37):
    """Check the analytic 1st/2nd-deriv symbols against the actual operator on a Fourier mode."""
    d = SpatialDerivative(order=6, ko_order=8)
    C1, C2, _ = _stencils()
    ng, N = d.ng, 40
    n = np.arange(-ng, N + ng)
    theta = 1.3
    mode = np.exp(1j * theta * n)                             # along axis 0
    u = jnp.asarray(np.einsum("i,j,k->ijk", mode, np.ones(8), np.ones(8)))
    c = N // 2 + ng                                           # an interior point
    g1 = np.asarray(d.compute_d1(u, dx, 0))[c, 4, 4] / mode[c]
    g2 = np.asarray(d.compute_d2(u, dx, 0))[c, 4, 4] / mode[c]
    e1 = abs(g1 - _sym1(theta, C1, dx))
    e2 = abs(g2 - _sym2(theta, C2, dx))
    return e1, e2


# ── per-mode symbol matrix + spectrum ────────────────────────────────────────────────────────
def symbol_matrix(theta, JF, JD, meta, C1, C2, CKO, dx, ko_sigma):
    """24x24 complex symbol at wave-vector theta=(tx,ty,tz)=k*dx."""
    J = JF.astype(np.complex128).copy()
    for d, (f, kind, axes) in enumerate(meta):
        if kind == "d1":
            s = _sym1(theta[axes[0]], C1, dx)
        elif kind == "d2diag":
            s = _sym2(theta[axes[0]], C2, dx)
        else:                                                # d2mix: d1 along i then d1 along j
            s = _sym1(theta[axes[0]], C1, dx) * _sym1(theta[axes[1]], C1, dx)
        J[:, _FIDX[f]] += JD[:, d] * s
    if ko_sigma > 0.0:                                        # KO is added to every field (diagonal)
        ko = sum(_symko(theta[a], CKO, dx, ko_sigma) for a in range(3))
        J[np.diag_indices(len(FIELD_INPUTS))] += ko
    return J


def spectrum(params=None, dx=1.0, dt=None, t=0.0, ko_sigma=0.0, n_theta=9):
    params = params or PhysicsParams()
    dt = dt if dt is not None else 0.25 * dx
    JF, JD, rhs0, meta = algebra_jacobians(params, dx, dt, t)
    C1, C2, CKO = _stencils()
    thetas = np.linspace(0.0, np.pi, n_theta)
    eigs = []
    for tx in thetas:
        for ty in thetas:
            for tz in thetas:
                J = symbol_matrix((tx, ty, tz), JF, JD, meta, C1, C2, CKO, dx, ko_sigma)
                eigs.append(np.linalg.eigvals(J))
    return np.concatenate(eigs), float(np.max(np.abs(rhs0)))


# ── CFL via RELATIVE stability (continuum-growth-aware) ──────────────────────────────────────
# A genuinely growing continuum mode (Re(lam)>0, e.g. the gauge/constraint mode) grows for ANY
# dt>0, so the naive "amplification <= 1" test reports CFL=0 even though it is not a dt constraint.
# Relative-stability criterion: a mode is dt-stable if the numerical amplification does not EXCEED
# the true continuum amplification, i.e. radius(lam*dt) <= max(1, exp(Re(lam)*dt)) + tol. For the
# oscillatory wave modes (Re~0) this is the usual <=1; for growing modes it allows the physical
# growth; the dt limit then comes from the high-|Im| wave modes, as it should.
def _max_stable_dt_rel(eigs, method, dt_hi=20.0, tol=1e-9):
    eigs = np.asarray(eigs)
    def ok(dt):
        for lam in eigs:
            bound = max(1.0, np.exp(lam.real * dt))
            if stability_radius(lam * dt, method) > bound + tol:
                return False
        return True
    lo, hi = 0.0, dt_hi
    if ok(hi):
        return hi
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return 0.5 * (lo + hi)


def growing_mode_diagnostic(params=None, dx=1.0, dt=0.25, t=0.0):
    """Find the most-positive-real eigenvalue across the zone and report its field content."""
    params = params or PhysicsParams()
    JF, JD, _, meta = algebra_jacobians(params, dx, dt, t)
    C1, C2, CKO = _stencils()
    best = None
    for tx in np.linspace(0, np.pi, 7):
        for ty in np.linspace(0, np.pi, 7):
            for tz in np.linspace(0, np.pi, 7):
                J = symbol_matrix((tx, ty, tz), JF, JD, meta, C1, C2, CKO, dx, 0.0)
                w, V = np.linalg.eig(J)
                idx = int(np.argmax(w.real))
                if best is None or w[idx].real > best[0]:
                    vec = np.abs(V[:, idx])
                    top = sorted(range(len(FIELD_INPUTS)), key=lambda i: -vec[i])[:4]
                    best = (w[idx].real, w[idx], (tx, ty, tz),
                            [(FIELD_INPUTS[i], round(float(vec[i]), 2)) for i in top])
    return best


# ── the ECF / CFL table ──────────────────────────────────────────────────────────────────────
def ecf_table(eigs, dx=1.0):
    methods = ["rk4", "rk4_2_1", "rk4_2_2", "rk4_3"]
    rows = []
    for m in methods:
        dt_max = _max_stable_dt_rel(eigs, m, dt_hi=20.0)
        cfl = dt_max / dx
        ecf = STAGES[m] / cfl if cfl > 0 else float("inf")
        rows.append((m, STAGES[m], PREV_BUFFERS[m], cfl, ecf))
    return rows, rows[0][4]


def main():
    p = PhysicsParams()
    e1, e2 = validate_symbols()
    print(f">> stencil-symbol check vs operator: |d1 err|={e1:.2e}  |d2 err|={e2:.2e}")
    gm = growing_mode_diagnostic(p)
    print(f">> most-positive-Re mode: Re={gm[0]:.3f}  lam={gm[1]:.3f}  at k*dx={tuple(round(x,2) for x in gm[2])}")
    print(f"   dominant fields: {gm[3]}  (continuum gauge/constraint growth, NOT a dt-CFL limit)\n")
    for ko in (0.0, 0.1, 0.2):
        eigs, rhs0 = spectrum(p, dx=1.0, dt=0.25, t=0.0, ko_sigma=ko, n_theta=9)
        rows, _ = ecf_table(eigs, dx=1.0)
        print(f"=== BSSN semi-discrete spectrum  (ko_sigma={ko}, Minkowski bg, dx=1) ===")
        print(f"    fixed-point residual |RHS(bg)|={rhs0:.2e} ; "
              f"max|Im|={np.max(np.abs(eigs.imag)):.3f}  max|Re|={np.max(np.abs(eigs.real)):.3f}  "
              f"(most-negative Re={eigs.real.min():.3f})")
        print(f"    {'method':>9} {'stages':>6} {'buf':>4} {'CFL_max':>8} {'ECF':>7} {'vs RK4':>8}")
        rk4_ecf = rows[0][4]
        for (m, st, buf, cfl, ecf) in rows:
            print(f"    {m:>9} {st:>6} {buf:>4} {cfl:>8.4f} {ecf:>7.4f} {rk4_ecf/ecf:>7.3f}x")
        print()
    print("   ECF = stages / CFL_max (cost to advance fixed physical time; lower is better).")
    print("   imag-axis intercepts (ref): RK4 %.3f  RK4-2(1) %.3f  RK4-2(2) %.3f  RK4-3 %.3f"
          % tuple(imag_axis_intercept(m) for m in ("rk4", "rk4_2_1", "rk4_2_2", "rk4_3")))


if __name__ == "__main__":
    main()
