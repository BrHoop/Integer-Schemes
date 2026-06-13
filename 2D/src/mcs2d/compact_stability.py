"""Von Neumann stability of compact (Padé/Lele) FD on the 2D MCS operator.

EXPLORATORY.  Stability is the open question for adopting compact FD, so this
quantifies it the same way the MSRK CFL work did: build the linearized
semi-discrete MCS symbol with the COMPACT modified wavenumbers (d1, d2) and the
chosen KO order, sweep the Brillouin zone, and find the max stable CFL under RK4.

Key stability facts it checks:
  * periodic compact d1 is anti-Hermitian (eigenvalues purely imaginary) and the
    LHS denom 1+2*alpha*cos(w) stays > 0 (no pole) -> the periodic operator is
    von-Neumann stable; the only question is HOW STIFF (the CFL),
  * compact resolves high-k better => the modified wavenumber reaches FURTHER up
    the imaginary axis => a STIFFER spectrum => a SMALLER max CFL than explicit
    (the price of resolution),
  * whether the KO order adequately damps the top of the band.

Non-periodic boundary closures (Sommerfeld/multipatch) are the OTHER stability
question and are NOT settled here — the rigorous route is SBP-compact operators
(cf. Diener, Dorband, Schnetter, J.Sci.Comput. 32 (2007), already a project ref).
"""
from __future__ import annotations

import os
from math import factorial as fac
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── compact coefficients ──────────────────────────────────────────────────────
# d1 stored as (alpha, [c_k]) with RHS = sum_k c_k (f_{i+k} - f_{i-k})
#   modified wavenumber  k'h = 2*sum_k c_k sin(k w) / (1 + 2 alpha cos w)
# d2 stored as (alpha, [d_k]) with RHS = sum_k d_k (f_{i+k} - 2 f_i + f_{i-k})
#   symbol  f'' h^2 = 2*sum_k d_k (cos kw - 1) / (1 + 2 alpha cos w)
C6_D1 = (1.0 / 3.0, [14.0 / 9.0 / 2.0, 1.0 / 9.0 / 4.0])
C6_D2 = (2.0 / 11.0, [12.0 / 11.0, 3.0 / 11.0 / 4.0])


def _derive_d1(reach: int):
    """Max-order tridiagonal d1 with RHS to +-reach -> (alpha, [c_k])."""
    nun = reach + 1                                  # c_1..c_reach + alpha
    A = np.zeros((nun, nun)); rhs = np.zeros(nun)
    for m in range(nun):                             # match w^(2m+1)
        p = 2 * m + 1
        for ki, k in enumerate(range(1, reach + 1)):
            A[m, ki] = 2 * ((-1) ** m) * k ** p / fac(p)
        A[m, -1] = -2 * ((-1) ** m) / fac(2 * m)     # -2 alpha (-1)^m/(2m)!
        rhs[m] = 1.0 if m == 0 else 0.0
    sol = np.linalg.solve(A, rhs)
    return sol[-1], list(sol[:-1])


def _derive_d2(reach: int):
    """Max-order tridiagonal d2 with RHS to +-reach -> (alpha, [d_k]).

    Moment conditions (matching g(w)=-w^2 to O(w^(2*reach+2))):
        sum_k d_k * k^(2j) - 2j(2j-1)*alpha = [j==1],   j = 1..reach+1.
    """
    nun = reach + 1
    A = np.zeros((nun, nun)); rhs = np.zeros(nun)
    for m in range(nun):
        j = m + 1
        for ki, k in enumerate(range(1, reach + 1)):
            A[m, ki] = k ** (2 * j)
        A[m, -1] = -2.0 * j * (2 * j - 1)            # -C_j * alpha
        rhs[m] = 1.0 if j == 1 else 0.0
    sol = np.linalg.solve(A, rhs)
    return sol[-1], list(sol[:-1])


C8_D1 = _derive_d1(3)
C8_D2 = _derive_d2(3)


# ── modified-wavenumber symbols ───────────────────────────────────────────────
def sym_d1(k, dx, scheme):
    w = k * dx
    al, cs = scheme
    num = 2.0 * sum(c * np.sin((i + 1) * w) for i, c in enumerate(cs))
    return 1j * num / (1.0 + 2.0 * al * np.cos(w)) / dx          # symbol of d/dx


def sym_d2(k, dx, scheme):
    w = k * dx
    al, ds = scheme
    num = 2.0 * sum(d * (np.cos((i + 1) * w) - 1.0) for i, d in enumerate(ds))
    return num / (1.0 + 2.0 * al * np.cos(w)) / dx ** 2          # symbol of d2/dx2


def sym_ko(k, dx, sigma, order):
    p = order // 2                                               # 6th->sin^6, 8th->sin^8
    return -(sigma / dx) * np.sin(0.5 * k * dx) ** (2 * p)


# explicit references (existing scheme)
def sym_d1_exp6(k, dx):
    t = k * dx
    return 1j / (30.0 * dx) * (45 * np.sin(t) - 9 * np.sin(2 * t) + np.sin(3 * t))


def sym_d2_exp6(k, dx):
    t = k * dx
    return (1.0 / (180.0 * dx * dx)) * (-490 + 540 * np.cos(t)
                                        - 54 * np.cos(2 * t) + 4 * np.cos(3 * t))


# ── MCS 10x10 semi-discrete symbol (structure mirrors validate._symbol) ───────
EX, EY, EZ, BX, BY, BZ, XI, PI, PSI, PHI = range(10)


def mcs_symbol(kx, ky, p, d1, d2, ko_order):
    dx, dy = p["dx"], p["dy"]
    cs, K1, K2, mc = p["cs"], p["K1"], p["K2"], p["mc"]
    sx, sy = d1(kx, dx), d1(ky, dy)
    qx, qy = d2(kx, dx), d2(ky, dy)
    ko = sym_ko(kx, dx, p["sigma"], ko_order) + sym_ko(ky, dy, p["sigma"], ko_order)
    M = np.zeros((10, 10), dtype=np.complex128)
    M[EX, BZ] += sy; M[EX, PSI] += -sx; M[EX, BX] += -mc
    M[EY, BZ] += -sx; M[EY, PSI] += -sy; M[EY, BY] += -mc
    M[EZ, BY] += sx; M[EZ, BX] += -sy; M[EZ, BZ] += -mc
    M[BX, EZ] += -sy; M[BX, PHI] += sx
    M[BY, EZ] += sx; M[BY, PHI] += sy
    M[BZ, EY] += -sx; M[BZ, EX] += sy
    M[XI, PI] += -cs
    M[PI, XI] += -cs * (qx + qy)
    M[PSI, EX] += -sx; M[PSI, EY] += -sy; M[PSI, PSI] += -K1
    M[PHI, BX] += sx; M[PHI, BY] += sy; M[PHI, PHI] += -K2
    M[np.arange(10), np.arange(10)] += ko
    return M


def spectrum(p, d1, d2, ko_order, n_k=120):
    ts = np.linspace(-np.pi, np.pi, n_k)
    eigs = []
    for kx in ts / p["dx"]:
        for ky in ts / p["dy"]:
            eigs.extend(np.linalg.eigvals(mcs_symbol(kx, ky, p, d1, d2, ko_order)))
    return np.asarray(eigs)


def rk4_radius(z):
    return abs(1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24)


def max_cfl_rk4(eigs, dx, hi=4.0, tol=1e-4):
    def stable(cfl):
        dt = cfl * dx
        return np.max(np.abs(1 + (eigs * dt) + (eigs * dt) ** 2 / 2
                             + (eigs * dt) ** 3 / 6 + (eigs * dt) ** 4 / 24)) <= 1 + 1e-9
    lo = 0.0
    while stable(hi):
        hi *= 2
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return 0.5 * (lo + hi)


# explicit/continuum d-symbols for the baseline
def sym_d1_cont(k, dx):
    return 1j * k


def sym_d2_cont(k, dx):
    return -k * k


def _max_real_eig(p, d1, d2, ko_order, n_k):
    """Max Re(eigenvalue) over the Brillouin zone, and the most-positive d2 symbol."""
    ts = np.linspace(-np.pi, np.pi, n_k)
    kxs, kys = ts / p["dx"], ts / p["dy"]
    max_re = -np.inf
    max_d2 = -np.inf
    for kx in kxs:
        max_d2 = max(max_d2, float(np.real(d2(kx, p["dx"]))))
        for ky in kys:
            ev = np.linalg.eigvals(mcs_symbol(kx, ky, p, d1, d2, ko_order))
            max_re = max(max_re, float(np.max(ev.real)))
    return max_re, max_d2


def spectral_check(n_k=140):
    """Hunt for POSITIVE real eigenvalues (exponential growth) in each scheme.

    Positive Re from CFJ physics (k < m_cs) is expected and present in the
    continuum too; the test for a NUMERICAL instability is (a) a CFJ-stable Lambda
    (lowest grid mode k_min > m_cs) where the continuum has NO growth -> any
    positive Re is numerical, and (b) at the default Lambda, the discrete max Re
    must not EXCEED the continuum's.  Also flags if the compact d2 symbol ever
    goes positive (would make the scalar sector a real, growing eigenvalue).
    """
    dx = dy = 10.0 / 512
    k_min = 2 * np.pi / 10.0
    schemes = [
        ("explicit-6", sym_d1_exp6, sym_d2_exp6),
        ("compact-6", lambda k, h: sym_d1(k, h, C6_D1),
         lambda k, h: sym_d2(k, h, C6_D2)),
        ("compact-8", lambda k, h: sym_d1(k, h, C8_D1),
         lambda k, h: sym_d2(k, h, C8_D2)),
    ]
    for L in (0.1, 0.4):
        m_cs = 2 * L
        cs = 1.0; Pi0 = (2 * L) / (2 * cs * L)
        regime = ("CFJ-STABLE (k_min=%.3f > m_cs=%.3f)" % (k_min, m_cs)
                  if k_min > m_cs else
                  "CFJ band present (k_min=%.3f < m_cs=%.3f -> physical growth)"
                  % (k_min, m_cs))
        print(f"\n=== Lambda={L}  {regime} ===")
        pc = dict(dx=dx, dy=dy, cs=cs, K1=1.0, K2=1.0, sigma=0.0,
                  mc=cs * 2 * L * Pi0)
        cont_re, _ = _max_real_eig(pc, sym_d1_cont, sym_d2_cont, 6, n_k)
        print(f"  continuum baseline      max Re = {cont_re:+.3e}")
        print(f"  {'scheme':11s} {'KO':>4s} {'max Re(lambda)':>15s} "
              f"{'max d2 symbol':>14s}  verdict")
        for name, d1, d2 in schemes:
            for sig, kotag in ((0.0, "off"), (0.05, "on")):
                p = dict(pc); p["sigma"] = sig
                koo = 8 if name == "compact-8" else 6
                mre, md2 = _max_real_eig(p, d1, d2, koo, n_k)
                # numerical growth = max Re exceeding continuum + roundoff
                bad = mre > cont_re + 1e-6
                d2bad = md2 > 1e-9
                verdict = ("NUMERICAL GROWTH" if bad else "ok") + \
                          (" | d2>0!" if d2bad else "")
                print(f"  {name:11s} {kotag:>4s} {mre:+15.3e} "
                      f"{md2:+14.3e}  {verdict}")


def run():
    dx = dy = 10.0 / 512                                # production grid
    cs, L = 1.0, 0.4
    Pi0 = (2 * L) / (2 * cs * L)
    p = dict(dx=dx, dy=dy, cs=cs, K1=1.0, K2=1.0, sigma=0.05,
             mc=cs * 2 * L * Pi0)

    print(f"alpha:  compact-6 d1={C6_D1[0]:.4f}  compact-8 d1={C8_D1[0]:.4f} "
          f"(=3/8)  compact-8 d2 alpha={C8_D2[0]:.4f}")
    print(f"8th-order d1 c_k = {[round(c,5) for c in C8_D1[1]]}\n")

    configs = [
        ("explicit-6 + KO6", sym_d1_exp6, sym_d2_exp6, 6),
        ("compact-6  + KO6", lambda k, h: sym_d1(k, h, C6_D1),
         lambda k, h: sym_d2(k, h, C6_D2), 6),
        ("compact-6  + KO8", lambda k, h: sym_d1(k, h, C6_D1),
         lambda k, h: sym_d2(k, h, C6_D2), 8),
        ("compact-8  + KO8", lambda k, h: sym_d1(k, h, C8_D1),
         lambda k, h: sym_d2(k, h, C8_D2), 8),
    ]

    print(f"{'config':20s} {'max|Im|*dx':>11s} {'max|Re|*dx':>11s} "
          f"{'CFL_max(RK4)':>12s} {'KO@Nyq':>9s}")
    for name, d1, d2, koo in configs:
        eigs = spectrum(p, d1, d2, koo)
        imr = np.max(np.abs(eigs.imag)) * dx
        rer = np.max(np.abs(eigs.real)) * dx
        cflm = max_cfl_rk4(eigs, dx)
        ko_nyq = abs(sym_ko(np.pi / dx, dx, p["sigma"], koo)) * dx   # damping at Nyquist
        print(f"{name:20s} {imr:11.3f} {rer:11.3f} {cflm:12.3f} {ko_nyq:9.3f}")

    print("\nReading: larger max|Im|*dx = stiffer (reaches higher up the imag axis)")
    print("  => smaller CFL_max.  This is the resolution<->stiffness price of compact.")
    print("  RK4 imag-axis stability limit is 2.828; CFL_max ~ 2.828/(max|Im|*dx).")
    print("  KO@Nyq = how hard the dissipation hits the Nyquist mode (sigma=0.05).")


if __name__ == "__main__":
    run()
