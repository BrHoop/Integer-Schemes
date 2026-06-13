"""Multistep Runge–Kutta (MSRK) time integrators for the 2D MCS solver.

Implements the three fourth-order MSRK schemes of Sanches, Brandt, Kalinani,
Ji & Schnetter, *Accelerating Numerical Relativity Simulations with New
Multistep Fourth-Order Runge-Kutta Methods* (arXiv:2603.05763, CQG 2026):

  * ``rk4_2_1`` — 2-step / 3-stage, solution (1).  Stores 1 previous RHS.
  * ``rk4_2_2`` — 2-step / 3-stage, solution (2).  Stores 1 previous RHS.
  * ``rk4_3``   — 3-step / 2-stage.                Stores 2 previous RHS.

All reuse previous-step RHS evaluations to cut the number of fresh RHS calls
per step from RK4's 4 to 3 (rk4_2) or 2 (rk4_3).  This module is the cheap
"correctness ground" assessment requested before any 3D / BSSN commitment:
it lets us measure each method's convergence order, stability region, effective
CFL, and accuracy on the 2D MCS oracle.

Coefficients are entered as exact fractions (paper Table 1, the ASR-intercept-
maximising values) and converted to float64; ``_validate_coeffs`` checks the
consistency condition sum(b)=1 and c_i = sum_j a_ij at import time.

Startup: an r-step method has no RHS history at t=0 (and, in production, after
each AMR regrid).  Following the paper, we take ``n_prev`` ordinary RK4 steps to
fill the missing previous-RHS levels before switching to the MSRK recurrence;
because RK4 is itself 4th-order this preserves the global order.

The ``companion_matrix`` / stability helpers are pure-numpy (no JAX) so the
linear von Neumann analysis can run anywhere.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List, Tuple

import numpy as np

# ── Method coefficient tables (paper Table 1) ─────────────────────────────────
# Each entry: a-matrix rows (a20,a21 / a30,a31,a32), b vector, c nodes, plus the
# structural metadata n_stages (fresh RHS evals/step), n_prev (stored prior RHS).
F = Fraction

_RK4_2_1 = dict(
    b=[F(-643, 1536), F(-4237, 1092), F(38125, 10752), F(4375, 2496)],
    a20=F(-49, 1250), a21=F(399, 1250),
    a30=F(7033, 960000), a31=F(-217633, 210000), a32=F(5473, 10752),
    c2=F(7, 25), c3=F(-13, 25),
)
_RK4_2_2 = dict(
    b=[F(-191, 882), F(48241, 59994), F(193750, 4351347), F(100000, 271791)],
    a20=F(1309, 15500), a21=F(-31999, 15500),
    a30=F(-241289, 5880000), a31=F(22846301, 16170000), a32=F(-936169, 2587200),
    c2=F(-99, 50), c3=F(101, 100),
)
_RK4_3 = dict(
    b=[F(-85, 1416), F(131, 408), F(-29, 24), F(15625, 8024)],
    a30=F(2511, 62500), a31=F(-2268, 15625), a32=F(29061, 62500),
    c3=F(9, 25),
)


class _Method:
    """Float coefficients + structure for one MSRK method."""

    def __init__(self, name: str, raw: dict, n_steps: int):
        self.name = name
        self.n_prev = n_steps - 1          # stored previous-RHS levels
        self.b = [float(x) for x in raw["b"]]
        if self.n_prev == 1:               # rk4_2 family: 3 fresh stages
            self.n_stages = 3
            self.a20, self.a21 = float(raw["a20"]), float(raw["a21"])
            self.a30, self.a31, self.a32 = (
                float(raw["a30"]), float(raw["a31"]), float(raw["a32"]))
        else:                              # rk4_3: 2 fresh stages
            self.n_stages = 2
            self.a30, self.a31, self.a32 = (
                float(raw["a30"]), float(raw["a31"]), float(raw["a32"]))
        _validate_coeffs(name, raw)


def _validate_coeffs(name: str, raw: dict) -> None:
    """Exact-arithmetic checks of the order/consistency conditions."""
    assert sum(raw["b"]) == 1, f"{name}: sum(b) != 1 (consistency)"
    if "a20" in raw:                       # 2-step: c2 = a20+a21, c3 = a30+a31+a32
        assert raw["a20"] + raw["a21"] == raw["c2"], f"{name}: c2 != a20+a21"
        assert raw["a30"] + raw["a31"] + raw["a32"] == raw["c3"], \
            f"{name}: c3 != a30+a31+a32"
    else:                                  # 3-step: c3 = a30+a31+a32
        assert raw["a30"] + raw["a31"] + raw["a32"] == raw["c3"], \
            f"{name}: c3 != a30+a31+a32"


METHODS: Dict[str, _Method] = {
    "rk4_2_1": _Method("rk4_2_1", _RK4_2_1, n_steps=2),
    "rk4_2_2": _Method("rk4_2_2", _RK4_2_2, n_steps=2),
    "rk4_3":   _Method("rk4_3",   _RK4_3,   n_steps=3),
}

# Display metadata used by the analysis / tables.
STAGES = {"rk4": 4, "rk4_2_1": 3, "rk4_2_2": 3, "rk4_3": 2}
PREV_BUFFERS = {"rk4": 0, "rk4_2_1": 1, "rk4_2_2": 1, "rk4_3": 2}


# ══════════════════════════════════════════════════════════════════════════════
#  Linear stability:  companion matrix of the method's stability polynomial
# ══════════════════════════════════════════════════════════════════════════════
#  Apply the method to the scalar model problem y' = lambda*y with z = lambda*dt.
#  The update is a linear recurrence y_{n+1} = sum_j P_j(z) y_{n-j}; its companion
#  matrix's spectral radius <= 1 is the absolute-stability root condition (paper
#  Eq. 8).  We build P_j(z) by applying the step map to unit basis states — this
#  is exact and avoids hand-deriving the polynomials.

def companion_matrix(z: complex, method: str) -> np.ndarray:
    """Companion matrix C(z) with [y_{n+1}, y_n, ...]^T = C(z) [y_n, y_{n-1}, ...]^T."""
    if method == "rk4":
        R = 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24
        return np.array([[R]], dtype=np.complex128)

    m = METHODS[method]
    b0, b1, b2, b3 = m.b

    if m.n_prev == 1:                      # rk4_2 family
        def step(yn: complex, ynm1: complex) -> complex:
            g0 = z * ynm1                  # dt*f(y_{n-1})  (stored)
            g1 = z * yn                    # dt*f(y_n)
            g2 = z * (yn + m.a20 * g0 + m.a21 * g1)
            g3 = z * (yn + m.a30 * g0 + m.a31 * g1 + m.a32 * g2)
            return yn + b0 * g0 + b1 * g1 + b2 * g2 + b3 * g3
        P, Q = step(1.0, 0.0), step(0.0, 1.0)
        return np.array([[P, Q], [1.0, 0.0]], dtype=np.complex128)

    # rk4_3: 3-step / 2-stage
    def step3(yn: complex, ynm1: complex, ynm2: complex) -> complex:
        g0 = z * ynm2                      # dt*f(y_{n-2})  (stored)
        g1 = z * ynm1                      # dt*f(y_{n-1})  (stored)
        g2 = z * yn                        # dt*f(y_n)
        g3 = z * (yn + m.a30 * g0 + m.a31 * g1 + m.a32 * g2)
        return yn + b0 * g0 + b1 * g1 + b2 * g2 + b3 * g3
    P = step3(1.0, 0.0, 0.0)
    Q = step3(0.0, 1.0, 0.0)
    R = step3(0.0, 0.0, 1.0)
    return np.array([[P, Q, R], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.complex128)


def stability_radius(z: complex, method: str) -> float:
    """Spectral radius of the companion matrix (<= 1  <=>  z in the ASR)."""
    return float(np.max(np.abs(np.linalg.eigvals(companion_matrix(z, method)))))


def imag_axis_intercept(method: str, *, hi: float = 4.0, tol: float = 1e-6) -> float:
    """Largest B>0 with i*B on the absolute-stability boundary (paper's 'intercept').

    Bisects the root condition along the positive imaginary axis.  Reproduces the
    paper's published values (RK4 sqrt(8)=2.8284; RK4-2(1) 2.5387; RK4-2(2)
    2.4620; RK4-3 1.3071) — a direct check of the coefficients + companion build.
    """
    def stable(B: float) -> bool:
        return stability_radius(1j * B, method) <= 1.0 + 1e-9
    lo = 0.0
    # ensure hi is outside (unstable); expand if a method has a huge region
    while stable(hi):
        hi *= 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return 0.5 * (lo + hi)


def max_stable_dt(eigs: np.ndarray, method: str, *,
                  dt_hi: float = 10.0, tol: float = 1e-9) -> float:
    """Largest dt for which every semi-discrete eigenvalue lands in the ASR.

    ``eigs`` are the eigenvalues of the linearized semi-discrete operator M
    (built from validate._symbol over the Brillouin zone).  For each candidate
    dt we require max_i stability_radius(eig_i * dt) <= 1.  Grid-/method-honest
    von Neumann limit; CFL_max = max_stable_dt / dx.
    """
    eigs = np.asarray(eigs)

    def stable(dt: float) -> bool:
        # vector-free but cheap: companion is 1..3 dim, eigs ~ O(10^4)
        for lam in eigs:
            if stability_radius(lam * dt, method) > 1.0 + 1e-9:
                return False
        return True

    lo, hi = 0.0, dt_hi
    if stable(hi):                         # unbounded sample → return the cap
        return hi
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════════════
#  JAX time-steppers (operate on WaveState pytrees via sim.rhs)
# ══════════════════════════════════════════════════════════════════════════════

def _import_jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def _axpy(base, dt: float, coeffs: List[float], ks: list):
    """base + dt * sum_i coeffs[i] * ks[i]  over a WaveState pytree."""
    import jax
    return jax.tree_util.tree_map(
        lambda b, *kk: b + dt * sum(c * a for c, a in zip(coeffs, kk)),
        base, *ks)


def rk4_step(rhs: Callable, state, dt: float):
    """One classical RK4 step.  Returns (new_state, k1) where k1 = f(state)
    (the leading-stage RHS, reused to seed the MSRK history at startup)."""
    import jax
    tm = jax.tree_util.tree_map
    k1 = rhs(state)
    k2 = rhs(tm(lambda s, k: s + 0.5 * dt * k, state, k1))
    k3 = rhs(tm(lambda s, k: s + 0.5 * dt * k, state, k2))
    k4 = rhs(tm(lambda s, k: s + dt * k, state, k3))
    new = tm(lambda s, a, b, c, d: s + (dt / 6.0) * (a + 2 * b + 2 * c + d),
             state, k1, k2, k3, k4)
    return new, k1


def _msrk2_step(rhs: Callable, m: _Method, state, fprev, dt: float):
    """One rk4_2 step.  carry = (state=y_n, fprev=f(y_{n-1}))."""
    k0 = fprev
    k1 = rhs(state)
    Y2 = _axpy(state, dt, [m.a20, m.a21], [k0, k1])
    k2 = rhs(Y2)
    Y3 = _axpy(state, dt, [m.a30, m.a31, m.a32], [k0, k1, k2])
    k3 = rhs(Y3)
    ynew = _axpy(state, dt, m.b, [k0, k1, k2, k3])
    return ynew, k1                        # k1 = f(y_n) → next step's fprev


def _msrk3_step(rhs: Callable, m: _Method, state, fnm1, fnm2, dt: float):
    """One rk4_3 step.  carry = (y_n, f(y_{n-1}), f(y_{n-2}))."""
    k0 = fnm2
    k1 = fnm1
    k2 = rhs(state)
    Y3 = _axpy(state, dt, [m.a30, m.a31, m.a32], [k0, k1, k2])
    k3 = rhs(Y3)
    ynew = _axpy(state, dt, m.b, [k0, k1, k2, k3])
    return ynew, k2, fnm1                  # new f_{n-1}=f(y_n)=k2; new f_{n-2}=old f_{n-1}


def evolve(sim, state, dt: float, n_total: int, method: str):
    """Advance `state` by `n_total` steps of `method`, returning the final state.

    method='rk4' is the plain reference.  MSRK methods take `n_prev` RK4 startup
    steps (filling the previous-RHS history), then run the MSRK recurrence under
    lax.scan.  All RHS calls go through sim.rhs (so BC / KO / constraint damping
    are exactly the production RHS).
    """
    jax, jnp = _import_jax()
    rhs = sim.rhs

    if method == "rk4":
        def body(s, _):
            new, _k = rk4_step(rhs, s, dt)
            return new, None
        return jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_total)[0])(state)

    m = METHODS[method]
    n_prev = m.n_prev
    if n_total <= n_prev:                  # too short to engage MSRK → all RK4
        def body(s, _):
            new, _k = rk4_step(rhs, s, dt)
            return new, None
        return jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_total)[0])(state)

    def run(s0):
        # ── startup: n_prev RK4 steps, recording f(y_0..y_{n_prev-1}) ──
        s = s0
        hist = []                          # hist[k] = f(y_k), oldest first
        for _ in range(n_prev):
            s, fk = rk4_step(rhs, s, dt)
            hist.append(fk)
        # now s = y_{n_prev}; the freshest stored RHS is f(y_{n_prev-1}) = hist[-1]

        if n_prev == 1:
            fprev = hist[-1]               # f(y_0)
            def body(carry, _):
                y, fp = carry
                ynew, fnew = _msrk2_step(rhs, m, y, fp, dt)
                return (ynew, fnew), None
            (sf, _), _ = jax.lax.scan(body, (s, fprev), None,
                                      length=n_total - n_prev)
            return sf
        else:                              # n_prev == 2 (rk4_3)
            fnm1, fnm2 = hist[-1], hist[-2]   # f(y_1), f(y_0)
            def body(carry, _):
                y, f1, f2 = carry
                ynew, f1n, f2n = _msrk3_step(rhs, m, y, f1, f2, dt)
                return (ynew, f1n, f2n), None
            (sf, _, _), _ = jax.lax.scan(body, (s, fnm1, fnm2), None,
                                         length=n_total - n_prev)
            return sf

    return jax.jit(run)(state)
