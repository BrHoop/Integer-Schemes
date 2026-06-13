"""Compact (Padé / Lele) finite-difference derivatives — *correctness probe*.

EXPLORATORY.  Implements 6th-order compact first/second derivatives as a
**truncated dense-band explicit stencil**, the GPU-deployable form of a compact
scheme: for periodic BCs the implicit operator is A^{-1}B, a fixed circulant
whose entries decay exponentially off-diagonal (ratio |r|=0.382 for the Lele
tridiagonal d1), so we precompute it once and truncate to a dense band of
half-width `band`.  No sequential tridiagonal solve — it drops straight into the
existing local-stencil machinery (`SpatialDerivative._apply`), just wider.

Why this is interesting for the thesis (see HANDOFF / CFD analysis): the
explicit 6th-order stencil cast as a GEMM is *banded* (width 7) → ~4.6× wasted
zeros on a dense tensor core (the rock the Ozaki-for-FD path was shelved on).
The compact operator is *dense* → no wasted zeros, full TC/Ozaki utilization,
AND spectral-like resolution (compact-6 out-resolves explicit-8 with a narrower
implicit stencil).  This file checks the dense-band form actually works in the
2D MCS solver: stable, matches the birefringent oracle, right convergence order.

Lele (1992) coefficients:
  d1: alpha=1/3,  a=14/9, b=1/9     (tridiagonal, 6th order)
  d2: alpha=2/11, a=12/11, b=3/11   (tridiagonal, 6th order)

Truncation note: truncating A^{-1}B to a fixed band introduces an
O(|r|^band) error that is INDEPENDENT of dx, so it becomes a convergence FLOOR
once dx^6 drops below it.  band=18 → ~3e-8, safely below dx^6 for nx<=128.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from mcs2d.schemes.floating_point import SpatialDerivative


# ── Lele coefficients ─────────────────────────────────────────────────────────
_D1 = dict(alpha=1.0 / 3.0, a=14.0 / 9.0, b=1.0 / 9.0)
_D2 = dict(alpha=2.0 / 11.0, a=12.0 / 11.0, b=3.0 / 11.0)

# Tridiagonal schemes as (alpha, [coeffs on (f_{i+k} -/+ ... f_{i-k})]):
#   d1: k'h = 2*sum_k c_k sin(kw)/(1+2a cos w)
#   d2: f''h^2 = 2*sum_k d_k (cos kw - 1)/(1+2a cos w)
# 6th order (reach +-2) and 8th order (reach +-3); 8th-order values derived from
# the moment conditions (see compact_stability._derive_d1/_derive_d2).
_C_D1 = {6: (1.0 / 3.0, [14.0 / 9.0 / 2.0, 1.0 / 9.0 / 4.0]),
         8: (3.0 / 8.0, [25.0 / 32.0, 1.0 / 20.0, -1.0 / 480.0])}
_C_D2 = {6: (2.0 / 11.0, [12.0 / 11.0, 3.0 / 11.0 / 4.0]),
         8: (0.2368421052631579,
             [0.9671052631578947, 0.13421052631578946,
              -0.003355263157894737])}

# Kreiss-Oliger stencils (delta^(2p)/2^(2p), with the DISSIPATIVE sign).
# 6th order (p=3): +delta^6/64 -> symbol -sin^6(w/2).
# 8th order (p=4): -delta^8/256 -> symbol -sin^8(w/2)  (sign FLIPS vs 6th, since
#   (delta^2)^p alternates sign — getting this wrong makes KO anti-dissipative).
_KO = {6: (np.array([1, -6, 15, -20, 15, -6, 1]) / 64.0, 3),
       8: (-np.array([1, -8, 28, -56, 70, -56, 28, -8, 1]) / 256.0, 4)}


def _circulant_solve_row(alpha, b_diag, b_off, *, n: int) -> np.ndarray:
    """Central row of A^{-1}B for periodic tridiagonal A and banded B.

    A = circ[..,alpha,1,alpha,..];  B given by its center/±1/±2 diagonals
    (dict offset->value in b_off plus b_diag at 0).  Returns the length-n
    central row (dimensionless stencil weights; the 1/dx factor is applied at
    runtime by `_apply`).
    """
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1.0
        A[i, (i - 1) % n] = alpha
        A[i, (i + 1) % n] = alpha
        B[i, i] = b_diag
        for off, val in b_off.items():
            B[i, (i + off) % n] = val
    M = np.linalg.solve(A, B)
    return M[n // 2]


def _banded_stencil(kind: str, band: int) -> np.ndarray:
    """Truncated, centered length-(2*band+1) stencil for d1 ('odd') or d2 ('even')."""
    n = 2 * band + 60                       # buffer so the central row converges
    if kind == "d1":
        c = _D1
        b_off = {1: c["a"] / 2.0, -1: -c["a"] / 2.0,
                 2: c["b"] / 4.0, -2: -c["b"] / 4.0}
        row = _circulant_solve_row(c["alpha"], 0.0, b_off, n=n)
    else:                                   # d2
        c = _D2
        b_diag = -2.0 * c["a"] - 0.5 * c["b"]
        b_off = {1: c["a"], -1: c["a"], 2: c["b"] / 4.0, -2: c["b"] / 4.0}
        row = _circulant_solve_row(c["alpha"], b_diag, b_off, n=n)
    mid = n // 2
    w = row[mid - band: mid + band + 1].copy()
    # Truncating A^{-1}B breaks the stencil's low-order moment conditions by
    # O(band*|r|^band), which appears as a CONSTANT (dx-independent) error in the
    # derivative scaling — a convergence floor.  Renormalise the dominant moment
    # so the truncated stencil differentiates exactly: Sum_d d*w = 1 (d1),
    # Sum_d d^2*w = 2 (d2).  This restores high-order convergence at a small band.
    d = np.arange(-band, band + 1)
    if kind == "d1":
        w /= np.sum(d * w)                  # enforce 1st moment = 1
    else:
        w *= 2.0 / np.sum(d ** 2 * w)       # enforce 2nd moment = 2
    return w


def truncation_band_for_order(kind: str = "d1", target: float = 1e-10) -> int:
    """Smallest dense-band half-width whose moment-truncation tail < target.

    Quantifies the FINDING that the dense-band-stencil shortcut is impractical:
    the d^6 * |r|^d moment tail decays so slowly that holding 6th order needs
    band ~ 40 (an 81-wide stencil).  Compact's GPU-deployable form is therefore
    the FULL dense circulant operator (a genuine dense GEMM, what CompactDerivative
    applies via FFT below), not a truncated wide stencil.
    """
    c = _D1 if kind == "d1" else _D2
    b_off = ({1: c["a"] / 2, -1: -c["a"] / 2, 2: c["b"] / 4, -2: -c["b"] / 4}
             if kind == "d1"
             else {1: c["a"], -1: c["a"], 2: c["b"] / 4, -2: c["b"] / 4})
    b_diag = 0.0 if kind == "d1" else -2 * c["a"] - 0.5 * c["b"]
    n = 400
    row = _circulant_solve_row(c["alpha"], b_diag, b_off, n=n)
    mid = n // 2
    for b in range(4, mid):
        d = np.arange(b + 1, mid)
        if 2 * np.sum((d ** 6) * np.abs(row[mid + d])) < target:
            return b
    return mid


class CompactDerivative(SpatialDerivative):
    """6th-order compact (Lele) d1/d2 — the FULL periodic operator A^{-1}B.

    Drop-in for SpatialDerivative (same compute_d1/d2/ko interface).  Periodic
    BC only: A^{-1}B is circulant, so it is applied exactly and cheaply by FFT
    (eigenvalues B_hat(k)/A_hat(k)) — equivalent to the dense circulant GEMM that
    is the real GPU target, but O(N log N) on CPU for this probe.  Exactly 6th
    order, ~15x lower error constant than explicit-6.

    KO dissipation stays the standard explicit 6th-order operator (a filter, not
    a derivative), so ng = 3 suffices.
    """

    def __init__(self, order: int = 6, ko_order: int = None):
        if order not in (6, 8):
            raise NotImplementedError("compact scheme supports order 6 or 8")
        self.order = order
        self.ko_order = ko_order if ko_order is not None else order
        self._d1 = _C_D1[order]              # (alpha, [c_k])
        self._d2 = _C_D2[order]              # (alpha, [d_k])
        cko, ko_ng = _KO[self.ko_order]
        self.CKO = jnp.asarray(cko)
        self.ng = ko_ng                      # ghost cells = KO footprint (FFT op needs none)

    def _eig_d1(self, n: int) -> jnp.ndarray:
        th = 2.0 * jnp.pi * jnp.arange(n) / n
        al, cs = self._d1
        num = 2.0 * sum(c * jnp.sin((i + 1) * th) for i, c in enumerate(cs))
        return 1j * num / (1 + 2 * al * jnp.cos(th))

    def _eig_d2(self, n: int) -> jnp.ndarray:
        th = 2.0 * jnp.pi * jnp.arange(n) / n
        al, ds = self._d2
        num = 2.0 * sum(d * (jnp.cos((i + 1) * th) - 1.0) for i, d in enumerate(ds))
        return num / (1 + 2 * al * jnp.cos(th))

    def _fft_diff(self, grid, dx, axis, eig, power):
        """Apply the circulant compact operator to the periodic interior."""
        from jax import lax
        n = grid.shape[axis] - 2 * self.ng
        interior = lax.slice_in_dim(grid, self.ng, self.ng + n, axis=axis)
        lam = eig(n)
        shape = [1] * grid.ndim
        shape[axis] = n
        lam = lam.reshape(shape)
        Fhat = jnp.fft.fft(interior, axis=axis)
        out = jnp.real(jnp.fft.ifft(lam * Fhat, axis=axis)) / dx ** power
        pad = [(self.ng, self.ng) if i == axis else (0, 0)
               for i in range(grid.ndim)]
        return jnp.pad(out, pad, mode="edge")

    def compute_d1(self, grid, dx, axis):
        return self._fft_diff(grid, dx, axis, self._eig_d1, 1)

    def compute_d2(self, grid, dx, axis):
        return self._fft_diff(grid, dx, axis, self._eig_d2, 2)
