"""Step 3.2e / M0 — CPU reference for the 2.5D streaming derivative stage.

This is the host/NumPy port of the **plane-window march** the CUDA derivative kernel will run
(`step_3.2e_2p5d_geometry.md`). It computes the same 138 derivatives as `derivative_bundle`, but
via the 2.5D geometry — horizontal `T×T` tiling with a halo, marching the z-axis with a resident
`2·reach+1` plane window — so the streaming/halo/composition logic is validated, in an easy-to-
debug environment, *before* any GPU time. M0's exit bar (per the plan) is exactly this: it equals
`derivative_bundle` to round-off.

Geometry (mirrors the kernel):
  * **z is the streamed axis.** For each output z-plane the kernel holds a window of `2·rz+1`
    planes; z-derivatives are the window stencil, in-plane derivatives use the window's centre.
  * **x,y are tiled** into `T×T` blocks, each carrying a `reach`-wide halo ring → the in-plane
    halo redundancy `((T+2·reach)/T)²` the cost model trades on. Tiles may be ragged at the
    domain edge (handled by a per-tile valid size).
  * **The halo is drawn from an edge-padded copy** of the field, so boundary tiles reproduce the
    `mode='edge'` padding `derivative_bundle` applies — interior tiles match the global stencil
    trivially (a local stencil partitions exactly), boundary tiles match via the same edge pad.

Key subtlety this validates: the **derivative reach is `order//2` (= 3 at 6th order), NOT the
grid `ng` (= 4 for the 8th-order KO)** — the derivative stage's halo is narrower than the grid's
ghost width. And **mixed second derivatives** compose `d1` along i then j; for z-mixed
(`grad2_0_2`, `grad2_1_2`) the in-plane pass is applied to *every* window plane before the
z-stencil, which (because the in-plane pass commutes with the orthogonal z edge-pad) equals
`derivative_bundle`'s two-step `compute_d1` composition.

`streamed_derivative_bundle(state, diff_op, dx, dy, dz, T)` returns the same dict as
`derivative_bundle`. Validated in `tests/unit/test_deriv_2p5d_ref.py`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ._bssn_rhs_generated import GRAD1_INPUTS, GRAD2_INPUTS
from .derivative_bundle import field_dict


def _reach(coeffs: np.ndarray) -> int:
    return (coeffs.shape[0] - 1) // 2


def _slice_axis(a: np.ndarray, start: int, length: int, axis: int) -> np.ndarray:
    idx: List = [slice(None)] * a.ndim
    idx[axis] = slice(start, start + length)
    return a[tuple(idx)]


def _apply_1d(a_padded: np.ndarray, coeffs: np.ndarray, factor: float, axis: int) -> np.ndarray:
    """Apply a 1-D centred stencil along ``axis`` of an array padded by the stencil's reach on
    that axis; return the valid region (``axis`` length shrinks by ``2·reach``). Summation runs
    k ascending, matching ``SpatialDerivative._apply``."""
    r = _reach(coeffs)
    n = a_padded.shape[axis] - 2 * r
    out = None
    for k in range(coeffs.shape[0]):
        term = float(coeffs[k]) * factor * _slice_axis(a_padded, k, n, axis)
        out = term if out is None else out + term
    return out


def streamed_derivative_bundle(state, diff_op, dx: float, dy: float, dz: float,
                               T: int) -> Dict[str, np.ndarray]:
    """2.5D field-streamed port of :func:`derivative_bundle`. Returns ``{deriv_name: array}``
    for all 138 referenced inputs, == ``derivative_bundle`` to round-off."""
    F = field_dict(state)
    C1 = np.asarray(diff_op.C1, dtype=np.float64)
    C2 = np.asarray(diff_op.C2, dtype=np.float64)
    r = max(_reach(C1), _reach(C2))            # derivative halo (3 at order 6) — NOT diff_op.ng
    inv = (1.0 / dx, 1.0 / dy, 1.0 / dz)
    inv2 = (1.0 / dx ** 2, 1.0 / dy ** 2, 1.0 / dz ** 2)

    # which derivatives each field needs (mirrors derivative_bundle's GRAD*_INPUTS iteration)
    g1_by_f: Dict[str, List[int]] = {}
    g2_by_f: Dict[str, List[Tuple[int, int]]] = {}
    for axis, f in GRAD1_INPUTS:
        g1_by_f.setdefault(f, []).append(axis)
    for i, j, f in GRAD2_INPUTS:
        g2_by_f.setdefault(f, []).append((i, j))

    out: Dict[str, np.ndarray] = {}
    fields = sorted(set(g1_by_f) | set(g2_by_f))
    for f in fields:
        arr = np.asarray(F[f], dtype=np.float64)
        Sx, Sy, Sz = arr.shape
        Pf = np.pad(arr, ((r, r), (r, r), (r, r)), mode="edge")   # edge-pad all axes by reach
        for axis in g1_by_f.get(f, []):
            out[f"grad_{axis}_{f}"] = np.empty((Sx, Sy, Sz))
        for (i, j) in g2_by_f.get(f, []):
            out[f"grad2_{i}_{j}_{f}"] = np.empty((Sx, Sy, Sz))

        for tx in range(0, Sx, T):
            for ty in range(0, Sy, T):
                Tx, Ty = min(T, Sx - tx), min(T, Sy - ty)
                # haloed tile: (Tx+2r) × (Ty+2r) in x,y; full padded z-extent
                tile = Pf[tx:tx + Tx + 2 * r, ty:ty + Ty + 2 * r, :]
                for k in range(Sz):                      # march the streamed z-axis
                    win = tile[:, :, k:k + 2 * r + 1]    # resident z-window (2r+1 planes)
                    ctr = win[:, :, r]                   # centre plane (in-plane derivs)

                    for axis in g1_by_f.get(f, []):
                        if axis == 0:
                            v = _apply_1d(ctr, C1, inv[0], 0)[:, r:r + Ty]
                        elif axis == 1:
                            v = _apply_1d(ctr, C1, inv[1], 1)[r:r + Tx, :]
                        else:
                            v = _apply_1d(win, C1, inv[2], 2)[r:r + Tx, r:r + Ty, 0]
                        out[f"grad_{axis}_{f}"][tx:tx + Tx, ty:ty + Ty, k] = v

                    for (i, j) in g2_by_f.get(f, []):
                        if i == j:                       # diagonal 2nd derivative
                            if i == 0:
                                v = _apply_1d(ctr, C2, inv2[0], 0)[:, r:r + Ty]
                            elif i == 1:
                                v = _apply_1d(ctr, C2, inv2[1], 1)[r:r + Tx, :]
                            else:
                                v = _apply_1d(win, C2, inv2[2], 2)[r:r + Tx, r:r + Ty, 0]
                        else:                            # mixed: d1 along i then d1 along j
                            if (i, j) == (0, 1):
                                v = _apply_1d(_apply_1d(ctr, C1, inv[0], 0), C1, inv[1], 1)
                            elif (i, j) == (0, 2):
                                v = _apply_1d(_apply_1d(win, C1, inv[0], 0),
                                              C1, inv[2], 2)[:, r:r + Ty, 0]
                            elif (i, j) == (1, 2):
                                v = _apply_1d(_apply_1d(win, C1, inv[1], 1),
                                              C1, inv[2], 2)[r:r + Tx, :, 0]
                            else:
                                raise ValueError(f"unexpected mixed index {(i, j)}")
                        out[f"grad2_{i}_{j}_{f}"][tx:tx + Tx, ty:ty + Ty, k] = v
    return out
