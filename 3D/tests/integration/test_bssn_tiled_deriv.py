"""Increment 1 of the fp32 tiled fused kernel (3.2d-GPU): the on-chip derivative core.

`tiled_deriv.tiled_derivative_bundle` computes the 138 derivatives over power-of-2 haloed
tiles via the proven `pallas_ozaki` Triton pattern (stencil-as-GEMM + one-hot crop), the
prerequisite to fusing the algebra on top. Interpret mode validates the MATH; the Triton
LOWERING (power-of-2, fp32 dot, transpose) is the H200 gate (`python -m bssn3d.tiled_deriv`
on a GPU node).

  * fp64 → bit-exact vs `derivative_bundle` (the tiling/stencil/crop is correct);
  * fp32 → bounded, with the error concentrated in the cancellation-sensitive 2nd
    derivatives (fp64 `dot` is banned in Triton, so the in-kernel stencil-GEMM is forced
    to fp32 — the precision gap the Phase-4 Ozaki-II derivative is meant to close).
"""

import os

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from mcs_common.derivatives import SpatialDerivative
from bssn3d.grid import Grid
from bssn3d.derivative_bundle import derivative_bundle
from bssn3d import initial_data as bid


def _run(fp32: bool):
    # tiled_deriv reads BSSN_PALLAS_FP32 at import → set before importing fresh.
    os.environ["BSSN_PALLAS_FP32"] = "1" if fp32 else "0"
    import importlib
    import bssn3d.tiled_deriv as td
    importlib.reload(td)
    order = 6
    g = Grid.from_domain(16, order=order)            # N=16, BS=8 → 2 tiles/axis
    s = bid.gauge_wave(g, amplitude=0.02)
    ref = derivative_bundle(s, SpatialDerivative(order=order), g.dx, g.dy, g.dz)
    got = td.tiled_derivative_bundle(s, order, g.dx, g.dy, g.dz)
    ng = order // 2
    w1 = w2 = 0.0
    for name in td.DERIV_NAMES:
        r = np.asarray(ref[name])[ng:-ng, ng:-ng, ng:-ng]
        d = float(np.max(np.abs(r - np.asarray(got[name]))))
        if name.startswith("grad2"):
            w2 = max(w2, d)
        else:
            w1 = max(w1, d)
    return w1, w2


def test_tiled_deriv_fp64_exact():
    """fp64: the tiling + stencil-as-GEMM + one-hot crop reproduces derivative_bundle."""
    w1, w2 = _run(fp32=False)
    assert max(w1, w2) < 1e-12, f"fp64 tiled deriv off by {max(w1, w2):.2e}"


def test_tiled_deriv_fp32_bounded():
    """fp32: bounded, and the error lives in the 2nd derivatives (cancellation)."""
    w1, w2 = _run(fp32=True)
    assert w1 < 1e-4, f"fp32 1st-deriv error {w1:.2e}"
    assert w2 < 1e-3, f"fp32 2nd-deriv error {w2:.2e}"
    assert w2 > w1, "expected the 2nd derivatives to carry the fp32 cancellation error"
