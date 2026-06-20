"""Step 3.2d probe — overlapping L2-served halo load, and the deriv-EXPRESSION A/B.

Two questions, one file:

1. **Load mechanism (resolved, YES):** can each block read its overlapping window from a
   SINGLE padded grid (``memory_space=pl.ANY`` ref + ``pl.ds(program_id*BS, …)``) so the
   field grid is one HBM copy and the cube-halo is L2-served? → lowers + bit-exact on H200.

2. **Derivative expression (the compile-time A/B):** the original ``bmr`` mode computes each
   derivative by loading ONE ``HP**3`` window and doing a broadcast-multiply-reduce — which
   materializes a 4D transient and was measured to compile **~quadratically** in unrolled
   deriv count (24 derivs = 256 s on H200 → 138 ≈ 2 h). ``shift`` mode computes the SAME FD
   derivative as a weighted sum of ``2*NG+1`` shifted ``BS**3`` ``pl.ds`` loads — only the
   proven ref-load, **no 4D transient** (every value is BS**3). Hypothesis: ptxas register
   pressure collapses → near-linear compile. ``BSSN_PROBE_MODE=bmr|shift`` selects.

CPU ``interpret`` checks the math; the decisive bit (compile time + lowering) is the GPU run.

    BSSN_PROBE_MODE=shift BSSN_PROBE_NDERIV=24 python -m bssn3d.probe_overlap_load --gpu
    BSSN_PROBE_MODE=bmr   BSSN_PROBE_NDERIV=24 python -m bssn3d.probe_overlap_load --gpu
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from mcs_common.derivatives import SpatialDerivative
from ._bssn_rhs_generated import FIELD_INPUTS, GRAD1_INPUTS, GRAD2_INPUTS
from .tiled_deriv import _axis_matrices, _derivative, _OPS, _ALL_DERIV_NAMES

if not os.environ.get("JAX_PALLAS_USE_MOSAIC_GPU"):
    jax.config.update("jax_pallas_use_mosaic_gpu", False)

# ORDER=8 → NG=4 → HP=BS+2*NG=16 (pow2, zero pad), the realistic operating point.
ORDER = int(os.environ.get("BSSN_PROBE_ORDER", "8"))
BS = int(os.environ.get("BSSN_PROBE_BS", "8"))
NDERIV = int(os.environ.get("BSSN_PROBE_NDERIV", "6"))
MODE = os.environ.get("BSSN_PROBE_MODE", "bmr")          # "bmr" | "shift"
DERIV_NAMES = _ALL_DERIV_NAMES[:NDERIV]
_NG = ORDER // 2

# Centred FD coeffs as plain Python floats (a pallas kernel may not capture jax arrays).
_diff = SpatialDerivative(order=ORDER)
_C1 = tuple(float(x) for x in np.asarray(_diff.C1))      # length 2*NG+1
_C2 = tuple(float(x) for x in np.asarray(_diff.C2))


# --------------------------------------------------------------------------- #
# bmr mode — one HP window per deriv + broadcast-multiply-reduce (4D transient)
# --------------------------------------------------------------------------- #
def _kernel_bmr(grid_ref, mat_ref, out_ref):
    bx = pl.program_id(0)
    by = pl.program_id(1)
    bz = pl.program_id(2)
    HP = BS + 2 * _NG
    mats = [mat_ref[r] for r in range(7)]
    for k, name in enumerate(DERIV_NAMES):
        f, axis_ops = _OPS[name]
        fidx = FIELD_INPUTS.index(f)
        win = grid_ref[fidx, pl.ds(bx * BS, HP), pl.ds(by * BS, HP), pl.ds(bz * BS, HP)]
        out_ref[k] = _derivative(win, mats, axis_ops)


# --------------------------------------------------------------------------- #
# shift mode — 2*NG+1 shifted BS**3 pl.ds loads, weighted-summed (no transient)
# --------------------------------------------------------------------------- #
def _shift_taps():
    """Per-deriv: (field, [(coeff, (o0,o1,o2)), ...], invd_axes). coeff excludes 1/d**n
    (runtime); invd_axes says which inverse-spacings multiply (grad1→(a,), grad2_ii→(i,i),
    grad2_ij→(i,j))."""
    ops = {}
    for a, f in GRAD1_INPUTS:
        taps = []
        for m in range(2 * _NG + 1):
            off = [0, 0, 0]; off[a] = m - _NG
            taps.append((_C1[m], tuple(off)))
        ops[f"grad_{a}_{f}"] = (f, taps, (a,))
    for i, j, f in GRAD2_INPUTS:
        taps = []
        if i == j:
            for m in range(2 * _NG + 1):
                off = [0, 0, 0]; off[i] = m - _NG
                taps.append((_C2[m], tuple(off)))
            invd = (i, i)
        else:                                            # mixed: outer product of two D1
            for mi in range(2 * _NG + 1):
                for mj in range(2 * _NG + 1):
                    off = [0, 0, 0]; off[i] = mi - _NG; off[j] = mj - _NG
                    taps.append((_C1[mi] * _C1[mj], tuple(off)))
            invd = (i, j)
        ops[f"grad2_{i}_{j}_{f}"] = (f, taps, invd)
    return ops


_SHIFT_OPS = _shift_taps()


def _kernel_shift(grid_ref, invd_ref, out_ref):
    # base data index of this tile's interior (grid ghosts == NG since order matches grid)
    b0 = _NG + pl.program_id(0) * BS
    b1 = _NG + pl.program_id(1) * BS
    b2 = _NG + pl.program_id(2) * BS
    invd = [invd_ref[0], invd_ref[1], invd_ref[2]]
    for k, name in enumerate(DERIV_NAMES):
        f, taps, invd_axes = _SHIFT_OPS[name]
        fidx = FIELD_INPUTS.index(f)
        scale = invd[invd_axes[0]]
        for ax in invd_axes[1:]:
            scale = scale * invd[ax]
        acc = None
        for c, off in taps:
            w = grid_ref[fidx, pl.ds(b0 + off[0], BS), pl.ds(b1 + off[1], BS),
                         pl.ds(b2 + off[2], BS)]
            term = c * w
            acc = term if acc is None else acc + term
        out_ref[k] = scale * acc


def overlap_load_bundle(state, dx, dy, dz):
    """Compute the first NDERIV derivatives via the selected MODE → {name: (Nx,Ny,Nz)}."""
    NG = _NG
    data = state.data                                    # (NF, Sx, Sy, Sz), ghosts = NG
    NF, Sx, Sy, Sz = data.shape
    N = (Sx - 2 * NG, Sy - 2 * NG, Sz - 2 * NG)
    assert all(n % BS == 0 for n in N), f"interior {N} not divisible by BS={BS}"
    nt = [n // BS for n in N]
    interpret = jax.default_backend() == "cpu"
    common = dict(
        grid=(nt[0], nt[1], nt[2]),
        out_specs=pl.BlockSpec((NDERIV, BS, BS, BS), lambda i, j, k: (0, i, j, k)),
        out_shape=jax.ShapeDtypeStruct((NDERIV, N[0], N[1], N[2]), jnp.float64),
        interpret=interpret,
    )
    if MODE == "shift":
        invd = jnp.asarray([1.0 / dx, 1.0 / dy, 1.0 / dz], dtype=jnp.float64)
        out = pl.pallas_call(
            _kernel_shift,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY),
                      pl.BlockSpec((3,), lambda i, j, k: (0,))],
            **common,
        )(data, invd)
    else:
        HP = BS + 2 * NG
        mats = jnp.asarray(_axis_matrices(ORDER, (dx, dy, dz), BS, HP, NG),
                           dtype=jnp.float64)
        out = pl.pallas_call(
            _kernel_bmr,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY),
                      pl.BlockSpec((7, BS, HP), lambda i, j, k: (0, 0, 0))],
            **common,
        )(data, mats)
    return {name: out[k] for k, name in enumerate(DERIV_NAMES)}


def _check(gpu: bool):
    jax.config.update("jax_enable_x64", True)
    from .grid import Grid
    from . import initial_data as bid
    from .derivative_bundle import derivative_bundle
    g = Grid.from_domain(16, order=ORDER)                 # N=16, BS=8 → 2 tiles/axis
    s = bid.gauge_wave(g, amplitude=0.02)
    diff = SpatialDerivative(order=ORDER)
    ref = derivative_bundle(s, diff, g.dx, g.dy, g.dz)
    got = overlap_load_bundle(s, g.dx, g.dy, g.dz)
    worst = 0.0
    for name in DERIV_NAMES:
        r = np.asarray(ref[name])[_NG:-_NG, _NG:-_NG, _NG:-_NG]
        worst = max(worst, float(np.max(np.abs(r - np.asarray(got[name])))))
    where = "GPU (Triton lowered)" if gpu else "CPU interpret (math/trace only)"
    print(f">> overlap_load_bundle [MODE={MODE}] vs derivative_bundle: "
          f"max|Δ| = {worst:.2e}  [{where}]")
    print(f"   ORDER={ORDER} NG={_NG} BS={BS} NDERIV={NDERIV}")
    if not gpu:
        print("   NOTE: interpret mode does NOT test compile time/lowering — run --gpu and "
              "time bmr vs shift to see if the shifted stencil kills the quadratic compile.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true",
                    help="(Marylou) actually compile+run on GPU — the compile-time A/B")
    a = ap.parse_args()
    _check(a.gpu)
