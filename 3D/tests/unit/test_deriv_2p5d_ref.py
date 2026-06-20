"""Step 3.2e / M0 gate: the CPU 2.5D streaming derivative reference == derivative_bundle.

`deriv_2p5d_ref.streamed_derivative_bundle` is the host port of the plane-window march the CUDA
derivative kernel will run — horizontal T×T tiling with a halo, marching z with a resident
2·reach+1 plane window. M0's exit bar (per `step_3.2e_2p5d_geometry.md`) is that it reproduces
`derivative_bundle` to round-off. These tests pin that across slab widths and the awkward cases
the CUDA port will face: a single tile (T ≥ domain), ragged tiles (T ∤ domain), and the
production grid (ng=4 for 8th-order KO while the derivative stencil reach is only 3).

In practice the agreement is **bit-identical (0.0)** — the tiling partitions the domain exactly,
the halo is drawn from the same edge pad, and the mixed-derivative composition commutes with the
orthogonal edge pad — but the assertion uses the round-off bar (1e-12) to stay platform-robust.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from bssn3d.grid import Grid
from bssn3d.state import BSSNState
from bssn3d.derivative_bundle import derivative_bundle
from bssn3d.deriv_2p5d_ref import streamed_derivative_bundle
from mcs_common.derivatives import SpatialDerivative


@pytest.fixture(scope="module")
def case():
    g = Grid.from_domain(12, order=6)                 # ng=4 (8th-order KO), 6th-order FD
    diff = SpatialDerivative(order=6, ko_order=8)
    rng = np.random.default_rng(7)
    state = BSSNState(jnp.array(rng.standard_normal((24,) + g.shape)))
    ref = {k: np.asarray(v) for k, v in
           derivative_bundle(state, diff, g.dx, g.dy, g.dz).items()}
    return g, diff, state, ref


def test_bundle_has_138(case):
    _, _, _, ref = case
    assert len(ref) == 138


# T=8 (design slab), 13 (alt), 16 (≠ divisor), 20 (single tile = domain), 5 (ragged, T ∤ 20)
@pytest.mark.parametrize("T", [8, 13, 16, 20, 5])
def test_streamed_matches_bundle(case, T):
    g, diff, state, ref = case
    s = streamed_derivative_bundle(state, diff, g.dx, g.dy, g.dz, T)
    assert set(s) == set(ref)                          # same 138 keys
    worst = max(float(np.max(np.abs(ref[k] - s[k]))) for k in ref)
    assert worst <= 1e-12, f"T={T}: max|stream - bundle| = {worst:.3e}"


def test_single_tile_and_ragged_agree(case):
    """Tile reassembly is exact: a single tile covering the domain and ragged tiles that don't
    divide it produce identical derivatives (the seam/halo logic is bug-free)."""
    g, diff, state, _ = case
    single = streamed_derivative_bundle(state, diff, g.dx, g.dy, g.dz, 64)   # one tile
    ragged = streamed_derivative_bundle(state, diff, g.dx, g.dy, g.dz, 7)    # 7 ∤ 20
    worst = max(float(np.max(np.abs(single[k] - ragged[k]))) for k in single)
    assert worst <= 1e-12


def test_reach_is_derivative_order_not_grid_ng(case):
    """The derivative halo is the 6th-order stencil reach (3), independent of the grid ng=4 (the
    8th-order-KO width). A grid with MORE ghosts must not change the result."""
    g, diff, state, ref = case
    assert diff.ng == 4 and (diff.C1.shape[0] - 1) // 2 == 3
    s = streamed_derivative_bundle(state, diff, g.dx, g.dy, g.dz, 8)
    worst = max(float(np.max(np.abs(ref[k] - s[k]))) for k in ref)
    assert worst <= 1e-12
