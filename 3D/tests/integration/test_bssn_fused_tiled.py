"""Step 3.2d Increment 2 CPU gate: the fused TILED BSSN kernel == verbatim (interior).

``fused_tiled`` computes the 138 derivatives on-chip over power-of-2 tiles, reading each
field's overlapping ``HP**3`` window from the SINGLE padded grid via ``pl.ds`` (the
L2-served halo load proven to lower on H200, ``probe_overlap_load``), then runs the fp32
scheduled algebra — all in one ``pallas_call``. Pallas runs here in ``interpret=True``
(CPU math simulator), so these validate the **math**, not the SMEM/spill/regime payoff
(the H200 push, 3.2d gate).

Unlike the whole-grid ``fused`` scheme, the tiled kernel computes only the INTERIOR
(ghosts left 0; ``BSSNEvolution`` re-syncs them each substage) and recomputes the
derivatives via the broadcast-reduce matrix form — a different fp summation order than
``derivative_bundle`` — so the agreement is round-off (~1e-13), not bit-identity. We
compare on the interior with the same mixed abs/rel bar as the other fused gates.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams
from bssn3d.rhs import BSSNSolver
from bssn3d import initial_data as bid


def _interior(arr, ng):
    return arr[:, ng:-ng, ng:-ng, ng:-ng]


@pytest.mark.parametrize("t", [0.0, 0.3])
def test_fused_tiled_rhs_matches_verbatim_interior(t):
    """Full BSSNSolver A/B on a 3D gauge wave: fused_tiled (on-chip tiled FD + fp32-off
    fp64 algebra) == verbatim on the interior, across SSL ramp times. N=16, BS=8 → 2
    tiles/axis exercises the multi-tile reassembly + the overlapping halo windows."""
    order = 6
    ng = order // 2
    grid = Grid.from_domain(16, order=order)
    state = bid.gauge_wave(grid, amplitude=0.01)
    params = PhysicsParams()
    out_v = BSSNSolver(grid, params, order=order, scheme="verbatim").rhs(state, t=t)
    out_t = BSSNSolver(grid, params, order=order, scheme="fused_tiled").rhs(state, t=t)
    dv = _interior(np.asarray(out_v.data), ng)
    dt = _interior(np.asarray(out_t.data), ng)
    atol, rtol = 1e-11, 1e-9
    bad = np.abs(dv - dt) - (atol + rtol * np.abs(dv))
    assert np.all(bad <= 0), (
        f"fused_tiled vs verbatim (interior) exceeds atol={atol:.0e}+rtol={rtol:.0e}: "
        f"max excess {float(bad.max()):.2e}, max |Δ| {float(np.abs(dv-dt).max()):.2e}")


def test_fused_tiled_ghosts_are_zero():
    """The tiled kernel computes only the interior; the wrapper leaves ghosts at 0
    (evolve re-syncs them). Guards the documented drop-in contract."""
    order = 6
    ng = order // 2
    grid = Grid.from_domain(16, order=order)
    state = bid.gauge_wave(grid, amplitude=0.01)
    out_t = np.asarray(BSSNSolver(grid, None, order=order, scheme="fused_tiled").rhs(state).data)
    # a representative ghost slab (x-low) must be exactly zero
    assert np.all(out_t[:, :ng, :, :] == 0.0)
