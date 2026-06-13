"""
Constraint conservation under AMR evolution.

The MCS PDE preserves two constraints:
   divE = ∂Ex/∂x + ∂Ey/∂y + 2·cs·Λ·(Bx·∂xi/∂x + By·∂xi/∂y) = 0
   divB = ∂Bx/∂x + ∂By/∂y                                    = 0

With constraint damping (K1, K2 > 0) the evolution suppresses any drift.  For
the birefringent IC both constraints start at machine zero and must stay
bounded over many steps.

If AMR mishandles ghost-zone data, the constraints blow up — so this is the
most sensitive test of the AMR layer for physical correctness.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import BS, NG, NF, MAX_BLOCKS
from mcs2d.amr.kernels import (
    sync_ghosts_within_level_root_periodic,
    sync_ghosts_across_levels,
    prolongate,
)
from mcs2d.amr.evolve import (
    make_root_step, make_two_level_step,
    amr_state_from_global, amr_state_to_global,
)
from mcs2d.main import (
    MaxwellChernSimons2D, InitialData, load_parameters,
)


EX, EY, EZ = 0, 1, 2
BX, BY, BZ = 3, 4, 5
XI = 6


# ── 6th-order centred FD helpers (periodic) ──────────────────────────────────
# These mirror the kernel's 6th-order C1 coefficients exactly so the divergence
# we compute is the SAME quantity the PDE itself sees.

_C1 = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0], dtype=np.float64) / 60.0


def _d_axis_periodic(u, dx, axis):
    """6th-order centred first derivative along `axis` with periodic BC."""
    out = np.zeros_like(u)
    for k, c in enumerate(_C1):
        if c == 0.0:
            continue
        shift = k - 3   # stencil index 0 ↔ shift -3
        out += c * np.roll(u, -shift, axis=axis)
    return out / dx


def _divE(state_interior, dx, dy, cs, L):
    """Constraint-damped divE on a periodic interior of shape (NF, Nx, Ny)."""
    Ex = state_interior[EX]; Ey = state_interior[EY]
    Bx = state_interior[BX]; By = state_interior[BY]
    xi = state_interior[XI]
    return (
        _d_axis_periodic(Ex, dx, axis=0) + _d_axis_periodic(Ey, dy, axis=1)
        + 2.0 * cs * L * (Bx * _d_axis_periodic(xi, dx, axis=0)
                          + By * _d_axis_periodic(xi, dy, axis=1))
    )


def _divB(state_interior, dx, dy):
    Bx = state_interior[BX]; By = state_interior[BY]
    return _d_axis_periodic(Bx, dx, axis=0) + _d_axis_periodic(By, dy, axis=1)


def _make_birefringent_ic(nx, ny, params_file):
    """Standard birefringent IC via main.InitialData."""
    params = load_parameters(params_file)
    params.update({
        'scheme': 'floating_point',
        'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'birefringent', 'bc_type': 'periodic',
        'sponge_strength': 0.0,
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, params['Lambda'], params)
    state = InitialData(sim, params).generate()
    return sim, state, params


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.amr
class TestConstraintsRootOnly:
    """Root-only AMR must conserve constraints to the same level as the
    non-AMR fused solver (since the math is identical — see test_amr_vs_fused)."""

    N_STEPS = 100
    DIV_TOL = 1e-7

    def test_root_amr_birefringent_constraints(self, params_file):
        nbx, nby = 2, 2
        nx, ny = nbx * BS, nby * BS
        sim, state, params = _make_birefringent_ic(nx, ny, params_file)
        interior = np.asarray(state.data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny])

        # Initial constraints: confirm IC is constraint-satisfying.
        dx, dy = sim.dx, sim.dy
        cs = params.get('enable_cs', 1.0); L = params['Lambda']
        dE0 = _divE(interior, dx, dy, cs, L)
        dB0 = _divB(interior, dx, dy)
        assert np.max(np.abs(dE0)) < 1e-10, f"IC divE = {np.max(np.abs(dE0)):.2e}"
        assert np.max(np.abs(dB0)) < 1e-12, f"IC divB = {np.max(np.abs(dB0)):.2e}"

        # Evolve.
        blocks = amr_state_from_global(jnp.asarray(interior), nbx, nby)
        step = make_root_step(
            dx=dx, dy=dy, cs=cs, L=L,
            K1=params['K1'], K2=params['K2'], ko_sigma=params['ko_sigma'],
            dt=sim.dt, nbx=nbx, nby=nby,
        )
        def body(carry, _):
            return step(carry), None
        blocks_final = jax.jit(
            lambda s: jax.lax.scan(body, s, None, length=self.N_STEPS)[0]
        )(blocks)
        interior_final = np.asarray(amr_state_to_global(blocks_final, nbx, nby))

        dEf = _divE(interior_final, dx, dy, cs, L)
        dBf = _divB(interior_final, dx, dy)
        assert np.max(np.abs(dEf)) < self.DIV_TOL, (
            f"divE blew up: {np.max(np.abs(dEf)):.2e} > {self.DIV_TOL:.0e}"
        )
        assert np.max(np.abs(dBf)) < self.DIV_TOL, (
            f"divB blew up: {np.max(np.abs(dBf)):.2e} > {self.DIV_TOL:.0e}"
        )


@pytest.mark.amr
class TestConstraintsTwoLevel:
    """2-level AMR with a single fine block: constraints must stay bounded on
    BOTH the coarse global grid AND the fine block interior.

    Note: Phase 1 has no restriction wired in, so coarse cells under the fine
    block evolve independently of the fine block.  Constraints are checked
    separately on the coarse field and on the fine interior."""

    N_STEPS = 50            # halved (fine CFL halves dt)
    DIV_TOL_COARSE = 1e-7
    # Fine block uses cross-level interpolation for halo data, which carries
    # O(dx_coarse^6) error.  That error propagates inward via the 6-cell
    # stencil over many timesteps.  Phase 2 (which adds restriction) and
    # Phase 3 (sub-cycling) will tighten this.
    DIV_TOL_FINE   = 1e-4

    def test_two_level_constraints(self, params_file):
        nbx, nby = 2, 2
        nx, ny = nbx * BS, nby * BS
        sim, state, params = _make_birefringent_ic(nx, ny, params_file)
        interior = np.asarray(state.data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny])
        dx, dy = sim.dx, sim.dy
        dx_fine, dy_fine = dx / 2.0, dy / 2.0
        cs = params.get('enable_cs', 1.0); L = params['Lambda']

        coarse_blocks = amr_state_from_global(jnp.asarray(interior), nbx, nby)

        # One fine child at slot 0 of fine level → corner (0, 0) of root slot 0.
        parent_slot = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        child_cx    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        child_cy    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
        fine_active = jnp.zeros(MAX_BLOCKS, dtype=bool).at[0].set(True)

        coarse_synced = sync_ghosts_within_level_root_periodic(
            coarse_blocks, nbx, nby
        )
        child0 = prolongate(coarse_synced[0], (0, 0))
        fine_blocks = jnp.zeros(
            (MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64
        ).at[0].set(child0)

        dt = params['cfl'] * dx_fine
        step = make_two_level_step(
            dx_coarse=dx, dy_coarse=dy, dt=dt,
            cs=cs, L=L, K1=params['K1'], K2=params['K2'],
            ko_sigma=params['ko_sigma'],
            nbx_root=nbx, nby_root=nby,
        )
        def body(carry, _):
            c, f = carry
            c2, f2 = step(c, f, parent_slot, child_cx, child_cy, fine_active)
            return (c2, f2), None
        (cf, ff), _ = jax.jit(
            lambda c, f: jax.lax.scan(body, (c, f), None, length=self.N_STEPS)
        )(coarse_blocks, fine_blocks)

        # Constraints on the global coarse interior.
        coarse_interior = np.asarray(amr_state_to_global(cf, nbx, nby))
        dEc = _divE(coarse_interior, dx, dy, cs, L)
        dBc = _divB(coarse_interior, dx, dy)
        assert np.max(np.abs(dEc)) < self.DIV_TOL_COARSE, (
            f"coarse divE = {np.max(np.abs(dEc)):.2e} > {self.DIV_TOL_COARSE:.0e}"
        )
        assert np.max(np.abs(dBc)) < self.DIV_TOL_COARSE, (
            f"coarse divB = {np.max(np.abs(dBc)):.2e} > {self.DIV_TOL_COARSE:.0e}"
        )

        # Constraints on the fine block interior (use 6th-order FD with dx_fine).
        # Note: the fine block is NOT periodic by itself; using _divE here gives
        # spurious wrap-around at the fine-block edges.  Restrict the divergence
        # check to the interior away from the fine-block boundary (drop NG on each
        # side from the divergence array) — those interior points have valid 6-point
        # stencils that don't touch the (wrong) periodic-wrap from np.roll.
        fine_interior_full = np.asarray(ff[0, :, NG:NG+BS, NG:NG+BS])
        dEf = _divE(fine_interior_full, dx_fine, dy_fine, cs, L)
        dBf = _divB(fine_interior_full, dx_fine, dy_fine)
        margin = NG + 1
        dEf_inner = dEf[margin:-margin, margin:-margin]
        dBf_inner = dBf[margin:-margin, margin:-margin]
        assert np.max(np.abs(dEf_inner)) < self.DIV_TOL_FINE, (
            f"fine divE = {np.max(np.abs(dEf_inner)):.2e} > {self.DIV_TOL_FINE:.0e}"
        )
        assert np.max(np.abs(dBf_inner)) < self.DIV_TOL_FINE, (
            f"fine divB = {np.max(np.abs(dBf_inner)):.2e} > {self.DIV_TOL_FINE:.0e}"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
