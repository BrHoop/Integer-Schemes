"""
Restriction mechanics under 2-level evolution.

`make_two_level_step(restrict_at_end=True)` should produce a state where the
coarse cells covered by the fine block contain the 2×2 average of the matching
fine interior cells, bit-identical to applying `restrict_into_parent` directly
after the un-restricted step.

This does NOT check physics (constraint conservation / analytic match); see
the docstring on `make_two_level_step` for why naive restriction degrades the
physical solution at the coarse-fine boundary.  Flux correction is a Phase 3
deliverable.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import BS, NG, NF, MAX_BLOCKS
from mcs2d.amr.kernels import (
    sync_ghosts_within_level_root_periodic,
    prolongate,
)
from mcs2d.amr.evolve import make_two_level_step, amr_state_from_global
from mcs2d.main import MaxwellChernSimons2D, InitialData, load_parameters


def _setup_two_level(params_file, nbx=2, nby=2):
    """Build a coarse+fine state from the birefringent IC."""
    nx, ny = nbx * BS, nby * BS
    params = load_parameters(params_file)
    params.update({
        'scheme': 'floating_point', 'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'birefringent', 'bc_type': 'periodic',
        'sponge_strength': 0.0,
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, params['Lambda'], params)
    state = InitialData(sim, params).generate()
    interior = np.asarray(state.data[:, sim.ng:sim.ng+nx, sim.ng:sim.ng+ny])

    coarse_blocks = amr_state_from_global(jnp.asarray(interior), nbx, nby)
    coarse_synced = sync_ghosts_within_level_root_periodic(coarse_blocks, nbx, nby)
    child0 = prolongate(coarse_synced[0], (0, 0))
    fine_blocks = jnp.zeros(
        (MAX_BLOCKS, NF, BS + 2*NG, BS + 2*NG), dtype=jnp.float64
    ).at[0].set(child0)

    parent_slot = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
    child_cx    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
    child_cy    = jnp.zeros(MAX_BLOCKS, dtype=jnp.int32)
    fine_active = jnp.zeros(MAX_BLOCKS, dtype=bool).at[0].set(True)

    return (coarse_blocks, fine_blocks, sim, params,
            parent_slot, child_cx, child_cy, fine_active, nbx, nby, dx, dy)


@pytest.mark.amr
class TestRestrictAtEnd:
    """End-to-end restriction smoke test."""

    def test_restricted_quadrant_matches_fine_average(self, params_file):
        """After 1 step with `restrict_at_end=True`, the coarse parent's
        quadrant covered by the fine block must equal the 2×2 average of
        the fine block's evolved interior."""
        (cb, fb, sim, params, ps, ccx, ccy, fa, nbx, nby, dx, dy) = \
            _setup_two_level(params_file)
        cs = params.get('enable_cs', 1.0)
        L = params['Lambda']

        step_no_restrict = make_two_level_step(
            dx_coarse=dx, dy_coarse=dy, dt=params['cfl'] * dx / 2.0,
            cs=cs, L=L, K1=params['K1'], K2=params['K2'],
            ko_sigma=params['ko_sigma'],
            nbx_root=nbx, nby_root=nby,
            restrict_at_end=False,
        )
        step_restrict = make_two_level_step(
            dx_coarse=dx, dy_coarse=dy, dt=params['cfl'] * dx / 2.0,
            cs=cs, L=L, K1=params['K1'], K2=params['K2'],
            ko_sigma=params['ko_sigma'],
            nbx_root=nbx, nby_root=nby,
            restrict_at_end=True,
        )

        c_no,  f_no  = step_no_restrict(cb, fb, ps, ccx, ccy, fa)
        c_yes, f_yes = step_restrict   (cb, fb, ps, ccx, ccy, fa)

        # Fine INTERIOR is identical: restriction never touches fine data.
        # (Fine HALOS differ because the final cross-level sync sees a different
        # coarse-parent in the restricted run.)
        f_no_int  = np.asarray(f_no [0, :, NG:NG+BS, NG:NG+BS])
        f_yes_int = np.asarray(f_yes[0, :, NG:NG+BS, NG:NG+BS])
        assert np.array_equal(f_no_int, f_yes_int), \
            "restriction modified the fine interior"

        # Coarse parent (slot 0), quadrant (cx=0, cy=0): should equal 2×2 average
        # of fine slot 0's evolved interior.
        half_bs = BS // 2
        fine_int_evolved = np.asarray(f_yes[0, :, NG:NG+BS, NG:NG+BS])
        expected_quadrant = fine_int_evolved.reshape(
            NF, half_bs, 2, half_bs, 2
        ).mean(axis=(2, 4))
        got_quadrant = np.asarray(
            c_yes[0, :, NG:NG+half_bs, NG:NG+half_bs]
        )
        err = float(np.max(np.abs(got_quadrant - expected_quadrant)))
        assert err < 1e-13, (
            f"coarse-under-fine ≠ averaged fine; max err {err:.2e}"
        )

        # Cells NOT under the fine block: must match the no-restrict step (since
        # restriction only writes to the matching quadrant).
        # The fine block covers coarse slot 0, quadrant (0,0); compare every
        # other cell across the two runs.
        c_no_arr = np.asarray(c_no)
        c_yes_arr = np.asarray(c_yes)
        mask = np.ones(c_no_arr.shape, dtype=bool)
        mask[0, :, NG:NG+half_bs, NG:NG+half_bs] = False    # the covered quadrant
        # Halos get re-synced so they're not exactly equal between the two runs
        # (different coarse interiors → different periodic neighbors).  Restrict
        # the comparison to the INTERIOR cells outside the covered quadrant.
        interior_mask = np.zeros(c_no_arr.shape, dtype=bool)
        interior_mask[:, :, NG:NG+BS, NG:NG+BS] = True
        compare = mask & interior_mask
        # On all root slots EXCEPT slot 0, no cell is "covered"; on slot 0 only
        # the (0,0) quadrant is covered.  Compare interiors only.
        for slot in range(nbx * nby):
            slot_mask = compare[slot]
            diff = (c_yes_arr[slot] - c_no_arr[slot])[slot_mask]
            assert np.max(np.abs(diff)) < 1e-13, (
                f"slot {slot}: cells outside the covered quadrant changed "
                f"(max |Δ|={np.max(np.abs(diff)):.2e})"
            )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
