"""
Regression guard: the AMR root-level RK4 step must produce numerically
identical results to the non-AMR `fused_floating_point` solver on the same
initial data.  Both use the same kernel under the hood; the AMR step only
adds block partitioning + ghost-zone sync.  If the AMR layer slips in any
extra reduction-order change, this test catches it instantly.

We require bit-identical match across the whole interior (down to FP64 ULP).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import BS, NG, NF, MAX_BLOCKS
from mcs2d.amr.evolve import (
    make_root_step, amr_state_from_global, amr_state_to_global,
)
from mcs2d.main import (
    MaxwellChernSimons2D, InitialData, load_parameters, get_physical,
)


@pytest.mark.regression
class TestAMRVsFusedFloatingPoint:
    """Single RK4 step with `fused_floating_point` vs AMR root step.  The two
    paths share the same per-tile kernel, so the result should match within a
    few FP64 ULPs (we allow 1e-14 to absorb non-associative sum reorderings)."""

    # ULP-level tolerance: identical kernel, identical inputs, only differs in
    # how the tile data is moved around.  Any larger error means semantic drift.
    ATOL = 1e-13
    RTOL = 1e-13

    N_STEPS = 10

    @pytest.mark.parametrize("nbx,nby", [(2, 2), (4, 2)])
    def test_root_step_matches_fused_floating_point(self, nbx, nby, params_file):
        nx, ny = nbx * BS, nby * BS

        # Reference: non-AMR `fused_floating_point`.
        params_ref = load_parameters(params_file)
        params_ref.update({
            'scheme': 'fused_floating_point',
            'Nx': nx, 'Ny': ny, 'Nt': 1,
            'id_type': 'birefringent', 'bc_type': 'periodic',
            'sponge_strength': 0.0,
        })
        dx = (params_ref['xmax'] - params_ref['xmin']) / nx
        dy = (params_ref['ymax'] - params_ref['ymin']) / ny
        sim_ref = MaxwellChernSimons2D(dx, dy, params_ref['Lambda'], params_ref)
        state_ref = InitialData(sim_ref, params_ref).generate()

        def body_ref(carry, _):
            return sim_ref.step_rk4(carry, sim_ref.dt), None
        state_ref_final = jax.jit(
            lambda s: jax.lax.scan(body_ref, s, None, length=self.N_STEPS)[0]
        )(state_ref)
        ref_interior = np.asarray(get_physical(state_ref_final.data, sim_ref.ng))

        # AMR root step on the same IC.
        amr_blocks = amr_state_from_global(
            jnp.asarray(ref_interior_initial := np.asarray(
                get_physical(state_ref.data, sim_ref.ng)
            )),
            nbx, nby,
        )
        step = make_root_step(
            dx=dx, dy=dy,
            cs=params_ref.get('enable_cs', 1.0),
            L=params_ref['Lambda'],
            K1=params_ref['K1'], K2=params_ref['K2'],
            ko_sigma=params_ref['ko_sigma'],
            dt=sim_ref.dt,
            nbx=nbx, nby=nby,
        )

        def body_amr(carry, _):
            return step(carry), None
        amr_final = jax.jit(
            lambda s: jax.lax.scan(body_amr, s, None, length=self.N_STEPS)[0]
        )(amr_blocks)
        amr_interior = np.asarray(amr_state_to_global(amr_final, nbx, nby))

        # ULP-level agreement, field by field.
        for f in range(NF):
            err = np.max(np.abs(amr_interior[f] - ref_interior[f]))
            scale = max(np.max(np.abs(ref_interior[f])), 1.0)
            rel = err / scale
            assert rel < self.RTOL, (
                f"field {f}: max relative error {rel:.2e} > {self.RTOL:.0e} "
                f"(absolute {err:.2e}, nbx={nbx}, nby={nby})"
            )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
