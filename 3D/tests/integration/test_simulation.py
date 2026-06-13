"""
Integration tests: multi-step 3D evolution against the analytic oracle (Phase 1).

These exercise the full pipeline (IC -> RHS -> RK4 -> BC) end-to-end and assert
the evolved solution tracks the 3D birefringent oracle, stays bounded, and keeps
the constraints at round-off over a real (if short) run.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from mcs3d.validate import run_sim, field_l2_errors
from mcs3d.main import get_physical, calc_constraints, l2norm


class TestOracleEvolution:
    def test_tracks_oracle_over_a_run(self):
        """A 60-step birefringent evolution must stay close to the analytic oracle
        in every propagating field (small, resolution-appropriate error)."""
        N, STEPS = 24, 60
        sim, state, params = run_sim("floating_point", N, N, N, STEPS, Lambda=0.4)
        errs = field_l2_errors(sim, state, params, STEPS * sim.dt)
        for f in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
            assert errs[f] < 1e-3, f"{f} drifted from oracle: L2 = {errs[f]:.2e}"
        assert errs["xi"] < 1e-7, f"xi tracking error {errs['xi']:.2e}"

    def test_no_blowup_and_constraints_clean(self):
        """The run must remain bounded and divergence-free to round-off."""
        N, STEPS = 24, 60
        sim, state, _ = run_sim("floating_point", N, N, N, STEPS, Lambda=0.4)
        data = np.asarray(get_physical(state.data, sim.ng))
        assert np.all(np.isfinite(data)), "non-finite values in evolved state"
        assert np.max(np.abs(data[:6])) < 5.0, "fields grew unphysically"
        divE, divB = calc_constraints(sim, state)
        assert float(l2norm(get_physical(divE, sim.ng))) < 1e-12
        assert float(l2norm(get_physical(divB, sim.ng))) < 1e-12


# NOTE: a dynamical CFJ-tachyon growth test is deliberately NOT here.  The
# Carroll-Field-Jackiw instability is a property of the linearization at the
# Pi = Pi_0 background (where the CS coupling becomes a mass term); the
# birefringent IC sits on the always-stable omega=sqrt(k^2+m_cs k) branch and a
# zero-background seed has the CS coupling switched OFF entirely.  Exciting the
# tachyon cleanly requires reconstructing the unstable eigenmode at the Pi_0
# background -- fragile, and redundant with the rigorous spectral certificate in
# tests/validation/test_spectrum.py::TestModeStructure (growing mode below m_cs,
# none above).  That eigenvalue statement is the correct home for the claim.


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
