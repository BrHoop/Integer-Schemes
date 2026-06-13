"""
Boundary-condition correctness for the MCS 3D solver (Phase 1).

The 3D mirror of `2D/tests/validation/test_boundaries.py`.  The accuracy/spectral
suites all use PERIODIC boundaries; this file is the only coverage of the other
BC code paths.  A boundary condition must never be a SOURCE -- it can remove
energy (absorbing/radiative) or conserve it (periodic), but it must not inject
energy.  Energy injection is exactly the "boundary error" that contaminates a BBH
waveform extraction, so we test for it directly.

  * periodic   : exact ghost wrap; conserves energy to round-off (no injection).
  * sommerfeld : passive (energy non-increasing) and genuinely absorbing.

Grids are kept small (N^3 cost); the heavier absorption run is marked slow.
"""

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs3d.main import (
    MaxwellChernSimons3D, InitialData, load_parameters, get_physical,
)

_PF = str(Path(__file__).resolve().parents[2] / "params.toml")


def _gaussian_sim(bc, N=24, *, cs=0.0, ko=0.0, **over):
    """Pure-Maxwell (cs=0) Gaussian-pulse sim with the requested BC.  Pure Maxwell
    isolates the boundary behaviour from CS/constraint-damping energy changes."""
    p = load_parameters(_PF)
    p.update({"scheme": "floating_point", "Nx": N, "Ny": N, "Nz": N,
              "id_type": "gaussian", "bc_type": bc, "enable_cs": cs,
              "ko_sigma": ko, "K1": 0.0, "K2": 0.0, "sponge_strength": 0.0,
              "id_sigma": 1.0, **over})
    dx = (p["xmax"] - p["xmin"]) / N
    sim = MaxwellChernSimons3D(dx, dx, dx, p["Lambda"], p)
    state = InitialData(sim, p).generate()
    return sim, state


def _energy_trace(sim, state, n_steps, every=10):
    """Total interior field 'energy' (sum of squares) sampled every `every`."""
    step = jax.jit(lambda s: sim.step_rk4(s, sim.dt))
    E = []
    for i in range(n_steps + 1):
        if i % every == 0:
            d = np.asarray(get_physical(state.data, sim.ng))
            E.append(float(np.sum(d ** 2)))
        state = step(state)
    return np.array(E)


# ── 1. Periodic: exact and conservative ───────────────────────────────────────

class TestPeriodic:
    def test_ghost_wrap_is_exact(self):
        """bc_periodic must copy the wrapped interior into the ghost zones
        bit-exactly -- the periodic image, no interpolation error."""
        sim, state = _gaussian_sim("periodic", N=16)
        f = np.asarray(state.data[0])
        wrapped = np.asarray(sim.bc_periodic(jnp.asarray(f)))
        ng = sim.ng
        assert np.array_equal(wrapped[:ng, :, :], wrapped[-2 * ng:-ng, :, :])
        assert np.array_equal(wrapped[-ng:, :, :], wrapped[ng:2 * ng, :, :])
        assert np.array_equal(wrapped[:, :ng, :], wrapped[:, -2 * ng:-ng, :])
        assert np.array_equal(wrapped[:, -ng:, :], wrapped[:, ng:2 * ng, :])
        assert np.array_equal(wrapped[:, :, :ng], wrapped[:, :, -2 * ng:-ng])
        assert np.array_equal(wrapped[:, :, -ng:], wrapped[:, :, ng:2 * ng])

    def test_energy_conserved_to_roundoff(self):
        """Pure Maxwell, periodic, no dissipation: total energy is conserved to
        round-off -- the bulk scheme + periodic BC inject/leak nothing."""
        sim, state = _gaussian_sim("periodic", N=24, ko=0.0)
        E = _energy_trace(sim, state, 200)
        rel = float(np.max(np.abs(E - E[0])) / E[0])
        assert rel < 1e-9, f"periodic energy drift = {rel:.2e} (should be round-off)"


# ── 2. Sommerfeld: passive and absorbing ──────────────────────────────────────

class TestSommerfeld:
    def test_passive_no_energy_injection(self):
        """A radiative BC must never increase domain energy: each sample must be
        <= the previous one (to round-off)."""
        sim, state = _gaussian_sim("sommerfeld", N=24, ko=0.0)
        E = _energy_trace(sim, state, 300)
        max_increase = float(np.max(np.diff(E)))
        assert max_increase < 1e-6 * E[0], (
            f"sommerfeld injected energy: max step increase = {max_increase:.2e}")

    @pytest.mark.slow
    def test_absorbs_outgoing_pulse(self):
        """After the pulse has had time to reach and cross the boundary, most of
        the energy must be gone -- confirming the BC radiates (1st-order ABC).
        cfl=0.05 (vs the 3D default 0.02) is used so 600 steps reach t≈12.5, giving
        the pulse (origin, sigma=1) time to cross the r=5 boundary -- otherwise the
        run ends with the pulse still mid-domain."""
        sim, state = _gaussian_sim("sommerfeld", N=24, ko=0.0, cfl=0.05)
        E = _energy_trace(sim, state, 600)
        ratio = E[-1] / E[0]
        assert ratio < 0.6, f"sommerfeld retained {ratio:.2f} of energy (poor absorption)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
