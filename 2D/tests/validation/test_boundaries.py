"""
Boundary-condition correctness for the MCS 2D solver (Step 1.1).

The accuracy/spectral suites all use PERIODIC boundaries; this file is the only
coverage of the other BC code paths.  A boundary condition must never be a
SOURCE -- it can remove energy (an absorbing/radiative BC) or conserve it
(periodic), but it must not inject energy into the domain.  Energy injection is
exactly the "boundary error" that contaminates a BBH waveform extraction, so we
test for it directly.

Findings encoded here
---------------------
* periodic   : exact ghost wrap; conserves energy to round-off (no injection).
* sommerfeld : passive (energy non-increasing) and genuinely absorbing.

Historical note: a third BC, `directional` (advection-based outflow), was REMOVED
in Step 1.1 after this suite found it unstable for general data -- it injected
energy and blew up by ~1e9-1e15 regardless of advection velocity or KO.  Only
`periodic` and `sommerfeld` remain.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.main import (
    MaxwellChernSimons2D, InitialData, load_parameters, get_physical,
)

_PARAMS = "2D/params.toml"


def _gaussian_sim(bc, N=48, *, cs=0.0, ko=0.0, **over):
    """Pure-Maxwell (cs=0) Gaussian-pulse sim with the requested BC.  Pure Maxwell
    isolates the boundary behaviour from CS/constraint-damping energy changes."""
    from pathlib import Path
    pf = str(Path(__file__).resolve().parents[2] / "params.toml")
    p = load_parameters(pf)
    p.update({"scheme": "floating_point", "Nx": N, "Ny": N, "id_type": "gaussian",
              "bc_type": bc, "enable_cs": cs, "ko_sigma": ko, "K1": 0.0, "K2": 0.0,
              "sponge_strength": 0.0, "id_sigma": 1.0, **over})
    dx = (p["xmax"] - p["xmin"]) / N
    sim = MaxwellChernSimons2D(dx, dx, p["Lambda"], p)
    state = InitialData(sim, p).generate()
    return sim, state


def _energy_trace(sim, state, n_steps, every=10):
    """Total interior field 'energy' (sum of squares) sampled every `every` steps."""
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
        sim, state = _gaussian_sim("periodic", N=32)
        f = np.asarray(state.data[0])              # one field, with ghosts
        wrapped = np.asarray(sim.bc_periodic(jnp.asarray(f)))
        ng = sim.ng
        # left ghost == last interior columns; right ghost == first interior cols
        assert np.array_equal(wrapped[:ng, :], wrapped[-2 * ng:-ng, :])
        assert np.array_equal(wrapped[-ng:, :], wrapped[ng:2 * ng, :])
        assert np.array_equal(wrapped[:, :ng], wrapped[:, -2 * ng:-ng])
        assert np.array_equal(wrapped[:, -ng:], wrapped[:, ng:2 * ng])

    def test_energy_conserved_to_roundoff(self):
        """Pure Maxwell, periodic, no dissipation: total energy is conserved to
        round-off -- the bulk scheme + periodic BC inject/leak nothing."""
        sim, state = _gaussian_sim("periodic", N=48, ko=0.0)
        E = _energy_trace(sim, state, 400)
        rel = float(np.max(np.abs(E - E[0])) / E[0])
        assert rel < 1e-9, f"periodic energy drift = {rel:.2e} (should be round-off)"


# ── 2. Sommerfeld: passive and absorbing ──────────────────────────────────────

class TestSommerfeld:
    def test_passive_no_energy_injection(self):
        """A radiative BC must never increase domain energy: each sample must be
        <= the previous one (to round-off).  Energy injection here would be the
        boundary acting as a spurious source -- the failure we most need to rule
        out for waveform extraction."""
        sim, state = _gaussian_sim("sommerfeld", N=48, ko=0.0)
        E = _energy_trace(sim, state, 600)
        max_increase = float(np.max(np.diff(E)))
        assert max_increase < 1e-6 * E[0], (
            f"sommerfeld injected energy: max step increase = {max_increase:.2e}")

    def test_absorbs_outgoing_pulse(self):
        """After the pulse has had time to reach and cross the boundary, most of
        the energy must be gone -- confirming the BC actually radiates (it is a
        1st-order ABC, so absorption is good but not perfect)."""
        sim, state = _gaussian_sim("sommerfeld", N=48, ko=0.0)
        E = _energy_trace(sim, state, 600)
        ratio = E[-1] / E[0]
        assert ratio < 0.6, f"sommerfeld retained {ratio:.2f} of energy (poor absorption)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
