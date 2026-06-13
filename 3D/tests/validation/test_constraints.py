"""
Constraint-damping demonstration for the MCS 3D solver (Phase 1).

The 3D mirror of `2D/tests/validation/test_constraints.py`.  Constraint-violation
blow-up is the #1 killer of numerical-relativity runs, so the constraint-damping
terms (K1 on divE via Psi, K2 on divB via Phi) must actually DAMP an injected
violation, not merely sit quietly when it is already zero.

The spectrum suite proves this in the linearized eigenvalue picture (the
constraint-cleaning pair damps at rate K/2).  Here we prove it dynamically: seed
a real violation into the full nonlinear solver and watch it decay -- and decay
faster as K grows.  A sign error in the damping (the analogue of the load-bearing
KO sign bug) would make the violation GROW; these tests catch it.

CPU-safe (small grids, pure-Maxwell so the constraint sector is isolated).
"""

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs3d.main import (
    MaxwellChernSimons3D, InitialData, load_parameters, get_physical,
    calc_constraints, l2norm,
)
from mcs_common.wave_state import WaveState

_PF = str(Path(__file__).resolve().parents[2] / "params.toml")


def _sim(K, N=24, ko=0.0, cfl=None):
    p = load_parameters(_PF)
    p.update({"scheme": "floating_point", "Nx": N, "Ny": N, "Nz": N,
              "id_type": "gaussian", "bc_type": "periodic", "enable_cs": 0.0,
              "ko_sigma": ko, "K1": K, "K2": K, "sponge_strength": 0.0})
    if cfl is not None:
        p["cfl"] = cfl
    dx = (p["xmax"] - p["xmin"]) / N
    sim = MaxwellChernSimons3D(dx, dx, dx, p["Lambda"], p)
    return sim, p


def _seed_psi(sim, p, mode=2):
    """Zero state with a single-mode Psi (divE-cleaning potential) violation."""
    data = np.zeros((10, sim.Nx_tot, sim.Ny_tot, sim.Nz_tot))
    x = np.asarray(sim.x)
    k = 2 * np.pi * mode / (p["xmax"] - p["xmin"])
    data[8] = np.cos(k * x)[:, None, None]   # PSI = index 8
    return WaveState(jnp.asarray(data))


def _seed_divE(sim, p, mode=2, amp=0.5):
    """Zero state with a seeded divE violation: Ex = amp*sin(k x) -> divE != 0."""
    data = np.zeros((10, sim.Nx_tot, sim.Ny_tot, sim.Nz_tot))
    x = np.asarray(sim.x)
    k = 2 * np.pi * mode / (p["xmax"] - p["xmin"])
    data[0] = amp * np.sin(k * x)[:, None, None]   # EX = index 0
    return WaveState(jnp.asarray(data))


def _trace(sim, state, field_fn, n_steps, every=10):
    step = jax.jit(lambda s: sim.step_rk4(s, sim.dt))
    vals, ts = [], []
    for i in range(n_steps + 1):
        if i % every == 0:
            vals.append(field_fn(sim, state)); ts.append(i * sim.dt)
        state = step(state)
    return np.array(ts), np.array(vals)


def _psi_l2(sim, state):
    return float(l2norm(get_physical(state.data[8], sim.ng)))


def _constraint_energy(sim, state):
    """sqrt of the constraint-sector energy: |Psi|^2 + |E|^2.  Unlike |Psi| alone
    (which oscillates as energy sloshes between Psi and the longitudinal E within
    the e^{-Kt/2} envelope), this is the monotone envelope quantity -- its decay
    rate settles cleanly to the Gundlach K/2 once equipartition is reached."""
    d = np.asarray(get_physical(state.data, sim.ng))
    return float(np.sqrt(np.mean(d[8] ** 2) + np.mean(d[0] ** 2)
                         + np.mean(d[1] ** 2) + np.mean(d[2] ** 2)))


def _divE_l2(sim, state):
    divE, _ = calc_constraints(sim, state)
    return float(l2norm(get_physical(divE, sim.ng)))


# ── 1. A seeded violation decays (sign correctness) ───────────────────────────

class TestViolationDecays:
    N_STEPS = 400

    def test_psi_violation_damped(self):
        """A seeded Psi violation must decay substantially and never grow.  If the
        damping sign were wrong (anti-damping), Psi would grow instead."""
        sim, p = _sim(K=1.0)
        t, P = _trace(sim, _seed_psi(sim, p), _psi_l2, self.N_STEPS)
        assert P[-1] < 0.3 * P[0], f"Psi only decayed to {P[-1]/P[0]:.2f} of initial"
        assert np.max(P) < 1.05 * P[0], (
            f"Psi grew to {np.max(P)/P[0]:.2f}x initial -- damping sign wrong?")

    def test_divE_violation_damped(self):
        """The literal constraint quantity L2(divE) must decay when seeded as a
        real divergence in E."""
        sim, p = _sim(K=2.0)
        t, D = _trace(sim, _seed_divE(sim, p), _divE_l2, self.N_STEPS)
        assert D[-1] < 0.5 * D[0], f"divE only decayed to {D[-1]/D[0]:.2f} of initial"
        assert np.max(D) < 1.1 * D[0], f"divE grew to {np.max(D)/D[0]:.2f}x initial"


# ── 2. Damping rate scales with K ─────────────────────────────────────────────

class TestDampingRateScalesWithK:
    # The seeded Psi mode oscillates within its decay envelope, so the rate only
    # settles to the asymptotic K/2 after a few oscillation periods.  Use a longer
    # physical time (cfl=0.1 is well within stability for the pure-Maxwell
    # constraint subsystem) and the monotone constraint-energy envelope.
    N, CFL, N_STEPS = 20, 0.1, 420

    def _final_ratio(self, K):
        sim, p = _sim(K, N=self.N, cfl=self.CFL)
        t, E = _trace(sim, _seed_psi(sim, p), _constraint_energy, self.N_STEPS,
                      every=20)
        # Fit the LATE half, where the envelope has reached equipartition and
        # decays at the Gundlach K/2.
        half = len(t) // 2
        slope = float(np.polyfit(t[half:], np.log(E[half:]), 1)[0])
        return E[-1] / E[0], -slope

    def test_larger_K_decays_faster(self):
        """Stronger constraint damping must remove the violation faster."""
        r05, _ = self._final_ratio(0.5)
        r10, _ = self._final_ratio(1.0)
        r20, _ = self._final_ratio(2.0)
        assert r20 < r10 < r05, (
            f"residual not monotone in K: K0.5={r05:.3f}, K1={r10:.3f}, K2={r20:.3f}")

    def test_rate_consistent_with_half_K(self):
        """The measured envelope decay rate must be consistent with the linear-
        theory prediction of ~K/2 (the underdamped Gundlach rate).  The seeded
        mode is not a pure eigenmode, so we allow a band around K/2."""
        for K in (0.5, 1.0, 2.0):
            _, rate = self._final_ratio(K)
            assert 0.3 * K < rate < 0.9 * K, (
                f"K={K}: decay rate {rate:.3f} outside [0.3K, 0.9K] "
                f"(expected ~K/2 = {K/2})")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
