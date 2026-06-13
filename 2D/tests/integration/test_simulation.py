"""
System-level correctness tests for the MCS 2D solver.

Covers the three things the unit tests cannot catch:
  1. The assembled PDE produces the correct physical solution
     (all schemes compared against the exact birefringent wave).
  2. The constraint evolution (divB ≈ 0, divE ≈ 0) is maintained.
  3. All schemes agree with fused_floating_point (the FP64 reference) over a
     full multi-step trajectory.
"""

from pathlib import Path


import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.main import (
    MaxwellChernSimons2D, InitialData,
    calc_constraints, get_physical, load_parameters, l2norm,
)

_PARAMS_FILE = str(Path(__file__).resolve().parent.parent.parent / 'params.toml')

SCHEMES = ["floating_point", "ozaki", "fused_ozaki", "fused_floating_point",
           "pallas_ozaki"]

# `pallas_ozaki` is the only scheme that lowers through Pallas-Triton.
# The kernels are written for the JAX 0.9.x Triton lowering (pow2 tile padding +
# split/ref-indexing, no value-slice/scatter) — the project pin.  These tests run
# in CPU interpret mode (correctness; interpret skips Triton lowering), so they
# pass on any version, but we skip on < 0.9 since the kernels target 0.9.x
# primitives and the older 0.8.x Triton lowering differs.  GPU Triton-lowering
# validity must be checked on the H200 separately.
_JAX_MAJOR_MINOR = tuple(int(x) for x in jax.__version__.split(".")[:2])
_skip_pallas = pytest.mark.skipif(
    _JAX_MAJOR_MINOR < (0, 9),
    reason=(f"pallas kernels target the JAX 0.9.x Triton lowering; running JAX "
            f"{jax.__version__}."))


# ── Helpers ────────────────────────────────────────────────────────────────────

class BirefringentOracle:
    """Exact analytical solution for the 2.5D left-circularly polarised wave."""

    def __init__(self, params):
        Lx = params["xmax"] - params["xmin"]
        Ly = params["ymax"] - params["ymin"]
        k_x = 2.0 * jnp.pi / Lx
        k_y = 2.0 * jnp.pi / Ly
        k   = jnp.sqrt(k_x**2 + k_y**2)
        m_cs = params.get("id_m_cs", params.get("Lambda", 1.0) * 2.0)
        self.k_x   = k_x
        self.k_y   = k_y
        self.omega  = jnp.sqrt(k**2 + m_cs * k)
        self.E0     = params.get("id_amp", 1.0)

    def Ez(self, X, Y, t):
        return self.E0 * jnp.sin(self.k_x * X + self.k_y * Y - self.omega * t)


def _make_sim(scheme, nx=64, ny=64, **overrides):
    """Build (sim, state, params) with birefringent IC and periodic BCs."""
    params = load_parameters(_PARAMS_FILE)
    params.update({
        'scheme': scheme, 'Nx': nx, 'Ny': ny, 'Nt': 1,
        'id_type': 'birefringent', 'bc_type': 'periodic',
        'sponge_strength': 0.0,
        **overrides,
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim   = MaxwellChernSimons2D(dx, dy, params['Lambda'], params)
    state = InitialData(sim, params).generate()
    return sim, state, params


def _run(sim, state, n_steps):
    """Advance state by n_steps RK4 steps under jit+scan."""
    def body(carry, _):
        return sim.step_rk4(carry, sim.dt), None
    return jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_steps)[0])(state)


# ── 1. Analytical accuracy ─────────────────────────────────────────────────────

class TestAnalyticalAccuracy:
    """
    Each scheme must reproduce the exact Ez of the birefringent wave after
    100 steps.  Expected L2 error for a 6th-order scheme on 64×64: ~1e-8.
    Tolerance 1e-7 gives a comfortable 10× margin while catching any wrong PDE
    term or incorrect divisor.
    """
    N_STEPS = 100
    L2_TOL  = 1e-7

    def _check(self, scheme):
        sim, state, params = _make_sim(scheme)
        state  = _run(sim, state, self.N_STEPS)
        oracle = BirefringentOracle(params)

        t_final = self.N_STEPS * sim.dt
        x_phys  = sim.x[sim.ng:-sim.ng]
        y_phys  = sim.y[sim.ng:-sim.ng]
        X, Y    = jnp.meshgrid(x_phys, y_phys, indexing='ij')

        l2_err = float(l2norm(get_physical(state.data[sim.EZ], sim.ng)
                               - oracle.Ez(X, Y, t_final)))
        assert l2_err < self.L2_TOL, (
            f"{scheme}: L2(Ez − exact) = {l2_err:.2e}  (tol {self.L2_TOL:.0e})"
        )

    def test_floating_point(self):       self._check("floating_point")
    def test_ozaki(self):                self._check("ozaki")
    def test_fused_ozaki(self):          self._check("fused_ozaki")
    def test_fused_floating_point(self): self._check("fused_floating_point")
    @_skip_pallas
    def test_pallas_ozaki(self):         self._check("pallas_ozaki")


# ── 2. Constraint conservation ─────────────────────────────────────────────────

class TestConstraintConservation:
    """
    The magnetic constraint (div B = 0) and electric constraint (div E = MCS source)
    must remain near their initial values throughout the evolution.

    For the birefringent IC with periodic BCs both constraints start
    near machine zero, so the L2 norm after 100 steps is a direct measure of
    constraint violation growth.
    """
    N_STEPS  = 100
    DIVB_TOL = 1e-8   # div B is analytically zero; should stay at float64 round-off
    DIVE_TOL = 1e-7   # MCS modifies Gauss's law; slightly looser

    def _check(self, scheme):
        sim, state, _ = _make_sim(scheme)
        state = _run(sim, state, self.N_STEPS)

        divE, divB = calc_constraints(sim, state)
        divB_l2 = float(l2norm(get_physical(divB, sim.ng)))
        divE_l2 = float(l2norm(get_physical(divE, sim.ng)))

        assert divB_l2 < self.DIVB_TOL, (
            f"{scheme}: L2(divB) = {divB_l2:.2e}  (tol {self.DIVB_TOL:.0e})"
        )
        assert divE_l2 < self.DIVE_TOL, (
            f"{scheme}: L2(divE) = {divE_l2:.2e}  (tol {self.DIVE_TOL:.0e})"
        )

    def test_floating_point(self):       self._check("floating_point")
    def test_ozaki(self):                self._check("ozaki")
    def test_fused_ozaki(self):          self._check("fused_ozaki")
    def test_fused_floating_point(self): self._check("fused_floating_point")
    @_skip_pallas
    def test_pallas_ozaki(self):         self._check("pallas_ozaki")


# ── 3. Cross-scheme agreement ──────────────────────────────────────────────────

class TestSchemeAgreement:
    """
    All schemes must reproduce fused_floating_point to within ATOL after 100 steps.

    fused_floating_point is the reference: it is the fastest pure-FP64 path and
    is proven identical to floating_point (same arithmetic, different execution).
    Using it as the reference keeps cross-scheme tests fast without changing the
    numerical bar.

    - floating_point: unfused, untiled; must agree to near machine epsilon.
    - ozaki / fused_ozaki: Ozaki INT8 RNS is designed to be exact in float64
      arithmetic; any deviation is purely floating-point round-off.
    """
    N_STEPS = 100
    ATOL    = 1e-8

    def _final_state(self, scheme):
        sim, state, _ = _make_sim(scheme)
        return sim.ng, np.array(_run(sim, state, self.N_STEPS).data)

    def _check(self, scheme):
        ng, ref = self._final_state("fused_floating_point")
        _,  alt = self._final_state(scheme)
        diff = float(np.max(np.abs(ref[:, ng:-ng, ng:-ng] - alt[:, ng:-ng, ng:-ng])))
        assert diff < self.ATOL, f"{scheme} vs fused_floating_point: max_diff = {diff:.2e}"

    def test_floating_point_vs_fused_fp(self):        self._check("floating_point")
    def test_ozaki_vs_fused_fp(self):                 self._check("ozaki")
    def test_fused_ozaki_vs_fused_fp(self):           self._check("fused_ozaki")
    @_skip_pallas
    def test_pallas_ozaki_vs_fused_fp(self):          self._check("pallas_ozaki")


# ── 4. Non-trivial initial data ────────────────────────────────────────────────

class TestGaussianIC:
    """
    Gaussian pulse IC exercises non-zero xi/Pi initial data and the K1/K2
    constraint-damping terms.

    The Gaussian IC builds B from finite-difference derivatives of a scalar,
    so discrete commutation error leaves a non-zero initial divB (~1e-6).
    This is an IC construction artefact, not a solver bug; constraint
    conservation is verified rigorously in TestConstraintConservation using
    the birefringent IC where divB starts at machine zero.

    Checks here: no NaN/Inf, field amplitude remains bounded (no instability).
    """
    N_STEPS   = 50
    AMP_BOUND = 10.0   # L2 norm of all fields over interior; initial ~O(0.1)

    def _check(self, scheme):
        sim, state, _ = _make_sim(scheme, id_type='gaussian', bc_type='periodic')
        state = _run(sim, state, self.N_STEPS)

        assert not bool(jnp.any(jnp.isnan(state.data))), f"{scheme}: NaN in state"
        assert not bool(jnp.any(jnp.isinf(state.data))), f"{scheme}: Inf in state"

        amp = float(l2norm(get_physical(state.data, sim.ng)))
        assert amp < self.AMP_BOUND, (
            f"{scheme}: field L2 = {amp:.2e} > {self.AMP_BOUND} — possible instability"
        )

    def test_floating_point(self):       self._check("floating_point")
    def test_ozaki(self):                self._check("ozaki")
    def test_fused_ozaki(self):          self._check("fused_ozaki")
    def test_fused_floating_point(self): self._check("fused_floating_point")
    @_skip_pallas
    def test_pallas_ozaki(self):         self._check("pallas_ozaki")


# ── 5. Long-run stability ─────────────────────────────────────────────────────

class TestLongRunStability:
    """
    Long-horizon evolution to catch slowly-growing instabilities that a
    100-step test would miss.  Critical for the BSSN-bound use case where
    physical evolutions span thousands of timesteps.

    Each scheme runs N_STEPS of the birefringent wave and is checked for:
      1. No NaN / Inf
      2. Peak |E|, |B| stays bounded (no blow-up).  Note: xi grows linearly
         in time analytically (xi(t) = -c_s·Pi_0·t), so it is excluded.
      3. L2 error of Ez vs the exact oracle stays below L2_TOL
    """
    N_STEPS = 200
    L2_TOL  = 1e-5    # generous: accumulated error over 200 steps at 64x64
    EB_BOUND = 1.5    # |E|, |B| should not exceed ~initial amplitude (~0.8)

    def _check(self, scheme):
        sim, state, params = _make_sim(scheme)
        oracle = BirefringentOracle(params)

        def body(carry, _):
            return sim.step_rk4(carry, sim.dt), None

        state = jax.jit(lambda s: jax.lax.scan(body, s, None,
                                               length=self.N_STEPS)[0])(state)
        state.data.block_until_ready()

        assert not bool(jnp.any(jnp.isnan(state.data))), f"{scheme}: NaN"
        assert not bool(jnp.any(jnp.isinf(state.data))), f"{scheme}: Inf"

        # E and B fields (indices 0..5) — the actual electromagnetic wave;
        # xi grows linearly by design, Pi stays at Pi_0, Psi/Phi are damping.
        eb_amp = float(jnp.max(jnp.abs(get_physical(state.data[:6], sim.ng))))
        assert eb_amp < self.EB_BOUND, (
            f"{scheme}: peak |E|,|B|={eb_amp:.3f} > {self.EB_BOUND} "
            f"after {self.N_STEPS} steps — possible instability"
        )

        x_phys = sim.x[sim.ng:-sim.ng]
        y_phys = sim.y[sim.ng:-sim.ng]
        X, Y = jnp.meshgrid(x_phys, y_phys, indexing='ij')
        t = self.N_STEPS * sim.dt
        l2_err = float(l2norm(get_physical(state.data[sim.EZ], sim.ng)
                              - oracle.Ez(X, Y, t)))
        assert l2_err < self.L2_TOL, (
            f"{scheme}: L2(Ez - exact) = {l2_err:.2e} after {self.N_STEPS} "
            f"steps (tol {self.L2_TOL:.0e})"
        )

    def test_floating_point(self):       self._check("floating_point")
    def test_ozaki(self):                self._check("ozaki")
    def test_fused_ozaki(self):          self._check("fused_ozaki")
    def test_fused_floating_point(self): self._check("fused_floating_point")
    @_skip_pallas
    def test_pallas_ozaki(self):         self._check("pallas_ozaki")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
