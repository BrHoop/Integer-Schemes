"""
Correctness tests for the fused vmap RHS kernel.

Run with pytest:
    cd Integer-Schemes
    python -m pytest mcs2d/tests/ -v

Or standalone:
    cd Integer-Schemes && python mcs2d/tests/fused_rhs_test.py
"""

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.schemes.fused_rhs_fp import make_fused_rhs, NF, NG, BS
from mcs2d.main import MaxwellChernSimons2D, InitialData, load_parameters

# All field names in order
FIELD_NAMES = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'xi', 'Pi', 'Psi', 'Phi']

# Tolerance for fused vs FP64 unfused comparison.
# Both use identical FP64 arithmetic, so agreement should be within a few ULPs.
ATOL = 1e-12


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_sim(nx=64, ny=64, bc_type='periodic', id_type='birefringent',
              ko_sigma=0.05, cs=1.0, Lambda=1.0, K1=1.0, K2=1.0):
    """Build a floating-point reference sim and return (sim, state, params)."""
    params = load_parameters(str(Path(__file__).resolve().parent.parent.parent / 'params.toml'))
    params.update({
        'Nx': nx, 'Ny': ny, 'Nt': 1,
        'bc_type': bc_type, 'id_type': id_type,
        'sponge_strength': 0.0, 'ko_sigma': ko_sigma,
        'enable_cs': cs, 'Lambda': Lambda,
        'K1': K1, 'K2': K2, 'scheme': 'floating_point',
    })
    dx = (params['xmax'] - params['xmin']) / nx
    dy = (params['ymax'] - params['ymin']) / ny
    sim = MaxwellChernSimons2D(dx, dy, params['Lambda'], params)
    state = InitialData(sim, params).generate()
    return sim, state, params


def _run_fused(sim, state, params):
    """Run the fused kernel and return its RHS array (NF, Nx_tot, Ny_tot)."""
    fn = make_fused_rhs(
        sim.Nx_tot, sim.Ny_tot, sim.dx, sim.dy,
        params['enable_cs'], params['Lambda'],
        params['K1'], params['K2'], params['ko_sigma'],
    )
    # Apply the same periodic BC pre-sync that _rhs_unfused does
    data_synced = jax.vmap(sim.bc_periodic)(state.data)
    return fn(data_synced)


def _compare(sim, rhs_fp, rhs_fused, atol=ATOL):
    """Return per-field linf errors over the interior (ghost zones excluded)."""
    ng = sim.ng
    errs = {}
    for f, name in enumerate(FIELD_NAMES):
        fp_v  = rhs_fp.data[f, ng:-ng, ng:-ng]
        fus_v = rhs_fused[f, ng:-ng, ng:-ng]
        errs[name] = float(jnp.max(jnp.abs(fp_v - fus_v)))
    return errs


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFusedRhsCorrectness:
    """Fused kernel must agree with the FP64 unfused RHS to float64 precision."""

    def test_birefringent_64x64_all_fields(self):
        sim, state, params = _make_sim(nx=64, ny=64)
        rhs_fp = sim.rhs(state)
        rhs_fused = _run_fused(sim, state, params)
        errs = _compare(sim, rhs_fp, rhs_fused)
        for name, err in errs.items():
            assert err < ATOL, f"Field {name}: linf={err:.2e} > {ATOL:.0e}"

    def test_birefringent_128x128_all_fields(self):
        """4x4 block grid exercises block-stitching path."""
        sim, state, params = _make_sim(nx=128, ny=128)
        rhs_fp = sim.rhs(state)
        rhs_fused = _run_fused(sim, state, params)
        errs = _compare(sim, rhs_fp, rhs_fused)
        for name, err in errs.items():
            assert err < ATOL, f"Field {name}: linf={err:.2e} > {ATOL:.0e}"

    def test_gaussian_64x64(self):
        """Gaussian initial data (non-trivial xi, Pi fields)."""
        sim, state, params = _make_sim(nx=64, ny=64, id_type='gaussian')
        rhs_fp = sim.rhs(state)
        rhs_fused = _run_fused(sim, state, params)
        errs = _compare(sim, rhs_fp, rhs_fused)
        for name, err in errs.items():
            assert err < ATOL, f"Field {name}: linf={err:.2e} > {ATOL:.0e}"

    def test_no_ko_dissipation(self):
        """ko_sigma=0 disables the KO branch; kernel must still be correct."""
        sim, state, params = _make_sim(nx=64, ny=64, ko_sigma=0.0)
        rhs_fp = sim.rhs(state)
        rhs_fused = _run_fused(sim, state, params)
        errs = _compare(sim, rhs_fp, rhs_fused)
        for name, err in errs.items():
            assert err < ATOL, f"Field {name}: linf={err:.2e} > {ATOL:.0e}"

    def test_zero_cs_coupling(self):
        """cs=0 turns MCS into Maxwell + scalar wave — verify PDE terms vanish.
        Uses gaussian IC because birefringent IC divides by cs."""
        sim, state, params = _make_sim(nx=64, ny=64, cs=0.0, id_type='gaussian')
        rhs_fp = sim.rhs(state)
        rhs_fused = _run_fused(sim, state, params)
        errs = _compare(sim, rhs_fp, rhs_fused)
        for name, err in errs.items():
            assert err < ATOL, f"Field {name}: linf={err:.2e} > {ATOL:.0e}"

    def test_non_square_grid(self):
        """Grid whose size is not a multiple of BS — tests the padding path."""
        sim, state, params = _make_sim(nx=96, ny=64)
        rhs_fp = sim.rhs(state)
        rhs_fused = _run_fused(sim, state, params)
        errs = _compare(sim, rhs_fp, rhs_fused)
        for name, err in errs.items():
            assert err < ATOL, f"Field {name}: linf={err:.2e} > {ATOL:.0e}"

    def test_output_shape(self):
        """Output shape must match (NF, Nx_tot, Ny_tot)."""
        sim, state, params = _make_sim(nx=64, ny=64)
        rhs_fused = _run_fused(sim, state, params)
        assert rhs_fused.shape == (NF, sim.Nx_tot, sim.Ny_tot)
        assert rhs_fused.dtype == jnp.float64

    def test_output_no_nan_inf(self):
        """Output must not contain NaN or Inf."""
        sim, state, params = _make_sim(nx=64, ny=64)
        rhs_fused = _run_fused(sim, state, params)
        assert not bool(jnp.any(jnp.isnan(rhs_fused))), "NaN in fused output"
        assert not bool(jnp.any(jnp.isinf(rhs_fused))), "Inf in fused output"


class TestFusedRhsIntegration:
    """Run a short time integration with both schemes and compare trajectories."""

    def test_short_evolution_agreement(self):
        """10 RK4 steps of fused vs unfused should agree to near float64 precision."""
        nx, ny = 64, 64
        sim, state, params = _make_sim(nx=nx, ny=ny)
        params_fused = dict(params)
        params_fused['scheme'] = 'fused_ozaki'

        dx = sim.dx; dy = sim.dy
        sim_fused = MaxwellChernSimons2D(dx, dy, params['Lambda'], params_fused)

        @jax.jit
        def step_fp(s):    return sim.step_rk4(s, sim.dt)
        @jax.jit
        def step_fused(s): return sim_fused.step_rk4(s, sim_fused.dt)

        from mcs_common.wave_state import WaveState
        state_fp    = WaveState(state.data)
        state_fused = WaveState(state.data)

        for _ in range(10):
            state_fp    = step_fp(state_fp)
            state_fused = step_fused(state_fused)

        ng = sim.ng
        diff = jnp.max(jnp.abs(
            state_fp.data[:, ng:-ng, ng:-ng] - state_fused.data[:, ng:-ng, ng:-ng]
        ))
        assert float(diff) < 1e-10, f"10-step trajectory diverged: max_diff={float(diff):.2e}"


# ── Standalone smoke test ─────────────────────────────────────────────────────

def _smoke():
    print("=" * 60)
    print("Fused RHS Pallas Smoke Test")
    print("=" * 60)
    import jax
    print(f"JAX backend: {jax.default_backend()}")
    print(f"interpret mode: {jax.default_backend() == 'cpu'}")
    print()

    sim, state, params = _make_sim(nx=64, ny=64)
    rhs_fp    = sim.rhs(state)
    rhs_fused = _run_fused(sim, state, params)
    errs      = _compare(sim, rhs_fp, rhs_fused)

    all_ok = True
    for name, err in errs.items():
        ok = err < ATOL
        all_ok = all_ok and ok
        print(f"  {'OK' if ok else 'FAIL':4s}  {name:4s}  linf={err:.2e}")

    print()
    print("All fields match." if all_ok else "SOME FIELDS FAILED.")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    _smoke()
