"""Phase-4.C — RK4-2(2) MSRK integrator in evolve.py: correctness, 4th-order, stability, timing.

RK4-2(2) reuses the previous step's RHS to do 3 fresh evals/step instead of RK4's 4. At the
production Courant factor (CFL=0.1, KO=0.4 — far below the stability limit) this is a clean ~1.33x
stage-count win with identical 4th-order accuracy. These tests pin: (1) it reproduces RK4's
solution to truncation, (2) the agreement is genuinely 4th-order (drops ~16x under dt-halving),
(3) it is stable over a longer run at production CFL/KO. Timing is printed (the decisive wall-clock
A/B is on the GPU `scheme="cuda_fused"` path; see the recipe in the test docstring).

GPU wall-clock A/B (Marylou): build BSSNEvolution(..., scheme="cuda_fused", integrator="rk4") vs
integrator="rk4_2_2"; time evolve() per step. Expect ~1.33x fewer RHS evals -> ~1.33x/step at CFL=0.1.
"""
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bssn3d.grid import Grid
from bssn3d.evolve import BSSNEvolution
from bssn3d import initial_data

CFL, KO = 0.1, 0.4          # production operating point


def _rms(a):
    return float(jnp.sqrt(jnp.mean(a ** 2)))


def _evo(grid, integ):
    return BSSNEvolution(grid, ko_sigma=KO, bc="periodic", integrator=integ)


def test_rk42_reproduces_rk4_to_truncation():
    grid = Grid.from_domain(16)
    ic = initial_data.gauge_wave(grid, amplitude=0.01)
    dt = CFL * grid.dx
    s4 = _evo(grid, "rk4").evolve(ic, dt, 40)
    s2 = _evo(grid, "rk4_2_2").evolve(ic, dt, 40)
    assert bool(jnp.all(jnp.isfinite(s2.data)))
    rel = _rms(s2.data - s4.data) / _rms(s4.data)
    assert rel < 1e-5, rel               # two 4th-order methods agree to truncation


def test_agreement_is_fourth_order():
    """Halving dt to the same final time should shrink the RK4-2(2)-vs-RK4 gap ~16x (4th order)."""
    grid = Grid.from_domain(16)
    ic = initial_data.gauge_wave(grid, amplitude=0.01)
    dx = grid.dx

    def gap(cfl, nsteps):
        dt = cfl * dx
        s4 = _evo(grid, "rk4").evolve(ic, dt, nsteps)
        s2 = _evo(grid, "rk4_2_2").evolve(ic, dt, nsteps)
        return _rms(s2.data - s4.data)

    g_coarse = gap(0.10, 20)
    g_fine = gap(0.05, 40)               # same final time, dt halved
    ratio = g_coarse / g_fine
    assert ratio > 8.0, ratio            # ~16 ideal; >8 confirms ~4th-order (loose for round-off)


def test_stable_at_production_cfl_ko():
    """Longer RK4-2(2) run at CFL=0.1, KO=0.4 stays finite and bounded (no blow-up)."""
    grid = Grid.from_domain(16)
    ic = initial_data.gauge_wave(grid, amplitude=0.01)
    dt = CFL * grid.dx
    s = _evo(grid, "rk4_2_2").evolve(ic, dt, 200)
    assert bool(jnp.all(jnp.isfinite(s.data)))
    assert _rms(s.data) < 10.0 * _rms(ic.data)   # bounded relative to the initial amplitude


def test_timing_ratio_print():
    """Info-only: per-step wall-clock RK4 vs RK4-2(2) (CPU is noisy; GPU is the real A/B)."""
    grid = Grid.from_domain(16)
    ic = initial_data.gauge_wave(grid, amplitude=0.01)
    dt = CFL * grid.dx
    out = {}
    for integ in ("rk4", "rk4_2_2"):
        e = _evo(grid, integ)
        e.evolve(ic, dt, 8).data.block_until_ready()      # warm up / compile
        t0 = time.perf_counter()
        e.evolve(ic, dt, 60).data.block_until_ready()
        out[integ] = (time.perf_counter() - t0) / 60.0
    print(f"\nper-step wall-clock @16^3 (CPU): RK4 {out['rk4']*1e3:.1f} ms  "
          f"RK4-2(2) {out['rk4_2_2']*1e3:.1f} ms  -> {out['rk4']/out['rk4_2_2']:.2f}x")
