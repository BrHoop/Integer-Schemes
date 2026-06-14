"""M3/M4 evolution: cross-seam convergence + stability (slow — 7-patch compile).

These compile a full 7-patch RK4 step (~1-3 min uncached), so they are marked
``slow``. Order is read at fixed dt across two resolutions so the RK4 temporal
floor doesn't mask the spatial order.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from multipatch import atlas as A, overlap as O, wave as W
from multipatch import mcs_multipatch as M
from multipatch.evolve import MultipatchEvolution, make_exact_dirichlet_bc

pytestmark = pytest.mark.slow

WK = (0.7, -0.4, 0.5)
MK = (0.8, 0.6, 0.5)
LAM = 0.1


def _run_wave(N, dt=0.01, nsteps=40):
    g = A.build_llama_grid(2.0, 1.8, 8.0, N, N, N, order=6)
    tbl = O.build_overlap_table(g, order=6)
    bc = make_exact_dirichlet_bc(g, tbl, lambda X, Y, Z, t: W.plane_wave_state(X, Y, Z, t, k=WK))
    ev = MultipatchEvolution(g, tbl, W.WaveSystem(), order=6, ko_sigma=0.0, outer_bc=bc)
    f0 = W.plane_wave_initial_data(g, t=0.0, k=WK)
    fF, _ = ev.evolve(f0, dt, nsteps)
    T = nsteps * dt
    maxe = 0.0
    for p, F in zip(g.patches, fF):
        ex = W.plane_wave_state(p.X, p.Y, p.Z, T, k=WK)
        intr = (slice(None),) + p.interior
        maxe = max(maxe, float(jnp.max(jnp.abs((F - ex)[intr]))))
    return maxe


def _run_mcs(N, dt=0.01, nsteps=40):
    g = A.build_llama_grid(2.0, 1.8, 8.0, N, N, N, order=6)
    tbl = O.build_overlap_table(g, order=6)
    bc = make_exact_dirichlet_bc(g, tbl, lambda X, Y, Z, t: M.mcs_exact_state(X, Y, Z, t, k=MK, Lambda=LAM))
    ev = MultipatchEvolution(g, tbl, M.MCSSystem(Lambda=LAM), order=6, ko_sigma=0.0, outer_bc=bc)
    f0 = M.mcs_initial_data(g, t=0.0, k=MK, Lambda=LAM)
    fF, _ = ev.evolve(f0, dt, nsteps)
    T = nsteps * dt
    maxe = 0.0
    for p, F in zip(g.patches, fF):
        ex = M.mcs_exact_state(p.X, p.Y, p.Z, T, k=MK, Lambda=LAM)
        intr = (slice(None),) + p.interior
        maxe = max(maxe, float(jnp.max(jnp.abs((F - ex)[intr]))))
    return maxe


def test_wave_converges_across_seams():
    e1, e2 = _run_wave(15), _run_wave(23)
    assert np.isfinite(e1) and np.isfinite(e2)
    order = np.log(e1 / e2) / np.log(23 / 15)
    assert order > 5.0


def test_mcs_converges_across_seams():
    e1, e2 = _run_mcs(15), _run_mcs(23)
    assert np.isfinite(e1) and np.isfinite(e2)
    order = np.log(e1 / e2) / np.log(23 / 15)
    assert order > 4.5            # MCS: 6th-order spatial, modest pre-asymptotic slack
