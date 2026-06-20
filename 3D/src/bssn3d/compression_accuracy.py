"""Phase-4 — BSSN accuracy gate for BFP storage compression (does BFP32 hold on the REAL RHS?).

The MCS gate showed BFP storage is accuracy-cheap on a smooth, coarse, short problem — the
optimistic case. This module runs the BSSN-specific tests the MCS gate cannot: the cancellation-
heavy curvature/constraint algebra, on strong-field data, over an evolution. It wraps the BSSN
derivative operator with `QuantizingDeriv` (every grad stored at `mant_bits`, computed in fp64) and
measures:

  1. `single_eval_rhs_error`     — relative L2 of one compressed RHS eval vs fp64.
  2. `constraint_perturbation`   — how much BFP-k derivs perturb the (cancellation-sensitive)
                                   Hamiltonian/momentum constraints vs fp64 derivs, same state.
  3. `evolution_constraint_growth` — evolve at fp64 vs BFP-k; monitor H/|M| (with an fp64 monitor)
                                   over the run — does compression accelerate constraint growth?

All CPU. Strong-field data = polarized Gowdy (nonlinear, real curvature). Caveat: no puncture ID
here, so the χ→0 dynamic-range extreme is only mildly stressed; Gowdy is the strongest available.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mcs_common.bfp_compress import QuantizingDeriv, BFP48, BFP40, BFP32, FP32_MANT
from .grid import Grid
from .rhs import BSSNSolver
from .evolve import BSSNEvolution
from .constraints import ConstraintSolver
from . import initial_data

_FP64 = 53
_CFL = 0.25


def _l2(a):
    return float(jnp.sqrt(jnp.mean(a ** 2)))


def _interior(arr, ng):
    sl = (slice(ng, -ng),) * 3
    return arr[(slice(None),) + sl] if arr.ndim == 4 else arr[sl]


def initial_state(grid, idname="gowdy"):
    if idname == "gowdy":
        return initial_data.gowdy(grid, t0=1.0)
    if idname == "gauge_wave":
        return initial_data.gauge_wave(grid, amplitude=0.1)
    raise ValueError(idname)


def _solver(grid, mant_bits):
    s = BSSNSolver(grid)
    if mant_bits < _FP64:
        s.diff_op = QuantizingDeriv(s.diff_op, mant_bits)
    return s


def single_eval_rhs_error(N, mant_bits, idname="gowdy"):
    """Global relative L2 error of one compressed BSSN RHS eval vs fp64, interior stack."""
    grid = Grid.from_domain(N)
    state = initial_state(grid, idname)
    dt = _CFL * grid.dx
    a = _interior(_solver(grid, _FP64).rhs(state, t=0.0, dt=dt).data, grid.ng)
    b = _interior(_solver(grid, mant_bits).rhs(state, t=0.0, dt=dt).data, grid.ng)
    return _l2(b - a) / _l2(a)


def constraint_perturbation(N, mant_bits, idname="gowdy"):
    """Relative change in the (H, |M|) constraints when the constraint derivatives are stored at
    `mant_bits` vs fp64, on the SAME state — the cancellation-sensitivity test."""
    grid = Grid.from_domain(N)
    state = initial_state(grid, idname)
    cs_ref = ConstraintSolver(grid)
    cs_cmp = ConstraintSolver(grid)
    cs_cmp.diff_op = QuantizingDeriv(cs_cmp.diff_op, mant_bits) if mant_bits < _FP64 else cs_cmp.diff_op
    H0, M0 = cs_ref.l2(state)
    H1, M1 = cs_cmp.l2(state)
    dH = abs(H1 - H0) / (H0 + 1e-300)
    dM = abs(M1 - M0) / (M0 + 1e-300)
    return H0, M0, dH, dM


def evolution_constraint_growth(N, mant_bits, nsteps, idname="gowdy",
                                ko_sigma=0.1, monitor_every=10):
    """Evolve at `mant_bits` (fp64 if >=53); return [(step, H_l2, |M|_l2)] from an fp64 monitor."""
    grid = Grid.from_domain(N)
    evo = BSSNEvolution(grid, ko_sigma=ko_sigma, bc="periodic")
    if mant_bits < _FP64:
        evo.solver.diff_op = QuantizingDeriv(evo.solver.diff_op, mant_bits)
        evo.diff_op = evo.solver.diff_op          # evolution KO uses this too
    mon = ConstraintSolver(grid)                  # fp64 diagnostic
    state = initial_state(grid, idname)
    dt = _CFL * grid.dx
    traj = []
    H, M = mon.l2(state)
    traj.append((0, H, M))
    done = 0
    while done < nsteps:
        chunk = min(monitor_every, nsteps - done)
        state = evo.evolve(state, dt, chunk, t0=done * dt)
        done += chunk
        H, M = mon.l2(state)
        traj.append((done, H, M))
    return traj
