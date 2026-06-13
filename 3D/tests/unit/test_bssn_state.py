"""Phase 2.1 unit tests: BSSN state, params, and test-only initial data.

Checks the structural invariants the BSSN constraint algebra relies on:
    det(~g_ij) = 1        (unit-determinant conformal metric)
    ~g^{ij} ~A_ij = 0     (conformal traceless extrinsic curvature)
plus the pytree contract and PhysicsParams usability as a static JIT arg.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bssn3d.state import BSSNState, PhysicsParams, NUM_VARS, SYM_IDX
from bssn3d.grid import Grid
from bssn3d import initial_data as bid

jax.config.update("jax_enable_x64", True)


def _sym_matrix(comps):
    """(6, ...) symmetric components -> (..., 3, 3) full matrix."""
    g = jnp.zeros(comps.shape[1:] + (3, 3), dtype=comps.dtype)
    for i in range(3):
        for j in range(3):
            g = g.at[..., i, j].set(comps[SYM_IDX[(i, j)]])
    return g


def _det_gt(state):
    return jnp.linalg.det(_sym_matrix(state.gt))


def _conformal_trace_At(state):
    gt = _sym_matrix(state.gt)
    At = _sym_matrix(state.At)
    igt = jnp.linalg.inv(gt)
    return jnp.einsum("...ij,...ij->...", igt, At)


@pytest.fixture
def grid():
    return Grid.from_domain(16, order=6, lo=-0.5, hi=0.5)


# --- state container ---------------------------------------------------------

def test_minkowski_shape_and_values(grid):
    s = BSSNState.minkowski(grid.shape)
    assert s.data.shape == (NUM_VARS,) + grid.shape
    assert jnp.allclose(s.alpha, 1.0)
    assert jnp.allclose(s.chi, 1.0)
    assert jnp.allclose(_det_gt(s), 1.0)
    assert jnp.allclose(s.At, 0.0)
    assert jnp.allclose(s.K, 0.0)


def test_pytree_roundtrip(grid):
    s = BSSNState.minkowski(grid.shape)
    leaves, treedef = jax.tree_util.tree_flatten(s)
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(s.data, s2.data)
    # tree_map (used by step_rk4) preserves the type and acts elementwise
    s3 = jax.tree_util.tree_map(lambda x: 2.0 * x, s)
    assert isinstance(s3, BSSNState)
    assert jnp.allclose(s3.alpha, 2.0)


def test_symmetric_accessors(grid):
    s = BSSNState.minkowski(grid.shape)
    assert jnp.allclose(s.gt_ij(0, 1), s.gt_ij(1, 0))   # symmetry
    assert jnp.allclose(s.gt_ij(0, 0), 1.0)
    assert jnp.allclose(s.gt_ij(1, 2), 0.0)


# --- physics params ----------------------------------------------------------

def test_params_hashable_static_arg():
    p = PhysicsParams()
    assert hash(p) == hash(PhysicsParams())          # frozen + hashable
    assert p.lam(0) == 1.0 and p.lam(3) == 1.0
    assert len(p.lmbda) == 4 and len(p.lambda_f) == 2

    def f(x, params):
        return x * params.eta
    # passes through jit as a static (hashable) argument
    out = jax.jit(f, static_argnums=1)(jnp.ones(3), p)
    assert jnp.allclose(out, 2.0)


# --- initial data ------------------------------------------------------------

def test_gauge_wave_invariants(grid):
    s = bid.gauge_wave(grid, amplitude=0.01)
    # Minkowski-in-disguise: unit-determinant conformal metric + trace-free At
    assert jnp.allclose(_det_gt(s), 1.0, atol=1e-12)
    assert jnp.max(jnp.abs(_conformal_trace_At(s))) < 1e-12
    assert jnp.all(s.chi > 0.0)
    # alpha = sqrt(H), H = 1 - A sin(phase)  ->  alpha in (sqrt(1-A), sqrt(1+A))
    assert jnp.all(s.alpha > 0.0)
    # alpha^2 = H and chi = H^(-1/3)  ->  alpha^2 = chi^(-3)
    assert jnp.max(jnp.abs(s.alpha**2 - s.chi**(-3.0))) < 1e-12


def test_gauge_wave_axes(grid):
    """Wave on each axis puts the H^(2/3) component in the matching slot."""
    for axis in range(3):
        s = bid.gauge_wave(grid, amplitude=0.02, axis=axis)
        assert jnp.allclose(_det_gt(s), 1.0, atol=1e-12)
        # the Gt vector is nonzero only along the wave axis
        for a in range(3):
            comp = s.Gt[a]
            if a == axis:
                assert jnp.max(jnp.abs(comp)) > 0.0
            else:
                assert jnp.allclose(comp, 0.0)


def test_robust_stability_bounded(grid):
    amp = 1e-10
    s = bid.robust_stability(grid, amp=amp, seed=3)
    assert s.data.shape == (NUM_VARS,) + grid.shape
    base = BSSNState.minkowski(grid.shape).data
    dev = jnp.abs(s.data - base)
    assert jnp.max(dev) <= amp + 1e-18
    assert jnp.max(dev) > 0.0          # noise actually applied
