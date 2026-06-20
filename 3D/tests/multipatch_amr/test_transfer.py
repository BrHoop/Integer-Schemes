"""Node-centered AMR transfer operators: coincident-copy, polynomial exactness,
restrict∘prolong identity, indicator sanity."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from multipatch.amr.state import BS, NG, NF
from multipatch.amr import kernels as K

W = BS + 2 * NG
CORNERS = list(itertools.product((0, 1), repeat=3))


def _rand_parent(seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((NF, W, W, W)))


@pytest.mark.parametrize("corner", CORNERS)
def test_coincident_copy(corner):
    """Even fine interior nodes coincide with parent nodes → EXACT copy, any data."""
    parent = _rand_parent()
    child = K.prolongate(parent, corner)
    cx, cy, cz = corner
    half = BS // 2
    # child[:, NG+2a, NG+2b, NG+2c] == parent[:, NG+cx*half+a, ...]  (a,b,c interior)
    a = np.arange(half)
    ci = NG + 2 * a
    pi_x = NG + cx * half + a
    pi_y = NG + cy * half + a
    pi_z = NG + cz * half + a
    got = np.asarray(child)[np.ix_(np.arange(NF), ci, ci, ci)]
    exp = np.asarray(parent)[np.ix_(np.arange(NF), pi_x, pi_y, pi_z)]
    assert np.max(np.abs(got - exp)) == 0.0


def test_polynomial_exact():
    """A degree-5-per-axis polynomial is reproduced to ~machine precision at the
    deep-interior fine nodes (6-point Lagrange is exact for degree ≤ 5)."""
    h = 0.37
    # separable degree-5 polynomials
    cx5 = np.array([0.4, -1.1, 0.7, 0.2, -0.3, 0.15])
    cy5 = np.array([-0.5, 0.9, -0.2, 0.6, 0.1, -0.25])
    cz5 = np.array([0.8, 0.3, -0.6, -0.1, 0.45, 0.05])

    def P(c, x):
        return sum(c[d] * x**d for d in range(6))

    p_idx = np.arange(W)
    xp = (p_idx - NG) * h
    Fx, Fy, Fz = P(cx5, xp), P(cy5, xp), P(cz5, xp)
    parent = jnp.asarray(np.einsum("i,j,k->ijk", Fx, Fy, Fz)[None].repeat(NF, 0))

    child = np.asarray(K.prolongate(parent, (0, 0, 0)))

    # child interior node ff at coord (ff-NG)*h/2 (corner 0); test deep interior
    # where the stencil stays inside the sampled parent block.
    lo, hi = NG + 3, NG + BS - 3
    ff = np.arange(lo, hi)
    xf = (ff - NG) * h / 2
    exp = np.einsum("i,j,k->ijk", P(cx5, xf), P(cy5, xf), P(cz5, xf))
    got = child[np.ix_(np.arange(NF), ff, ff, ff)]
    assert np.max(np.abs(got - exp[None])) < 1e-9


@pytest.mark.parametrize("corner", CORNERS)
def test_restrict_inverts_prolong(corner):
    """Injection restriction of a prolonged child recovers the parent footprint at
    the coincident nodes (restrict∘prolong = identity on coarse nodes)."""
    parent = _rand_parent(seed=3)
    child = K.prolongate(parent, corner)
    base = jnp.zeros_like(parent)
    out = np.asarray(K.restrict_into_parent(base, child, corner))
    cx, cy, cz = corner
    half = BS // 2
    a = np.arange(half)
    px = NG + cx * half + a
    py = NG + cy * half + a
    pz = NG + cz * half + a
    oi = NG + a
    got = out[np.ix_(np.arange(NF), NG + cx*half + a, NG + cy*half + a, NG + cz*half + a)]
    exp = np.asarray(parent)[np.ix_(np.arange(NF), px, py, pz)]
    assert np.max(np.abs(got - exp)) == 0.0


def test_indicator():
    const = jnp.ones((NF, W, W, W))
    assert float(K.compute_indicator_gradient(const)) == 0.0
    ramp = jnp.asarray(np.broadcast_to(
        np.arange(W)[None, :, None, None], (NF, W, W, W)).astype(float))
    assert float(K.compute_indicator_gradient(ramp)) == pytest.approx(1.0)
