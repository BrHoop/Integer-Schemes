"""M1/M1b curvilinear derivatives: order on shells, exactness on the cube."""
import jax.numpy as jnp
import numpy as np
import pytest

from multipatch import atlas as A
from multipatch.derivative_curvilinear import CurvilinearDerivative
from mcs_common.derivatives import SpatialDerivative

KA, KB, KC = 0.7, -0.4, 0.5


def _f(X, Y, Z):
    return jnp.sin(KA * X + KB * Y + KC * Z)


def _shell_op(N):
    g = A.build_llama_grid(2.0, 1.8, 8.0, N, N, N, order=6)
    return g.shells[0], CurvilinearDerivative(g.shells[0], order=6)


def _order(err_lo, err_hi, N_lo, N_hi):
    return np.log(err_lo / err_hi) / np.log(N_hi / N_lo)


def test_shell_first_derivative_order():
    def err(N):
        sh, d = _shell_op(N)
        F = _f(sh.X, sh.Y, sh.Z)
        Dx, Dy, Dz = d.grad(F)
        intr = sh.interior
        ex = jnp.max(jnp.abs((Dx - KA * jnp.cos(KA*sh.X+KB*sh.Y+KC*sh.Z))[intr]))
        return float(ex)
    e1, e2 = err(15), err(25)
    assert _order(e1, e2, 15, 25) > 5.0


def test_shell_second_derivative_order():
    def err(N):
        sh, d = _shell_op(N)
        F = _f(sh.X, sh.Y, sh.Z)
        intr = sh.interior
        s = jnp.sin(KA*sh.X+KB*sh.Y+KC*sh.Z)
        lap_ex = -(KA*KA+KB*KB+KC*KC) * s
        dxy_ex = -KA*KB * s
        el = float(jnp.max(jnp.abs((d.laplacian(F) - lap_ex)[intr])))
        em = float(jnp.max(jnp.abs((d.d2_world(F, 0, 1) - dxy_ex)[intr])))
        return el, em
    (el1, em1), (el2, em2) = err(15), err(25)
    assert _order(el1, el2, 15, 25) > 5.0
    assert _order(em1, em2, 15, 25) > 5.0


def test_cube_reduces_to_cartesian():
    g = A.build_llama_grid(2.0, 1.8, 8.0, 17, 17, 17, order=6)
    cube = g.cube
    d = CurvilinearDerivative(cube, order=6)
    F = _f(cube.X, cube.Y, cube.Z)
    hx = 2 * g.cube_half_width / (17 - 1)
    op = SpatialDerivative(6)
    # first derivative
    assert float(jnp.max(jnp.abs(d.d_world(F, 0) - op.compute_d1(F, hx, 0)))) < 1e-12
    # laplacian
    lap_plain = (op.compute_d2(F, hx, 0) + op.compute_d2(F, hx, 1)
                 + op.compute_d2(F, hx, 2))
    assert float(jnp.max(jnp.abs(d.laplacian(F) - lap_plain))) < 1e-12
    # mixed second derivative
    dxy_plain = op.compute_d1(op.compute_d1(F, hx, 0), hx, 1)
    assert float(jnp.max(jnp.abs(d.d2_world(F, 0, 1) - dxy_plain))) < 1e-12


def test_hessian_world_keys():
    g = A.build_llama_grid(2.0, 1.8, 8.0, 13, 13, 13, order=6)
    d = CurvilinearDerivative(g.shells[0], order=6)
    H = d.hessian_world(_f(g.shells[0].X, g.shells[0].Y, g.shells[0].Z))
    assert set(H.keys()) == {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
    # consistency with d2_world
    intr = g.shells[0].interior
    F = _f(g.shells[0].X, g.shells[0].Y, g.shells[0].Z)
    assert float(jnp.max(jnp.abs((H[(0, 1)] - d.d2_world(F, 0, 1))[intr]))) < 1e-13
