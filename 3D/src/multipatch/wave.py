"""First-order scalar wave system on the Llama multipatch grid (M3 test vehicle).

The scalar wave equation ``d_tt phi = laplacian(phi)`` is written as the
first-order symmetric-hyperbolic reduction used by cubesphere's ``fowave``:

    state = (phi, pi, chi_x, chi_y, chi_z),   chi_i := d_i phi,  pi := d_t phi

    d_t phi   = pi
    d_t pi    = d_x chi_x + d_y chi_y + d_z chi_z      (= div chi = laplacian phi)
    d_t chi_i = d_i pi

Only **first** world-Cartesian derivatives appear, so the curvilinear machinery
in :mod:`derivative_curvilinear` (a single ``J^{-1}`` contraction) is all that's
needed. Wave speed is 1.

Exact solution for convergence tests: a Cartesian plane wave

    phi = A sin(k.x - omega t),  omega = |k|

is an exact solution everywhere (so it crosses every patch seam), giving
``pi = -A omega cos(...)`` and ``chi_i = A k_i cos(...)``. Imposed as initial
data and as the outer-boundary Dirichlet value, it isolates the multipatch
interior + seam + curvilinear-derivative accuracy.
"""
import jax.numpy as jnp

from .derivative_curvilinear import CurvilinearDerivative

# field indices
PHI, PI, CX, CY, CZ = 0, 1, 2, 3, 4
NF = 5


class WaveSystem:
    """The first-order scalar wave RHS, evaluated per patch."""
    NF = NF

    def rhs_patch(self, F, d: CurvilinearDerivative, t):
        phi, pi, cx, cy, cz = F[PHI], F[PI], F[CX], F[CY], F[CZ]
        r_phi = pi
        r_pi = d.divergence(cx, cy, cz)
        gx, gy, gz = d.grad(pi)
        return jnp.stack([r_phi, r_pi, gx, gy, gz])


def plane_wave_state(X, Y, Z, t, k=(0.7, -0.4, 0.5), amp=1.0):
    """Exact plane-wave state ``(phi, pi, chi_x, chi_y, chi_z)`` at time ``t``.

    ``X, Y, Z`` are world-coordinate arrays (any shape); returns a stacked
    ``(5, *shape)`` array.
    """
    kx, ky, kz = k
    omega = (kx * kx + ky * ky + kz * kz) ** 0.5
    phase = kx * X + ky * Y + kz * Z - omega * t
    s = jnp.sin(phase)
    c = jnp.cos(phase)
    phi = amp * s
    pi = -amp * omega * c
    chi_x = amp * kx * c
    chi_y = amp * ky * c
    chi_z = amp * kz * c
    return jnp.stack([phi, pi, chi_x, chi_y, chi_z])


def plane_wave_initial_data(grid, t=0.0, k=(0.7, -0.4, 0.5), amp=1.0):
    """Per-patch initial-data tuple for the plane wave."""
    return tuple(
        plane_wave_state(p.X, p.Y, p.Z, t, k=k, amp=amp) for p in grid.patches
    )
