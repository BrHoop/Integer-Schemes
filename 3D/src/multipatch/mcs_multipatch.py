"""3D Maxwell-Chern-Simons (System IV, NF=10) on the Llama multipatch grid (M4).

Transcribes the validated uniform-grid RHS (``mcs3d/main.py``
``MaxwellChernSimons3D.rhs``) onto curvilinear patches: every Cartesian first
derivative becomes a :class:`derivative_curvilinear.CurvilinearDerivative`
world-derivative, and the scalar Laplacian ``laplacian(xi)`` (the only second
derivative — the ``xi``/``Pi`` pair is a wave equation) uses the analytic-Hessian
curvilinear operator. Physics is byte-for-byte the same System IV; only the
spatial operators change.

Exact solution for convergence tests: the 3D birefringent (left-circularly
polarized) plane wave, identical to ``mcs3d/validate.py:FullBirefringentOracle`` —
E/B traveling waves with phase ``k.x - omega t`` (``omega = sqrt(k^2 + m_cs k)``),
``xi = -cs Pi0 t`` (spatially uniform), ``Pi = Pi0`` constant, ``Psi = Phi = 0``.
``k`` must not be z-aligned (triad singularity). Stable (no CFJ tachyon) when
``|k| > m_cs = 2 L``.
"""
import numpy as np
import jax.numpy as jnp

# field indices (match mcs3d)
EX, EY, EZ, BX, BY, BZ, XI, PI, PSI, PHI = range(10)
NF = 10


class MCSSystem:
    """Maxwell-Chern-Simons RHS evaluated per patch with curvilinear operators."""
    NF = NF

    def __init__(self, Lambda=0.1, cs=1.0, K1=1.0, K2=1.0):
        self.L = Lambda
        self.cs = cs
        self.K1 = K1
        self.K2 = K2

    def rhs_patch(self, F, d, t):
        L, cs, K1, K2 = self.L, self.cs, self.K1, self.K2
        Ex, Ey, Ez = F[EX], F[EY], F[EZ]
        Bx, By, Bz = F[BX], F[BY], F[BZ]
        xi, Pi, Psi, Phi = F[XI], F[PI], F[PSI], F[PHI]

        # world gradients (gF[0]=d/dx, [1]=d/dy, [2]=d/dz)
        gEx, gEy, gEz = d.grad(Ex), d.grad(Ey), d.grad(Ez)
        gBx, gBy, gBz = d.grad(Bx), d.grad(By), d.grad(Bz)
        gxi, gPsi, gPhi = d.grad(xi), d.grad(Psi), d.grad(Phi)
        lap_xi = d.laplacian(xi)

        dt_Ex = (gBz[1] - gBy[2]) - gPsi[0] - cs * 2 * L * (Pi * Bx - Ez * gxi[1] + Ey * gxi[2])
        dt_Ey = (gBx[2] - gBz[0]) - gPsi[1] - cs * 2 * L * (Pi * By - Ex * gxi[2] + Ez * gxi[0])
        dt_Ez = (gBy[0] - gBx[1]) - gPsi[2] - cs * 2 * L * (Pi * Bz - Ey * gxi[0] + Ex * gxi[1])
        dt_Bx = -gEz[1] + gEy[2] + gPhi[0]
        dt_By = -gEx[2] + gEz[0] + gPhi[1]
        dt_Bz = -gEy[0] + gEx[1] + gPhi[2]
        dt_xi = -Pi * cs
        dt_Pi = (-lap_xi + 2 * L * (Bx * Ex + By * Ey + Bz * Ez)) * cs
        dt_Psi = (-gEx[0] - gEy[1] - gEz[2] - K1 * Psi
                  - cs * 2 * L * (Bx * gxi[0] + By * gxi[1] + Bz * gxi[2]))
        dt_Phi = gBx[0] + gBy[1] + gBz[2] - K2 * Phi

        return jnp.stack([dt_Ex, dt_Ey, dt_Ez, dt_Bx, dt_By, dt_Bz,
                          dt_xi, dt_Pi, dt_Psi, dt_Phi])


def _triad(k):
    kx, ky, kz = k
    kmag = float(np.sqrt(kx * kx + ky * ky + kz * kz))
    nf = float(np.sqrt(kx * kx + ky * ky))
    if nf < 1e-14:
        raise ValueError("birefringent triad is singular for z-aligned k.")
    e1 = np.array([ky / nf, -kx / nf, 0.0])
    e2 = np.array([-kx * kz / (kmag * nf), -ky * kz / (kmag * nf), nf / kmag])
    return kmag, e1, e2


def mcs_exact_state(X, Y, Z, t, k=(0.8, 0.6, 0.5), Lambda=0.1, cs=1.0, E0=1.0):
    """Exact birefringent state ``(10, *shape)`` at time ``t`` (JAX arrays)."""
    kx, ky, kz = k
    m_cs = 2.0 * Lambda
    kmag, e1, e2 = _triad(k)
    omega = (kmag * kmag + m_cs * kmag) ** 0.5
    Pi0 = m_cs / (2.0 * cs * Lambda)
    b_scale = kmag / omega

    phase = kx * X + ky * Y + kz * Z - omega * t
    c, s = jnp.cos(phase), jnp.sin(phase)
    fields = []
    # E
    for i in range(3):
        fields.append(E0 * (e1[i] * c - e2[i] * s))
    # B
    for i in range(3):
        fields.append(E0 * b_scale * (-e1[i] * s - e2[i] * c))
    z = jnp.zeros_like(X)
    fields.append(z + (-cs * Pi0 * t))   # XI
    fields.append(z + Pi0)               # PI
    fields.append(z)                     # PSI
    fields.append(z)                     # PHI
    return jnp.stack(fields)


def mcs_initial_data(grid, t=0.0, k=(0.8, 0.6, 0.5), Lambda=0.1, cs=1.0, E0=1.0):
    """Per-patch initial-data tuple for the birefringent wave."""
    return tuple(
        mcs_exact_state(p.X, p.Y, p.Z, t, k=k, Lambda=Lambda, cs=cs, E0=E0)
        for p in grid.patches
    )
