"""Minimal 3D grid for BSSN testing (coords + spacing + ghost zones).

Mirrors ``MaxwellChernSimons3D._init_grid`` so initial-data generators and the
(Phase-2.2) RHS share the same vertex-centred, ghost-padded layout. Uses
``arange`` to avoid the duplicate-endpoint trap (same convention as the MCS grid).
"""

from dataclasses import dataclass

import jax.numpy as jnp
from mcs_common.derivatives import SpatialDerivative


@dataclass
class Grid:
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    ng: int
    xmin: float = -0.5
    ymin: float = -0.5
    zmin: float = -0.5

    def __post_init__(self):
        ng = self.ng
        self.x = self.xmin + jnp.arange(-ng, self.Nx + ng) * self.dx
        self.y = self.ymin + jnp.arange(-ng, self.Ny + ng) * self.dy
        self.z = self.zmin + jnp.arange(-ng, self.Nz + ng) * self.dz
        self.X, self.Y, self.Z = jnp.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.R = jnp.sqrt(self.X**2 + self.Y**2 + self.Z**2) + 1e-15

    @property
    def shape(self):
        return (self.Nx + 2 * self.ng, self.Ny + 2 * self.ng, self.Nz + 2 * self.ng)

    @classmethod
    def from_domain(cls, N, order=6, lo=-0.5, hi=0.5):
        """Cubic domain ``[lo, hi]^3`` with ``N`` interior cells per axis."""
        if isinstance(N, int):
            N = (N, N, N)
        ng = SpatialDerivative(order=order).ng
        d = [(hi - lo) / n for n in N]
        return cls(Nx=N[0], Ny=N[1], Nz=N[2], dx=d[0], dy=d[1], dz=d[2],
                   ng=ng, xmin=lo, ymin=lo, zmin=lo)
