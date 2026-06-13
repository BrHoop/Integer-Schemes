import jax.numpy as jnp
from jax.scipy.signal import correlate
from typing import ClassVar

class SpatialDerivative:
    """Evaluates high-order finite differences for N-dimensional grids.

    Calculates 1st derivatives, 2nd derivatives, and Kreiss-Oliger dissipation 
    using 8th-order central difference stencils.

    Attributes:
        order (int): The finite difference order (currently locked to 8).
        ng (int): Number of ghost cells required for the stencil.
    """

    C1: ClassVar[jnp.ndarray] = jnp.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
    C2: ClassVar[jnp.ndarray] = jnp.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]) / 5040.0
    CKO: ClassVar[jnp.ndarray] = jnp.array([-1, 8, -28, 56, -70, 56, -28, 8, -1]) / 256.0

    def __init__(self, order: int = 8):
        """Initializes the derivative operator."""
        if order != 8:
            raise NotImplementedError(
                f"Currently only 8th-order is supported, but {order} was requested."
            )
        self.order = order
        self.ng = order // 2

    def _apply(self, grid: jnp.ndarray, coeffs: jnp.ndarray, factor: float, axis: int) -> jnp.ndarray:
        """Core convolution engine for applying 1D stencils across N-D grids."""
        stencil = coeffs * factor
        
        stencil_shape = [1] * grid.ndim
        stencil_shape[axis] = stencil.size
        stencil_nd = jnp.reshape(stencil, stencil_shape)
        
        pad_width = [(0, 0)] * grid.ndim
        pad_width[axis] = (self.ng, self.ng)
        padded_grid = jnp.pad(grid, pad_width, mode='edge')
        
        return correlate(padded_grid, stencil_nd, mode='valid')

    def compute_d1(self, grid: jnp.ndarray, dx: float, axis: int) -> jnp.ndarray:
        """Computes the first spatial derivative.
        
        Args:
            grid: The input N-dimensional array.
            dx: Grid spacing along the target axis.
            axis: The dimension to take the derivative along (0 for x, 1 for y).
        """
        return self._apply(grid, self.C1, 1.0 / dx, axis)

    def compute_d2(self, grid: jnp.ndarray, dx: float, axis: int) -> jnp.ndarray:
        """Computes the second spatial derivative."""
        return self._apply(grid, self.C2, 1.0 / (dx**2), axis)

    def compute_ko(self, grid: jnp.ndarray, dx: float, sigma: float, axis: int) -> jnp.ndarray:
        """Computes Kreiss-Oliger dissipation for high-frequency noise filtering."""
        return self._apply(grid, self.CKO, sigma / dx, axis)