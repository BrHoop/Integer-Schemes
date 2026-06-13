import jax.numpy as jnp
from jax import lax
from typing import ClassVar

class SpatialDerivative:
    """Evaluates high-order finite differences for N-dimensional grids.

    Calculates 1st derivatives, 2nd derivatives, and Kreiss-Oliger dissipation
    using 6th-order central difference stencils.

    Attributes:
        order (int): The finite difference order (currently locked to 6).
        ng (int): Number of ghost cells required for the stencil.
    """

    C1: ClassVar[jnp.ndarray] = jnp.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
    C2: ClassVar[jnp.ndarray] = jnp.array([2, -27, 270, -490, 270, -27, 2]) / 180.0
    # Kreiss-Oliger dissipation: the 6th-order central difference δ⁶, scaled by
    # 1/2^6.  Added to the RHS with a +σ coefficient, this DAMPS high-wavenumber
    # modes (symbol −64 at the Nyquist frequency → −σ/Δx, dissipative).
    # NOTE: the sign here is load-bearing.  The previous value was the NEGATED
    # stencil [-1,6,-15,20,-15,6,-1], which made KO anti-dissipative and drove
    # an exponential grid-scale instability (stronger σ → faster blow-up).
    #
    # ORDER (deferred): this is a 6th-order KO operator (7-pt, reach ±3, fits
    # NG=3).  A 6th-order SCHEME formally wants an 8th-order KO (9-pt, reach ±4)
    # so the O(Δx^7) dissipation error stays below the O(Δx^6) truncation error;
    # 6th-order KO injects an O(Δx^5) term that can cap convergence at ~5th order
    # at very high resolution.  Empirically the scheme still shows clean 6th-order
    # accuracy at current resolutions with σ=0.05, so we keep NG=3 + 6th-order KO.
    # If a convergence study during BBH waveform work shows dissipation is the
    # accuracy floor, bump ONLY the FP path to NG=4 + 8th-order KO (the extra
    # ghost cell is free here; the SMEM-bound Ozaki/Pallas path should stay NG=3).
    CKO: ClassVar[jnp.ndarray] = jnp.array([1, -6, 15, -20, 15, -6, 1]) / 64.0

    def __init__(self, order: int = 6):
        """Initializes the derivative operator."""
        if order != 6:
            raise NotImplementedError(
                f"Currently only 6th-order is supported, but {order} was requested."
            )
        self.order = order
        self.ng = order // 2

    def _apply(self, grid: jnp.ndarray, coeffs: jnp.ndarray, factor: float, axis: int) -> jnp.ndarray:
        stencil = coeffs * factor
        pad_width = [(self.ng, self.ng) if i == axis else (0, 0) for i in range(grid.ndim)]
        padded = jnp.pad(grid, pad_width, mode='edge')
        n = grid.shape[axis]
        return sum(stencil[k] * lax.slice_in_dim(padded, k, k + n, axis=axis)
                   for k in range(stencil.size))

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