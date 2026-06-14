"""Llama-style multipatch (cubed-sphere) grids for the 3D solver.

A CPU research prototype (Phase 5.1 pulled forward): a central Cartesian cube
wrapped by 6 Thornburg-04 cubed-sphere shells, coupled by overlap + high-order
interpolation. Built to characterize how Llama grids behave in this project's
finite-difference / JAX framework, first on a scalar wave then on 3D MCS.

Modules
-------
coord_maps            analytic affine + cubed-sphere maps (Jacobian, inverse, contains)
atlas                 the 7-patch grid: per-node world coords + inverse Jacobians
derivative_curvilinear  reference-frame FD -> world-Cartesian via J^{-1}  (M1)
overlap               inter-patch ghost fill via precomputed Lagrange weights (M2)
wave                  first-order scalar-wave RHS on the multipatch grid       (M3)
evolve                RK4 over patches + overlap sync + outer BC + KO          (M3)
"""

from . import coord_maps
from . import atlas

__all__ = ["coord_maps", "atlas"]
