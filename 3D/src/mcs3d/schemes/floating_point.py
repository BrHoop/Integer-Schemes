"""3D MCS finite-difference scheme.

``SpatialDerivative`` was promoted to ``mcs_common.derivatives`` in Phase 2 (so
the BSSN port shares one source of truth). It is re-exported here verbatim, so
every existing ``from mcs3d.schemes.floating_point import SpatialDerivative``
keeps working with identical behavior.
"""

from mcs_common.derivatives import SpatialDerivative

__all__ = ["SpatialDerivative"]
