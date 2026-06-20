"""Block-structured AMR for the 3D Llama multipatch grid (node-centered).

A 3D, node-centered port of the validated 2D ``mcs2d/amr`` infrastructure:

  state    — AMRState (JAX pytree, ragged per-level) + AMRTopology (host)
  kernels  — node-centered prolongation / injection restriction / indicator
  sync     — within-level + cross-level ghost fill (added in Phase A2)
  regrid   — gradient-indicator regridding driver (added in Phase A4)
  evolve   — single-dt RK4 over the hierarchy (added in Phase A3)
"""
from . import state
from . import kernels

__all__ = ["state", "kernels"]
