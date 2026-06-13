import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
import bfp48 as _bfp48

@register_pytree_node_class
class WaveState:
    def __init__(self, arr):
        self.data = arr

    @property
    def Ex(self): return self.data[0]
    @property
    def Ey(self): return self.data[1]
    @property
    def Ez(self): return self.data[2]
    @property
    def Bx(self): return self.data[3]
    @property
    def By(self): return self.data[4]
    @property
    def Bz(self): return self.data[5]
    @property
    def xi(self): return self.data[6]
    @property
    def Pi(self): return self.data[7]
    @property
    def Psi(self): return self.data[8]
    @property
    def Phi(self): return self.data[9]

    def tree_flatten(self):
        return ((self.data,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class WaveStateBFP48:
    """BFP48-compressed WaveState for reduced HBM bandwidth.

    Each field has its own shared base-2 exponent. Values reconstruct as:
        v[f, i, j] = mantissas[f, i, j, :] (int48) * 2^(-exponents[f])

    Storage: 6 bytes/value instead of 8 (25% less HBM traffic).
    The int48 mantissa feeds directly into Ozaki RNS with no float scaling step.
    """

    def __init__(self, mantissas: jnp.ndarray, exponents: jnp.ndarray):
        self.mantissas = mantissas  # (F, Nx, Ny, 3) int16
        self.exponents = exponents  # (F,) int32

    @staticmethod
    def from_float64(data: jnp.ndarray) -> "WaveStateBFP48":
        """Pack (F, Nx, Ny) float64 → WaveStateBFP48. One exponent per field."""
        mantissas, exponents = jax.vmap(_bfp48.pack)(data)
        return WaveStateBFP48(mantissas, exponents)

    def to_float64(self) -> jnp.ndarray:
        """Unpack → (F, Nx, Ny) float64."""
        return jax.vmap(_bfp48.unpack)(self.mantissas, self.exponents)

    def tree_flatten(self):
        return ((self.mantissas, self.exponents), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
