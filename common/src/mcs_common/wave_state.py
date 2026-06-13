from jax.tree_util import register_pytree_node_class

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
    