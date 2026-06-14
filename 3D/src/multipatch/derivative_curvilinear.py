"""World-Cartesian derivatives on curvilinear patches.

A field is differentiated in the patch's *reference* coordinates ``xi`` with the
shared uniform stencils (:class:`mcs_common.derivatives.SpatialDerivative`), then
transformed to world-Cartesian via the per-node inverse Jacobian precomputed in
:class:`multipatch.atlas.Patch`::

    d f / d x_A  =  sum_a  (dxi^a / dx_A) * (d f / d xi^a)
                =  sum_a  jinv[a, A] * compute_d1(f, dxi_a, axis=a)

Second world-Cartesian derivatives use the analytic-connection form

    d2 f / dx_A dx_B = sum_{ab} jinv[a,A] jinv[b,B] f_{,ab}
                       + sum_a C[a,A,B] f_{,a}

where ``f_{,ab}`` is the reference Hessian (dedicated ``compute_d2`` on the
diagonal, composed ``compute_d1`` off-diagonal) and ``C[a,A,B] = dJinv[a,A]/dx_B``
is the precomputed connection (``atlas.Patch.d2coef``). Because every term
differences the *field* (whose ghosts are filled) and the geometric tensors are
precomputed, the result is valid on the full interior with the standard
``ng = order//2`` ghost width — no doubled ghosts, no finite-differenced metric.
This is the operator the BSSN RHS needs (``grad2_i_j``) as well.

For the affine cube ``jinv = I / (2a)`` and ``C = 0``, so ``d_world`` reduces
*exactly* to a
uniform Cartesian finite difference at spacing ``2a/(N-1)`` — the regression
anchor checked by the tests.

Convention: ``f`` has the patch's ghost-padded shape ``(nx, ny, nz)``. The
returned array has the same shape; only interior values are trustworthy (the
outermost ``ng`` layers use the stencil's edge-padding and are overwritten by
the overlap fill / BC each step, exactly as in the uniform solver).
"""
import jax.numpy as jnp

from mcs_common.derivatives import SpatialDerivative


class CurvilinearDerivative:
    """First-derivative + KO operator bound to one :class:`atlas.Patch`."""

    def __init__(self, patch, order: int = 6):
        if patch.ng < order // 2:
            raise ValueError(
                f"patch.ng ({patch.ng}) < order//2 ({order // 2}); rebuild the "
                f"grid with at least this ghost width.")
        self.patch = patch
        self.order = order
        self.op = SpatialDerivative(order=order)
        self.jinv = patch.jinv            # (3, 3, nx, ny, nz): [a_log, A_world]
        self.d2coef = patch.d2coef        # (3, 3, 3, ...): C[a,A,B]
        self.dxi = patch.dxi              # (dl0, dl1, dl2) logical spacings

    def _ref_grads(self, f):
        """The three reference-frame first derivatives ``df/dxi^a``."""
        return [self.op.compute_d1(f, self.dxi[a], axis=a) for a in range(3)]

    def _ref_hessian(self, f):
        """Reference first derivs ``G[a]`` and Hessian ``H[a][b]`` (symmetric).

        Diagonal entries use the dedicated 2nd-derivative stencil; off-diagonal
        entries compose two first derivatives. All are valid on the full
        interior at ``ng = order//2`` (they difference the field, whose ghosts
        are filled)."""
        G = self._ref_grads(f)
        H = [[None, None, None] for _ in range(3)]
        for a in range(3):
            H[a][a] = self.op.compute_d2(f, self.dxi[a], axis=a)
        for a in range(3):
            for b in range(a + 1, 3):
                m = self.op.compute_d1(G[a], self.dxi[b], axis=b)
                H[a][b] = m
                H[b][a] = m
        return G, H

    def d_world(self, f, A: int):
        """``df/dx_A`` (A=0,1,2 -> world x,y,z)."""
        dref = self._ref_grads(f)
        return sum(self.jinv[a, A] * dref[a] for a in range(3))

    def grad(self, f):
        """World gradient ``(df/dx, df/dy, df/dz)``, sharing the reference grads."""
        dref = self._ref_grads(f)
        return tuple(
            sum(self.jinv[a, A] * dref[a] for a in range(3)) for A in range(3)
        )

    def divergence(self, fx, fy, fz):
        """World divergence ``dfx/dx + dfy/dy + dfz/dz`` of a vector field."""
        return self.d_world(fx, 0) + self.d_world(fy, 1) + self.d_world(fz, 2)

    def d2_world(self, f, A: int, B: int):
        """Second world-Cartesian derivative ``d2 f / dx_A dx_B``."""
        G, H = self._ref_hessian(f)
        out = sum(self.jinv[a, A] * self.jinv[b, B] * H[a][b]
                  for a in range(3) for b in range(3))
        out = out + sum(self.d2coef[a, A, B] * G[a] for a in range(3))
        return out

    def hessian_world(self, f):
        """All six unique world second derivatives as a dict ``{(A,B): d2}``
        (A<=B). Shares one reference-Hessian computation."""
        G, H = self._ref_hessian(f)
        out = {}
        for A in range(3):
            for B in range(A, 3):
                d2 = sum(self.jinv[a, A] * self.jinv[b, B] * H[a][b]
                         for a in range(3) for b in range(3))
                d2 = d2 + sum(self.d2coef[a, A, B] * G[a] for a in range(3))
                out[(A, B)] = d2
        return out

    def laplacian(self, f):
        """World Laplacian ``sum_A d2 f / dx_A^2`` (single reference Hessian)."""
        G, H = self._ref_hessian(f)
        lap = 0.0
        for A in range(3):
            lap = lap + sum(self.jinv[a, A] * self.jinv[b, A] * H[a][b]
                            for a in range(3) for b in range(3))
            lap = lap + sum(self.d2coef[a, A, A] * G[a] for a in range(3))
        return lap

    def ko(self, f, sigma: float):
        """Kreiss-Oliger dissipation summed over the three *reference* axes.

        Applied in logical coordinates (the SBP-free interface stabilizer from
        the project's boundary strategy). The KO sign in
        ``SpatialDerivative.compute_ko`` is load-bearing — do not negate.
        """
        return sum(self.op.compute_ko(f, self.dxi[a], sigma, axis=a)
                   for a in range(3))
