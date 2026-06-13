"""BSSN time evolution: RK4 + Kreiss-Oliger dissipation + algebraic constraint
enforcement + (periodic) boundary sync, on top of the verbatim RHS.

This is the machinery the Phase-2.3 apples tests drive. It is kept separate from
``rhs.py`` (the bit-compared verbatim algebra) so the RHS stays a clean,
oracle-checkable object; everything stability-related (KO, det/trace enforcement,
floors, BC) lives here.

Algebraic enforcement after each step (standard moving-puncture hygiene):
  * det(~g_ij) = 1     — rescale ~g_ij by det^(-1/3)
  * ~g^{ij} ~A_ij = 0  — remove the conformal trace of ~A
  * chi >= chi_floor, alpha >= alpha_floor
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from .state import BSSNState, PhysicsParams, GT0, AT0, CHI, ALPHA
from .grid import Grid
from .rhs import BSSNSolver


class BSSNEvolution:
    def __init__(self, grid: Grid, params: PhysicsParams = None, order: int = 6,
                 ko_sigma: float = 0.0, bc: str = "periodic",
                 precision: str = "fp64"):
        self.grid = grid
        self.params = params or PhysicsParams()
        self.solver = BSSNSolver(grid, self.params, order, precision=precision)
        self.diff_op = self.solver.diff_op
        self.ng = grid.ng
        self.ko_sigma = ko_sigma
        self.bc = bc

    # --- boundary ---
    def _periodic_sync(self, data):
        ng = self.ng
        data = data.at[:, :ng, :, :].set(data[:, -2 * ng:-ng, :, :])
        data = data.at[:, -ng:, :, :].set(data[:, ng:2 * ng, :, :])
        data = data.at[:, :, :ng, :].set(data[:, :, -2 * ng:-ng, :])
        data = data.at[:, :, -ng:, :].set(data[:, :, ng:2 * ng, :])
        data = data.at[:, :, :, :ng].set(data[:, :, :, -2 * ng:-ng])
        data = data.at[:, :, :, -ng:].set(data[:, :, :, ng:2 * ng])
        return data

    # --- Kreiss-Oliger dissipation (added to the RHS; +sigma form is dissipative) ---
    def _ko(self, data):
        if self.ko_sigma <= 0.0:
            return 0.0
        s = self.ko_sigma
        acc = jnp.zeros_like(data)
        for axis, dx in enumerate((self.grid.dx, self.grid.dy, self.grid.dz)):
            acc = acc + jax.vmap(
                lambda u: self.diff_op.compute_ko(u, dx, s, axis))(data)
        return acc

    def rhs(self, state: BSSNState, t: float = 0.0, dt: float = None) -> BSSNState:
        data = self._periodic_sync(state.data) if self.bc == "periodic" else state.data
        rate = self.solver.rhs(BSSNState(data), t=t, dt=dt).data
        rate = rate + self._ko(data)
        return BSSNState(rate)

    # --- algebraic constraint enforcement ---
    def enforce(self, state: BSSNState) -> BSSNState:
        d = state.data
        g0, g1, g2, g3, g4, g5 = (d[GT0 + i] for i in range(6))
        det = g0 * (g3 * g5 - g4 * g4) - g1 * (g1 * g5 - g4 * g2) + g2 * (g1 * g4 - g3 * g2)
        scale = det ** (-1.0 / 3.0)
        for i in range(6):
            d = d.at[GT0 + i].set(d[GT0 + i] * scale)

        # unit-det inverse (closed form for symmetric 3x3, det == 1 after rescale)
        g0, g1, g2, g3, g4, g5 = (d[GT0 + i] for i in range(6))
        iuxx = g3 * g5 - g4 * g4
        iuxy = g2 * g4 - g1 * g5
        iuxz = g1 * g4 - g2 * g3
        iuyy = g0 * g5 - g2 * g2
        iuyz = g1 * g2 - g0 * g4
        iuzz = g0 * g3 - g1 * g1
        a0, a1, a2, a3, a4, a5 = (d[AT0 + i] for i in range(6))
        trA = (iuxx * a0 + iuyy * a3 + iuzz * a5
               + 2.0 * (iuxy * a1 + iuxz * a2 + iuyz * a4))
        third = trA / 3.0
        gt = (g0, g1, g2, g3, g4, g5)
        for i in range(6):
            d = d.at[AT0 + i].set(d[AT0 + i] - gt[i] * third)

        d = d.at[CHI].set(jnp.maximum(d[CHI], self.params.chi_floor))
        d = d.at[ALPHA].set(jnp.maximum(d[ALPHA], self.params.alpha_floor))
        return BSSNState(d)

    # --- RK4 step (+ enforcement) ---
    # The CAHD+SSL RHS is time-dependent (SSL Gaussian ramp), so each RK4 substage
    # is evaluated at its own time t, t+dt/2, t+dt/2, t+dt; the CAHD dx^2/dt factor
    # uses the full step dt at every substage.
    def step(self, state: BSSNState, t: float, dt: float) -> BSSNState:
        k1 = self.rhs(state, t, dt)
        k2 = self.rhs(tree_map(lambda s, k: s + 0.5 * dt * k, state, k1), t + 0.5 * dt, dt)
        k3 = self.rhs(tree_map(lambda s, k: s + 0.5 * dt * k, state, k2), t + 0.5 * dt, dt)
        k4 = self.rhs(tree_map(lambda s, k: s + dt * k, state, k3), t + dt, dt)
        new = tree_map(lambda s, a, b, c, e: s + (dt / 6.0) * (a + 2 * b + 2 * c + e),
                       state, k1, k2, k3, k4)
        return self.enforce(new)

    def evolve(self, state: BSSNState, dt: float, nsteps: int,
               t0: float = 0.0) -> BSSNState:
        step = jax.jit(lambda s, t: self.step(s, t, dt))

        def body(i, carry):
            s, t = carry
            return step(s, t), t + dt

        s, _ = jax.lax.fori_loop(0, nsteps, body, (state, t0))
        return s
