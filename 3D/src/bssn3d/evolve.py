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
                 precision: str = "fp64", scheme: str = "verbatim", ko_order: int = 8,
                 integrator: str = "rk4"):
        self.grid = grid
        self.params = params or PhysicsParams()
        # ``scheme`` selects the RHS algebra variant (verbatim oracle vs the Phase-3
        # staged/pallas/fused kernels) so the full RK4-step evolution — and the
        # apples/long-run accuracy checks — can run on the optimized kernels, not
        # only the verbatim baseline. Defaults to verbatim (the bit-compared anchor).
        self.scheme = scheme
        self.solver = BSSNSolver(grid, self.params, order,
                                 precision=precision, scheme=scheme, ko_order=ko_order)
        self.diff_op = self.solver.diff_op
        self.ng = grid.ng
        self.ko_sigma = ko_sigma
        self.bc = bc
        # ``integrator`` (Phase-4.C): "rk4" (4 evals/step) or a 2-step/3-stage MSRK
        # ("rk4_2_2" recommended; "rk4_2_1") that reuses the previous step's RHS to do
        # 3 fresh evals/step — ~1.33x at the production Courant factor (CFL=0.1, far below
        # the stability limit so cost is stage-count-limited). One RK4 startup step fills
        # the 1 prior-RHS buffer; the BSSN RHS is time-dependent (SSL ramp) so each MSRK
        # stage is evaluated at its own node time. See `bssn-codegen` memory `msrk-bssn-spectrum`.
        self.integrator = integrator
        self._msrk = None
        if integrator != "rk4":
            from mcs2d.msrk import METHODS
            if integrator not in METHODS or METHODS[integrator].n_prev != 1:
                raise ValueError(f"integrator {integrator!r}: only 'rk4' or a 2-step "
                                 f"rk4_2_* MSRK method are supported here")
            self._msrk = METHODS[integrator]
        # Built once on first evolve() and reused. The WHOLE chunk-loop (fori_loop
        # -> scan, wrapping the huge CSE step) is jitted as one unit, so a chunked
        # long run compiles the scan exactly once instead of once per chunk. dt/t0/
        # nsteps are TRACED args, so a new chunk size or dt does not force a recompile
        # either (only a new grid shape does).
        self._jit_evolve = None
        self._jit_evolve_msrk = None

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
        rate = rate + self._ko(data)             # KO is a separate pass for every scheme (fusing it
        return BSSNState(rate)                   # into the compute-bound cuda_fused kernel lost)

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

    # --- one 2-step/3-stage MSRK step (Phase 4.C) ---
    # carry = (y_n, fprev=f(y_{n-1})). Reuses fprev as stage 0 (node -1, i.e. at t-dt, already
    # computed last step), then 3 fresh RHS evals. Nodes c2,c3 = a-row sums (consistency
    # condition) → each fresh stage is evaluated at its own SSL-ramp time. Returns (enforce(y_{n+1}),
    # k1=f(y_n)) so k1 becomes the next step's fprev. KO/BC/enforce are identical to the RK4 path.
    def _msrk2_step(self, state, fprev, t, dt):
        m = self._msrk
        c2 = m.a20 + m.a21
        c3 = m.a30 + m.a31 + m.a32
        k0 = fprev
        k1 = self.rhs(state, t, dt)
        Y2 = tree_map(lambda s, a, b: s + dt * (m.a20 * a + m.a21 * b), state, k0, k1)
        k2 = self.rhs(Y2, t + c2 * dt, dt)
        Y3 = tree_map(lambda s, a, b, c: s + dt * (m.a30 * a + m.a31 * b + m.a32 * c),
                      state, k0, k1, k2)
        k3 = self.rhs(Y3, t + c3 * dt, dt)
        new = tree_map(lambda s, a, b, c, e: s + dt * (m.b[0] * a + m.b[1] * b
                                                       + m.b[2] * c + m.b[3] * e),
                       state, k0, k1, k2, k3)
        return self.enforce(new), k1

    def evolve(self, state: BSSNState, dt: float, nsteps: int,
               t0: float = 0.0) -> BSSNState:
        # dt and t are passed as TRACED, strongly-typed f64 scalars (not Python
        # floats baked into the graph) so the compiled step has one fixed signature
        # across every chunk and every dt -> compiles exactly once. The jitted step
        # is cached on the instance so a fresh lambda isn't created (and re-traced)
        # per call. (dx is still baked via the stencils, so a new resolution still
        # recompiles once -- but a long run no longer recompiles mid-flight.)
        dt = jnp.asarray(dt, dtype=jnp.float64)
        t0 = jnp.asarray(t0, dtype=jnp.float64)
        nsteps = jnp.asarray(nsteps, dtype=jnp.int64)

        if self.integrator != "rk4":
            # 1 RK4 startup step fills the prior-RHS buffer, then the MSRK recurrence.
            # NOTE: a chunked long run restarts MSRK (1 RK4 step) per evolve() call; for
            # tiny chunks thread fprev across calls. Per-call startup is exact (4th order).
            if self._jit_evolve_msrk is None:
                @jax.jit
                def _evolve_msrk(state, dt, nsteps, t0):
                    f0 = self.rhs(state, t0, dt)              # f(y_0) -> first fprev
                    y1 = self.step(state, t0, dt)            # RK4 startup step
                    def body(i, carry):
                        s, fp, t = carry
                        snew, fnew = self._msrk2_step(s, fp, t, dt)
                        return (snew, fnew, t + dt)
                    s, _, _ = jax.lax.fori_loop(1, nsteps, body, (y1, f0, t0 + dt))
                    return s
                self._jit_evolve_msrk = _evolve_msrk
            return self._jit_evolve_msrk(state, dt, nsteps, t0)

        if self._jit_evolve is None:
            @jax.jit
            def _evolve(state, dt, nsteps, t0):
                def body(i, carry):
                    s, t = carry
                    return self.step(s, t, dt), t + dt
                s, _ = jax.lax.fori_loop(0, nsteps, body, (state, t0))
                return s
            self._jit_evolve = _evolve
        return self._jit_evolve(state, dt, nsteps, t0)
