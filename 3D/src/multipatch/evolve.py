"""RK4 time evolution over a Llama multipatch grid.

Generic over the physical system (scalar wave now; MCS in M4). The state is a
tuple of per-patch field arrays ``(NF, nx, ny, nz)`` — a native JAX pytree, so
``jax.tree_util.tree_map`` does the RK4 vector arithmetic.

Each RHS evaluation (so, every RK4 substage) re-fills ghosts before
differentiating:

  1. ``apply_overlap_fill``  — inter-patch ghosts via precomputed interpolation.
  2. ``outer_bc(fields, t)`` — the genuine outer-boundary ghosts (no donor):
     Dirichlet-from-exact for convergence tests, or radiative/Sommerfeld for
     absorption tests.

then computes the per-patch RHS (curvilinear derivatives) and adds Kreiss-Oliger
dissipation in reference coordinates (sign per ``compute_ko`` — load-bearing).
Only interior values of the RHS matter; ghost values are overwritten by the next
fill, exactly as the uniform solver re-syncs periodic ghosts each substage.
"""
import jax
import jax.numpy as jnp

from .derivative_curvilinear import CurvilinearDerivative
from .overlap import apply_overlap_fill


def _tree_axpy(a, x, y):
    """y + a*x over the per-patch pytree (a scalar)."""
    return jax.tree_util.tree_map(lambda yy, xx: yy + a * xx, y, x)


class MultipatchEvolution:
    """RK4 evolution of ``system`` on ``grid`` with overlap coupling + outer BC."""

    def __init__(self, grid, table, system, order=6, ko_sigma=0.0, outer_bc=None):
        self.grid = grid
        self.table = table
        self.system = system
        self.order = order
        self.ko_sigma = ko_sigma
        self.ops = [CurvilinearDerivative(p, order=order) for p in grid.patches]
        # outer_bc(fields, t) -> fields; default: leave outer ghosts untouched.
        self.outer_bc = outer_bc if outer_bc is not None else (lambda f, t: f)

    # -- ghost fill (inter-patch interpolation + outer BC) ------------------- #
    def fill(self, fields, t):
        fields = apply_overlap_fill(fields, self.table)
        fields = self.outer_bc(fields, t)
        return tuple(fields)

    # -- RHS over all patches ----------------------------------------------- #
    def rhs(self, fields, t):
        fields = self.fill(fields, t)
        out = []
        for pi, F in enumerate(fields):
            d = self.ops[pi]
            r = self.system.rhs_patch(F, d, t)
            if self.ko_sigma > 0.0:
                r = r + jnp.stack([d.ko(F[k], self.ko_sigma)
                                   for k in range(self.system.NF)])
            out.append(r)
        return tuple(out)

    # -- one RK4 step ------------------------------------------------------- #
    def step(self, fields, t, dt):
        k1 = self.rhs(fields, t)
        k2 = self.rhs(_tree_axpy(0.5 * dt, k1, fields), t + 0.5 * dt)
        k3 = self.rhs(_tree_axpy(0.5 * dt, k2, fields), t + 0.5 * dt)
        k4 = self.rhs(_tree_axpy(dt, k3, fields), t + dt)
        new = jax.tree_util.tree_map(
            lambda s, a, b, c, e: s + (dt / 6.0) * (a + 2 * b + 2 * c + e),
            fields, k1, k2, k3, k4)
        return self.fill(new, t + dt)

    def make_jit_step(self):
        """Return a jitted ``step(fields, t, dt)`` (dt/t traced)."""
        return jax.jit(self.step)

    def evolve(self, fields, dt, nsteps, t0=0.0, diag=None):
        """Advance ``nsteps``. If ``diag`` is given it's called as
        ``diag(step_index, t, fields)`` (host-side) and its results collected
        into a returned list. Uses a Python loop with a jitted step (CPU
        prototype — keeps diagnostics simple)."""
        step = self.make_jit_step()
        fields = self.fill(fields, t0)
        t = t0
        records = []
        if diag is not None:
            records.append(diag(0, t, fields))
        for i in range(nsteps):
            fields = step(fields, t, dt)
            t = t0 + (i + 1) * dt
            if diag is not None:
                records.append(diag(i + 1, t, fields))
        return fields, records


# --------------------------------------------------------------------------- #
# Outer boundary conditions
# --------------------------------------------------------------------------- #


def make_exact_dirichlet_bc(grid, table, exact_fn):
    """Outer BC that sets each patch's boundary ghosts to an exact solution.

    ``exact_fn(X, Y, Z, t) -> (NF, *shape)`` returns the analytic state at the
    given world coords and time. Boundary-ghost flat indices come from the
    overlap table (the no-donor outer-radial layers). Used for convergence
    tests: it turns the outer face into a Dirichlet boundary fed by the exact
    solution, isolating interior + seam accuracy.
    """
    # precompute boundary-ghost world coords per patch
    bcoords = []
    for p, bidx in zip(grid.patches, table.boundary_idx):
        Xf = p.X.ravel()[bidx]
        Yf = p.Y.ravel()[bidx]
        Zf = p.Z.ravel()[bidx]
        bcoords.append((Xf, Yf, Zf))

    def bc(fields, t):
        fields = list(fields)
        for pi, (p, bidx) in enumerate(zip(grid.patches, table.boundary_idx)):
            if bidx.shape[0] == 0:
                continue
            Xf, Yf, Zf = bcoords[pi]
            vals = exact_fn(Xf, Yf, Zf, t)            # (NF, B)
            NFv = vals.shape[0]
            F = fields[pi]
            F = F.reshape(NFv, -1).at[:, bidx].set(vals).reshape(F.shape)
            fields[pi] = F
        return fields

    return bc


def make_sommerfeld_bc(grid, table):
    """Radiative outgoing outer BC by radial extrapolation of the boundary
    ghosts (approximate; for absorption/stability tests, not exact).

    Fills each outer-radial ghost by copying the nearest interior-radial value
    along the same angular line (a 0th-order outgoing fill). Combined with KO
    this damps outgoing energy without reflecting strongly. Good enough for a
    prototype 'how does the outer BC behave' probe; a full Bayley-Sommerfeld
    recipe is a later upgrade.
    """
    ng = grid.ng
    # Only shells have outer-radial boundaries; their boundary ghosts are the
    # high-lz layers. We copy from the last interior radial index.
    def bc(fields, t):
        fields = list(fields)
        for pi, p in enumerate(grid.patches):
            bidx = table.boundary_idx[pi]
            if bidx.shape[0] == 0:
                continue
            nz = p.shape[2]
            Nz = p.N[2]
            F = fields[pi]
            # last interior radial layer index:
            last = ng + Nz - 1
            for g in range(1, ng + 1):
                F = F.at[:, :, :, last + g].set(F[:, :, :, last])
            fields[pi] = F
        return fields

    return bc
