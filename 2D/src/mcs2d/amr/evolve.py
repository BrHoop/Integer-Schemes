"""
RK4 stepping for block-structured AMR (periodic root BC, shared dt).

Reuses the per-block RHS kernel from `fused_rhs_fp._make_kernel_fn` and the
ghost-zone sync from `amr.kernels`.  Three step builders, in increasing scope:

  * `make_root_step`     — level 0 only (the periodic root tiling).
  * `make_two_level_step`— root + one fine level; supports optional restriction.
  * `make_n_level_step`  — the full LEVELS-deep hierarchy.

All share a single dt across levels (no sub-cycling — that's Phase 3).  Each
returns a @jax.jit step compiled once and reused across regrid events: topology
is passed at call time, so its changing *values* (fixed *shape*) never retrace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mcs2d.amr.state import BS, NG, NF, LEVELS, mb, AMRState, AMRTopologyArrays


def _set_level(blocks: tuple, L: int, arr) -> tuple:
    """Return a new per-level block tuple with element L replaced by `arr`
    (the tuple analogue of `stacked_array.at[L].set(arr)`).  L is a Python int,
    so this just rebuilds the tuple of references — no data copy."""
    return blocks[:L] + (arr,) + blocks[L + 1:]
from mcs2d.amr.kernels import (
    sync_ghosts_within_level_root_periodic,
    sync_ghosts_within_level,
    sync_ghosts_across_levels,
)
from mcs2d.schemes.fused_rhs_fp import _make_kernel_fn


def make_root_step(
    dx, dy, cs, L, K1, K2, ko_sigma, dt,
    nbx: int, nby: int,
):
    """Build a JIT-compiled single RK4 step on the AMR root-level blocks.

    Args:
      dx, dy:            grid spacing at the root level.
      cs, L, K1, K2:     MCS physics parameters (matches main solver).
      ko_sigma:          Kreiss-Oliger dissipation coefficient.
      dt:                time step.
      nbx, nby:          static — root tiling.

    Returns:
      step(blocks_level0) → blocks_level0  (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG).
      Only the first nbx*nby slots carry meaningful data; the rest pass through.

    The returned state has its halo cells re-synced for periodic BC, so it's
    ready to be inspected or passed back in for another step.
    """
    kernel_fn = _make_kernel_fn(dx, dy, cs, L, K1, K2, ko_sigma)

    def rhs_all(blocks):
        # vmap kernel_fn over MAX_BLOCKS slots → (MAX_BLOCKS, NF, BS, BS)
        return jax.vmap(kernel_fn)(blocks)

    def add_to_interior(blocks, increment):
        """Add `increment` (shape (M, NF, BS, BS)) to each block's INTERIOR."""
        return blocks.at[:, :, NG:NG + BS, NG:NG + BS].add(increment)

    def set_interior(blocks, new_interior):
        return blocks.at[:, :, NG:NG + BS, NG:NG + BS].set(new_interior)

    @jax.jit
    def step(blocks_level0):
        # blocks_level0: (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG)
        # Pre-sync (assume caller has valid INTERIOR but possibly stale halo).
        u0 = sync_ghosts_within_level_root_periodic(blocks_level0, nbx, nby)
        u0_int = u0[:, :, NG:NG + BS, NG:NG + BS]

        # Stage 1
        k1 = rhs_all(u0)

        # Stage 2: u1 = u0 + dt/2 * k1
        u1 = set_interior(u0, u0_int + 0.5 * dt * k1)
        u1 = sync_ghosts_within_level_root_periodic(u1, nbx, nby)
        k2 = rhs_all(u1)

        # Stage 3: u2 = u0 + dt/2 * k2
        u2 = set_interior(u0, u0_int + 0.5 * dt * k2)
        u2 = sync_ghosts_within_level_root_periodic(u2, nbx, nby)
        k3 = rhs_all(u2)

        # Stage 4: u3 = u0 + dt * k3
        u3 = set_interior(u0, u0_int + dt * k3)
        u3 = sync_ghosts_within_level_root_periodic(u3, nbx, nby)
        k4 = rhs_all(u3)

        # Combine: u_new = u0 + dt/6 * (k1 + 2 k2 + 2 k3 + k4)
        u_new_int = u0_int + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u_new = set_interior(u0, u_new_int)
        # Final sync so caller sees correct halo.
        return sync_ghosts_within_level_root_periodic(u_new, nbx, nby)

    return step


def make_two_level_step(
    dx_coarse, dy_coarse, dt,
    cs, L, K1, K2, ko_sigma,
    nbx_root: int, nby_root: int,
    *,
    restrict_at_end: bool = False,
):
    """Build a JIT-compiled single RK4 step on a 2-level AMR state.

    Phase 1/2 hybrid (no sub-cycling yet): both levels step at the same dt.
    Per step:
      1. Sync coarse periodic halos.
      2. Sync fine halos from coarse parents.
      3. Advance both levels through 4 RK4 stages (with halo sync between).
      4. (optional, `restrict_at_end=True`) Restrict fine interiors into matching
         coarse parent quadrants, then re-sync.

    Topology arrays (`parent_slot`, `child_cx`, `child_cy`, `fine_active`) are
    passed at CALL TIME, not bound at construction.  This means a single
    compiled step survives regridding events as long as shapes stay constant —
    which they always do (MAX_BLOCKS is a Python constant).  Required for the
    Phase 2 zero-recompile-on-regrid promise.

    Note on `restrict_at_end`
    -------------------------
    Conservative 2×2 averaging of the fine interior into the coarse parent is
    available via this flag but is OFF BY DEFAULT.  Without flux correction at
    the coarse-fine boundary (planned for Phase 3 alongside sub-cycling), the
    restricted-vs-non-restricted coarse cells create a value jump that the
    6th-order divergence stencil picks up as ~O(1e-3) spurious divergence,
    violating constraint conservation by orders of magnitude.

    Until flux correction lands, leave `restrict_at_end=False` for physics runs
    (this matches the validated Phase 1 behavior).  Set it to True only for
    tests that verify the restriction mechanism works correctly in isolation.

    Returns:
      step(coarse_blocks, fine_blocks, parent_slot, child_cx, child_cy,
           fine_active) → (coarse_blocks, fine_blocks),
      both with halo cells re-synced.
    """
    from mcs2d.amr.kernels import (
        sync_ghosts_across_levels, restrict_all_into_parents,
    )

    dx_fine = dx_coarse / 2.0
    dy_fine = dy_coarse / 2.0

    kernel_coarse = _make_kernel_fn(dx_coarse, dy_coarse, cs, L, K1, K2, ko_sigma)
    kernel_fine   = _make_kernel_fn(dx_fine,   dy_fine,   cs, L, K1, K2, ko_sigma)

    def rhs_coarse(blocks): return jax.vmap(kernel_coarse)(blocks)
    def rhs_fine(blocks):   return jax.vmap(kernel_fine)(blocks)

    def sync_c(blocks):
        return sync_ghosts_within_level_root_periodic(blocks, nbx_root, nby_root)

    def set_interior(blocks, new_int):
        return blocks.at[:, :, NG:NG + BS, NG:NG + BS].set(new_int)

    @jax.jit
    def step(coarse_blocks, fine_blocks,
             parent_slot, child_cx, child_cy, fine_active):

        def sync_f(fine_blocks, coarse_blocks):
            return sync_ghosts_across_levels(
                fine_blocks, coarse_blocks,
                parent_slot, child_cx, child_cy, fine_active,
            )
        # Pre-sync.
        c0 = sync_c(coarse_blocks)
        f0 = sync_f(fine_blocks, c0)
        c0_int = c0[:, :, NG:NG + BS, NG:NG + BS]
        f0_int = f0[:, :, NG:NG + BS, NG:NG + BS]

        # Stage 1
        kc1 = rhs_coarse(c0)
        kf1 = rhs_fine(f0)

        # Stage 2: u1 = u0 + dt/2 * k1
        c1 = sync_c(set_interior(c0, c0_int + 0.5 * dt * kc1))
        f1 = sync_f(set_interior(f0, f0_int + 0.5 * dt * kf1), c1)
        kc2 = rhs_coarse(c1)
        kf2 = rhs_fine(f1)

        # Stage 3: u2 = u0 + dt/2 * k2
        c2 = sync_c(set_interior(c0, c0_int + 0.5 * dt * kc2))
        f2 = sync_f(set_interior(f0, f0_int + 0.5 * dt * kf2), c2)
        kc3 = rhs_coarse(c2)
        kf3 = rhs_fine(f2)

        # Stage 4: u3 = u0 + dt * k3
        c3 = sync_c(set_interior(c0, c0_int + dt * kc3))
        f3 = sync_f(set_interior(f0, f0_int + dt * kf3), c3)
        kc4 = rhs_coarse(c3)
        kf4 = rhs_fine(f3)

        c_new_int = c0_int + (dt / 6.0) * (kc1 + 2.0 * kc2 + 2.0 * kc3 + kc4)
        f_new_int = f0_int + (dt / 6.0) * (kf1 + 2.0 * kf2 + 2.0 * kf3 + kf4)

        c_new = set_interior(c0, c_new_int)
        f_new = set_interior(f0, f_new_int)

        if restrict_at_end:
            c_new = restrict_all_into_parents(
                c_new, f_new, parent_slot, child_cx, child_cy, fine_active,
            )

        # Final ghost sync so the returned state has valid halos everywhere.
        c_new = sync_c(c_new)
        f_new = sync_f(f_new, c_new)
        return c_new, f_new

    return step


# ── Cubic Hermite time interpolation ──────────────────────────────────────────

def _hermite_basis(s: float):
    """Cubic Hermite basis weights at normalized time s ∈ [0, 1].

    Returns (h00, h10, h01, h11) for the interpolant
        H(s) = h00·f0 + h10·dt·f0' + h01·f1 + h11·dt·f1'
    which matches values f0, f1 and derivatives f0', f1' at s=0, 1.  Exact for
    cubics → O(dt⁴) interpolation error (vs O(dt²) for linear), so the coarse-
    fine time boundary no longer caps the 4th-order RK4 temporal accuracy.

    `s` is a Python float (the sub-cycle stage fractions are static), so these
    weights are computed at trace time — no traced control flow.
    """
    s2 = s * s
    s3 = s2 * s
    return (2*s3 - 3*s2 + 1,      # h00
            s3 - 2*s2 + s,        # h10
            -2*s3 + 3*s2,         # h01
            s3 - s2)              # h11


# ── 2-level Berger-Oliger sub-cycled step ─────────────────────────────────────

def make_subcycled_two_level_step(
    dx_coarse, dy_coarse, dt_coarse,
    cs, L, K1, K2, ko_sigma,
    nbx_root: int, nby_root: int,
):
    """Build a JIT-compiled Berger-Oliger sub-cycled step for a 2-level state.

    One call advances BOTH levels by `dt_coarse`:
      1. Coarse level takes ONE RK4 step of dt_coarse (it is the root → periodic
         halos).
      2. Fine level takes TWO RK4 substeps of dt_fine = dt_coarse / 2.  During
         each fine RK4 stage the fine halo is filled by prolongating the coarse
         state interpolated IN TIME to that stage's instant — cubic Hermite
         using the coarse endpoint values AND time-derivatives (the coarse RHS,
         already computed), giving 4th-order time interpolation that matches the
         RK4 temporal order.
      3. After the fine level catches up, restrict the fine interior into the
         coarse parent (Berger-Oliger correction).

    This is the classical AMR time-stepping: the root advances at its own CFL
    while the fine level — which needs a smaller dt for ITS CFL — sub-cycles,
    rather than forcing the whole hierarchy onto the finest dt (the make_*_step
    "shared-dt" simplification, which wastes 2^(depth-1)× root work).

    Topology arrays are call-time args (shape fixed) → one compile, reused
    across regrids.  Single fine block / cross-level-only halos for now;
    multi-fine-block within-level sync is a separate Phase 3 item.

    Returns:
      step(coarse_blocks, fine_blocks, parent_slot, child_cx, child_cy,
           fine_active) → (coarse_blocks, fine_blocks).

    Time-interpolation order
    ------------------------
    The coarse-fine boundary uses cubic-Hermite (4th-order) time interpolation
    (`_hermite_basis`), matching the RK4 temporal order so the boundary does
    not cap temporal convergence.  The endpoint time-derivatives come from the
    coarse RHS already evaluated during the coarse RK4 (one extra coarse RHS at
    t+dt_c) — negligible cost versus the fine sub-cycle.
    """
    from mcs2d.amr.kernels import (
        sync_ghosts_across_levels, restrict_all_into_parents_highorder,
    )

    dx_fine = dx_coarse / 2.0
    dy_fine = dy_coarse / 2.0
    dt_fine = dt_coarse / 2.0

    kernel_coarse = _make_kernel_fn(dx_coarse, dy_coarse, cs, L, K1, K2, ko_sigma)
    kernel_fine   = _make_kernel_fn(dx_fine,   dy_fine,   cs, L, K1, K2, ko_sigma)

    def rhs_coarse(b): return jax.vmap(kernel_coarse)(b)
    def rhs_fine(b):   return jax.vmap(kernel_fine)(b)

    def sync_c(b):
        return sync_ghosts_within_level_root_periodic(b, nbx_root, nby_root)

    def set_interior(b, ni):
        return b.at[:, :, NG:NG + BS, NG:NG + BS].set(ni)

    @jax.jit
    def step(coarse_blocks, fine_blocks,
             parent_slot, child_cx, child_cy, fine_active):

        def sync_f_from(fine, coarse):
            return sync_ghosts_across_levels(
                fine, coarse, parent_slot, child_cx, child_cy, fine_active,
            )

        # ── 1. Coarse RK4, one full dt_coarse ────────────────────────────────
        c0 = sync_c(coarse_blocks)
        c0i = c0[:, :, NG:NG + BS, NG:NG + BS]
        kc1 = rhs_coarse(c0)        # = du/dt at t (interior), reused for Hermite
        c1 = sync_c(set_interior(c0, c0i + 0.5 * dt_coarse * kc1)); kc2 = rhs_coarse(c1)
        c2 = sync_c(set_interior(c0, c0i + 0.5 * dt_coarse * kc2)); kc3 = rhs_coarse(c2)
        c3 = sync_c(set_interior(c0, c0i + dt_coarse * kc3));        kc4 = rhs_coarse(c3)
        c_new = sync_c(set_interior(
            c0, c0i + (dt_coarse / 6.0) * (kc1 + 2.0 * kc2 + 2.0 * kc3 + kc4)
        ))   # coarse at t + dt_coarse, halos synced

        # Cubic-Hermite-in-time coarse state for the fine boundary.  Endpoint
        # interior derivatives: d0 = du/dt(t) (= kc1) and d1 = du/dt(t+dt_c).
        # Interpolate the INTERIOR with Hermite, then periodic-sync the halo —
        # for the root parent that keeps the whole block 4th-order in time (the
        # halo is just periodic copies of the 4th-order interior).
        cnew_i = c_new[:, :, NG:NG + BS, NG:NG + BS]
        d0_i = kc1
        d1_i = rhs_coarse(c_new)
        def coarse_at(frac):           # frac ∈ [0, 1] over the coarse step
            h00, h10, h01, h11 = _hermite_basis(frac)
            int_s = (h00 * c0i + h01 * cnew_i
                     + dt_coarse * (h10 * d0_i + h11 * d1_i))
            return sync_c(set_interior(c0, int_s))

        # ── 2. Fine: two RK4 substeps of dt_fine, per-stage time-interp halo ──
        fine = fine_blocks
        for sub in range(2):           # static: unrolled at trace time
            frac0 = 0.5 * sub          # substep start: 0.0 then 0.5
            f0i = fine[:, :, NG:NG + BS, NG:NG + BS]
            fa = sync_f_from(set_interior(fine, f0i),                      coarse_at(frac0));        kf1 = rhs_fine(fa)
            fb = sync_f_from(set_interior(fine, f0i + 0.5 * dt_fine * kf1), coarse_at(frac0 + 0.25)); kf2 = rhs_fine(fb)
            fc = sync_f_from(set_interior(fine, f0i + 0.5 * dt_fine * kf2), coarse_at(frac0 + 0.25)); kf3 = rhs_fine(fc)
            fd = sync_f_from(set_interior(fine, f0i + dt_fine * kf3),       coarse_at(frac0 + 0.5));  kf4 = rhs_fine(fd)
            fine = set_interior(fine, f0i + (dt_fine / 6.0) * (kf1 + 2.0 * kf2 + 2.0 * kf3 + kf4))
        # fine now at t + dt_coarse.  Sync its halo from the new coarse before
        # restricting — 6th-order restriction reaches into the halo, so it must
        # be valid (this matches what the N-level recursion does via filled()).
        fine = sync_f_from(fine, c_new)

        # ── 3. Restrict fine → coarse parent (6th-order), re-sync both levels ─
        c_final = restrict_all_into_parents_highorder(
            c_new, fine, parent_slot, child_cx, child_cy, fine_active,
        )
        c_final = sync_c(c_final)
        fine = sync_f_from(fine, c_final)
        return c_final, fine

    return step


# ── N-level recursive Berger-Oliger sub-cycled step — UNROLLED (reference) ────
# Kept for A/B comparison; the DEFAULT `make_subcycled_n_level_step` below is the
# rolled (lax.scan) version, which is bit-identical but compiles in O(LEVELS)
# instead of O(2^LEVELS).  See `make_subcycled_n_level_step` for the rationale.

def make_subcycled_n_level_step_unrolled(
    dx_root, dy_root, dt_root,
    cs, L_coupling, K1, K2, ko_sigma,
    nbx_root: int, nby_root: int,
):
    """Build a JIT-compiled recursive Berger-Oliger sub-cycled step (UNROLLED).

    Reference implementation: the two substeps at each level are two inlined
    Python calls, so the traced graph grows like 2^LEVELS — compile time is
    exponential in depth (~2.1×/level measured).  The default
    `make_subcycled_n_level_step` rolls those substeps into a `lax.scan` and is
    bit-identical but compiles ~linearly in depth; prefer it.  This one is kept
    for equivalence testing and small/shallow runs.

    One call advances the whole LEVELS-deep hierarchy by `dt_root`.  Level L
    uses dt_L = dt_root / 2^L and takes 2 substeps per parent step (refinement
    ratio 2), so the finest level takes 2^(LEVELS-1) substeps — each level runs
    at its own CFL instead of the whole hierarchy being forced onto the finest
    dt (the `make_n_level_step` shared-dt simplification).

    Algorithm (classical Berger-Oliger, unrolled at trace time so the graph is
    static → one compile, reused across regrids):

        advance(L, dt_L, parent brackets):
          1. RK4-advance level L by dt_L.  Each RK stage fills level L's halo
             from its parent, cubic-Hermite-interpolated IN TIME to the stage's
             instant (periodic sync for the root, L=0).
          2. If a finer level exists: take 2 substeps of advance(L+1, dt_L/2),
             passing level-L (value, time-derivative) brackets so L+1 can
             Hermite-interpolate its boundary; then restrict L+1 → L.

    (value, derivative) FULL-block pairs are threaded down the recursion: a
    level's halo time-derivative is the prolongation of its parent's time-
    derivative (periodic copy at the root), so Hermite interpolation of the
    parent full block — interior AND halo — is 4th-order in time at every level.

    Topology arrays are call-time args (fixed shape) → no recompile on regrid.

    Returns:
      step(state, topology) → state  (AMRState, AMRTopologyArrays).
    """
    from mcs2d.amr.kernels import (
        sync_ghosts_across_levels, sync_ghosts_within_level,
        restrict_all_into_parents_highorder,
    )

    kernels = [
        _make_kernel_fn(dx_root / (2 ** L), dy_root / (2 ** L),
                        cs, L_coupling, K1, K2, ko_sigma)
        for L in range(LEVELS)
    ]

    @jax.jit
    def step(state: AMRState, topology: AMRTopologyArrays) -> AMRState:
        blocks = state.blocks
        active = state.active

        def set_int(level_blk, new_int):
            return level_blk.at[:, :, NG:NG + BS, NG:NG + BS].set(new_int)

        def rhs(L, level_blk):
            return jax.vmap(kernels[L])(level_blk)   # (MAX_BLOCKS, NF, BS, BS)

        def sync_halo(L, level_blk, parent_full):
            """Fill level-L halo: periodic (root); else prolongate parent_full
            then override shared faces with same-level neighbours' exact data.
            Applied identically to the state and its time-derivative."""
            if L == 0:
                return sync_ghosts_within_level_root_periodic(level_blk, nbx_root, nby_root)
            synced = sync_ghosts_across_levels(
                level_blk, parent_full,
                topology.parent_slot[L], topology.child_cx[L], topology.child_cy[L],
                active[L],
            )
            return sync_ghosts_within_level(
                synced, topology.neighbor_slot[L], topology.neighbor_valid[L],
            )

        def full_deriv(L, level_full, parent_deriv_full):
            """Full-block time-derivative of level L: interior from the RHS
            kernel, halo from the same sync as the state (periodic copy of the
            interior derivative at the root; prolongation of the parent's
            derivative for L≥1)."""
            di = rhs(L, level_full)                       # interior derivative
            dz = set_int(jnp.zeros_like(level_full), di)
            return sync_halo(L, dz, parent_deriv_full)

        def hermite(lo, hi, dlo, dhi, dt_L, s):
            h00, h10, h01, h11 = _hermite_basis(s)
            return h00 * lo + h01 * hi + dt_L * (h10 * dlo + h11 * dhi)

        def advance(blocks, L, dt_L, par_lo, par_hi, dpar_lo, dpar_hi):
            level0 = blocks[L]
            u0 = level0[:, :, NG:NG + BS, NG:NG + BS]

            def parent_at(s):     # parent full block, Hermite-in-time (unused at L=0)
                if L == 0:
                    return None
                return hermite(par_lo, par_hi, dpar_lo, dpar_hi, dt_L, s)

            def filled(new_int, s):
                return sync_halo(L, set_int(level0, new_int), parent_at(s))

            a = filled(u0, 0.0);                       k1 = rhs(L, a)
            b = filled(u0 + 0.5 * dt_L * k1, 0.5);     k2 = rhs(L, b)
            c = filled(u0 + 0.5 * dt_L * k2, 0.5);     k3 = rhs(L, c)
            d = filled(u0 + dt_L * k3, 1.0);           k4 = rhs(L, d)
            u_new = u0 + (dt_L / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            lvl_old = a                       # level L full block @ frac 0
            lvl_new = filled(u_new, 1.0)      # level L full block @ frac 1
            blocks = _set_level(blocks, L, lvl_new)

            if L < LEVELS - 1:
                # Parent-derivative brackets for THIS level (linear interp of the
                # passed parent derivative; only feeds halo derivatives, a thin
                # correction).  Root has no parent derivative → None.
                def dpar_at(s):
                    if L == 0:
                        return None
                    return dpar_lo + s * (dpar_hi - dpar_lo)

                d_old = full_deriv(L, lvl_old, dpar_at(0.0))
                d_new = full_deriv(L, lvl_new, dpar_at(1.0))
                # Level-L state + derivative at the sub-bracket midpoint.
                lvl_mid = hermite(lvl_old, lvl_new, d_old, d_new, dt_L, 0.5)
                d_mid = full_deriv(L, lvl_mid, dpar_at(0.5))

                dt_f = dt_L / 2.0
                blocks = advance(blocks, L + 1, dt_f, lvl_old, lvl_mid, d_old, d_mid)
                blocks = advance(blocks, L + 1, dt_f, lvl_mid, lvl_new, d_mid, d_new)

                # Restrict L+1 → L (6th-order).
                new_coarse = restrict_all_into_parents_highorder(
                    blocks[L], blocks[L + 1],
                    topology.parent_slot[L + 1], topology.child_cx[L + 1],
                    topology.child_cy[L + 1], active[L + 1],
                )
                blocks = _set_level(blocks, L, new_coarse)

            return blocks

        blocks = advance(blocks, 0, dt_root, None, None, None, None)
        return AMRState(blocks=blocks, active=active)

    return step


# ── N-level Berger-Oliger sub-cycled step — DEFAULT (rolled, lax.scan) ─────────

def make_subcycled_n_level_step(
    dx_root, dy_root, dt_root,
    cs, L_coupling, K1, K2, ko_sigma,
    nbx_root: int, nby_root: int,
):
    """Build a JIT-compiled recursive Berger-Oliger sub-cycled step (DEFAULT).

    Differs from `make_subcycled_n_level_step_unrolled` in two performance-only ways
    (numerically equivalent to machine precision; same signature and API):

    1. COMPILE-TIME (C0).  The two substeps at each level run through a `lax.scan`
       instead of two inlined Python calls.  Inlining `advance(L+1)` twice per level
       makes the graph (and compile time) grow like 2^LEVELS (~2.1×/level → ~24 h at
       LEVELS≈15); the scan traces each level's body ONCE → O(LEVELS) nested `while`
       loops, ~linear compile (measured 9→16→23→29→34 s for LEVELS 2→6 vs
       4→10→25→53→115 s unrolled; L=15 ≈ 90 s).  This part is bit-identical.

    2. MEMORY (M1a).  Prolongation is a LINEAR operator, so the parent brackets are
       prolonged ONCE per `advance` and each RK substep / derivative halo is formed
       by a cheap Hermite/linear combine — `prolong(hermite(b, s)) ==
       hermite(prolong(b), s)` — instead of re-prolongating on every stage (~8→4
       prolongations/advance; prolongation intermediates were the dominant memory
       cost on the memory-bound GPU profile).  This reassociates the prolongation FP,
       so the rolled step matches the unrolled reference to ~1e-13, not bit-for-bit.

    Ragged per-level storage and static-`L` kernels are preserved; the step still
    traces exactly once across regrids.
    """
    from mcs2d.amr.kernels import (
        sync_ghosts_within_level, prolong_all, apply_prolonged_halo,
        restrict_all_into_parents_highorder,
    )

    kernels = [
        _make_kernel_fn(dx_root / (2 ** L), dy_root / (2 ** L),
                        cs, L_coupling, K1, K2, ko_sigma)
        for L in range(LEVELS)
    ]

    @jax.jit
    def step(state: AMRState, topology: AMRTopologyArrays) -> AMRState:
        blocks = state.blocks
        active = state.active

        def set_int(level_blk, new_int):
            return level_blk.at[:, :, NG:NG + BS, NG:NG + BS].set(new_int)

        def rhs(L, level_blk):
            return jax.vmap(kernels[L])(level_blk)

        def hermite(lo, hi, dlo, dhi, dt_L, s):
            h00, h10, h01, h11 = _hermite_basis(s)
            return h00 * lo + h01 * hi + dt_L * (h10 * dlo + h11 * dhi)

        def advance(blocks, L, dt_L, par_lo, par_hi, dpar_lo, dpar_hi):
            level0 = blocks[L]
            u0 = level0[:, :, NG:NG + BS, NG:NG + BS]

            # M1a — prolongation is LINEAR, so prolong the 4 parent brackets ONCE
            # and form each RK substep / derivative halo by a cheap Hermite/linear
            # combine: prolong(hermite(brackets, s)) == hermite(prolong(brackets), s).
            # Replaces ~8 full prolongations per advance (one per stage/deriv) with 4
            # — the dominant memory cost (prolongation intermediates).  L=0 has no
            # parent (periodic halo) so nothing is prolonged there.
            if L > 0:
                def _prol(parent_full):   # raw prolong (no mask — apply_prolonged_halo masks once)
                    return prolong_all(parent_full, topology.parent_slot[L],
                                       topology.child_cx[L], topology.child_cy[L])
                PH_lo, PH_hi   = _prol(par_lo),  _prol(par_hi)    # prolonged value brackets
                PDH_lo, PDH_hi = _prol(dpar_lo), _prol(dpar_hi)   # prolonged deriv brackets

            def within(blk):
                return sync_ghosts_within_level(
                    blk, topology.neighbor_slot[L], topology.neighbor_valid[L])

            def filled(new_int, s):
                """Level-L state with halo = prolonged parent state, Hermite-in-time."""
                blk = set_int(level0, new_int)
                if L == 0:
                    return sync_ghosts_within_level_root_periodic(blk, nbx_root, nby_root)
                halo_s = hermite(PH_lo, PH_hi, PDH_lo, PDH_hi, dt_L, s)
                return within(apply_prolonged_halo(blk, halo_s, active[L]))

            def deriv(level_full, s):
                """Full-block time-derivative; halo = prolonged parent DERIV, linear-in-s."""
                dz = set_int(jnp.zeros_like(level_full), rhs(L, level_full))
                if L == 0:
                    return sync_ghosts_within_level_root_periodic(dz, nbx_root, nby_root)
                halo_s = PDH_lo + s * (PDH_hi - PDH_lo)
                return within(apply_prolonged_halo(dz, halo_s, active[L]))

            a = filled(u0, 0.0);                       k1 = rhs(L, a)
            b = filled(u0 + 0.5 * dt_L * k1, 0.5);     k2 = rhs(L, b)
            c = filled(u0 + 0.5 * dt_L * k2, 0.5);     k3 = rhs(L, c)
            d = filled(u0 + dt_L * k3, 1.0);           k4 = rhs(L, d)
            u_new = u0 + (dt_L / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            lvl_old = a
            lvl_new = filled(u_new, 1.0)
            blocks = _set_level(blocks, L, lvl_new)

            if L < LEVELS - 1:
                d_old = deriv(lvl_old, 0.0)
                d_new = deriv(lvl_new, 1.0)
                lvl_mid = hermite(lvl_old, lvl_new, d_old, d_new, dt_L, 0.5)
                d_mid = deriv(lvl_mid, 0.5)

                dt_f = dt_L / 2.0
                # ROLLED: the two substeps differ only in their parent brackets.
                # Stack the brackets and scan — `advance(L+1)` is traced ONCE.
                #   substep 0: (lvl_old, lvl_mid, d_old, d_mid)
                #   substep 1: (lvl_mid, lvl_new, d_mid, d_new)
                lo  = jnp.stack([lvl_old, lvl_mid])
                hi  = jnp.stack([lvl_mid, lvl_new])
                dlo = jnp.stack([d_old,   d_mid])
                dhi = jnp.stack([d_mid,   d_new])

                def body(carry, x):
                    lo_i, hi_i, dlo_i, dhi_i = x
                    return advance(carry, L + 1, dt_f, lo_i, hi_i, dlo_i, dhi_i), None

                blocks, _ = jax.lax.scan(body, blocks, (lo, hi, dlo, dhi))

                new_coarse = restrict_all_into_parents_highorder(
                    blocks[L], blocks[L + 1],
                    topology.parent_slot[L + 1], topology.child_cx[L + 1],
                    topology.child_cy[L + 1], active[L + 1],
                )
                blocks = _set_level(blocks, L, new_coarse)

            return blocks

        blocks = advance(blocks, 0, dt_root, None, None, None, None)
        return AMRState(blocks=blocks, active=active)

    return step


# ── N-level shared-dt step ────────────────────────────────────────────────────

def make_n_level_step(
    dx_root, dy_root, dt,
    cs, L_coupling, K1, K2, ko_sigma,
    nbx_root: int, nby_root: int,
):
    """Build a JIT-compiled single RK4 step on an N-level AMR state.

    The hierarchy holds `LEVELS` levels (the AMRState array dimension, set
    globally via MCS_AMR_LEVELS).  Inactive levels/slots cost compute but are
    masked out — the price of shape-stable JIT.

    Shared dt — all levels evolve at the same time step (Phase 2 simplification;
    Berger-Oliger sub-cycling is Phase 3).  Choose `dt = CFL * dx_root / 2^(LEVELS-1)`
    to respect the finest CFL (or `/2^max_active_level` if you cap depth).

    Per step:
      1. Sync level-0 periodic halos.
      2. For each fine level L = 1..LEVELS-1: prolongate parent → fill L's halo.
      3. Advance all levels through 4 RK4 stages with halo sync between.
      4. Final halo sync.

    Cross-level coupling (restriction) is deferred until flux correction lands
    in Phase 3 — naive restriction degrades constraint conservation at the
    coarse-fine boundary (see `make_two_level_step` docstring for details).

    Args:
      dx_root, dy_root: cell size at level 0.  Level L has dx_root / 2^L.
      dt:               global time step (one RK4 step advances all levels by dt).
      cs, L_coupling, K1, K2, ko_sigma: MCS physics parameters.
      nbx_root, nby_root: root tiling (static).

    Returns:
      step(state, topology) → state, where `state` is an AMRState and
      `topology` is an AMRTopologyArrays snapshot.  The returned state has
      halos re-synced at every level.

    No-recompile property
    ---------------------
    Topology arrays are runtime arguments — feeding a fresh `topology` from a
    post-regrid `AMRTopology.to_jax_arrays()` does NOT cause re-tracing.
    Validated by `tests/regression/test_no_recompile.py`.
    """
    # Per-level kernels (one for each level; baked at construction).
    kernels = []
    for L in range(LEVELS):
        dx_L = dx_root / (2 ** L)
        dy_L = dy_root / (2 ** L)
        kernels.append(
            _make_kernel_fn(dx_L, dy_L, cs, L_coupling, K1, K2, ko_sigma)
        )

    def rhs_all(blocks):
        """blocks: per-level tuple → returns per-level tuple of (mb_L, NF, BS, BS).

        Computes RHS for every slot at every level (active or not).  Inactive
        slots produce garbage that is masked when consumed; this uniform compute
        is what keeps the function shape-stable (no recompile on regrid).  The
        ragged per-level sizing (MAX_BLOCKS_PER_LEVEL) is what bounds the waste."""
        return tuple(jax.vmap(kernels[L])(blocks[L]) for L in range(LEVELS))

    def sync_all(blocks, active, topology):
        """Sync halos at every level: level 0 periodic; each fine level from its
        parent (cross-level prolongation) THEN from same-level neighbours
        (within-level sync, which overrides the prolongated value on shared
        faces with the neighbour's exact fine data)."""
        synced = list(blocks)
        synced[0] = sync_ghosts_within_level_root_periodic(synced[0], nbx_root, nby_root)
        for L in range(1, LEVELS):
            synced[L] = sync_ghosts_across_levels(
                synced[L], synced[L - 1],
                topology.parent_slot[L], topology.child_cx[L], topology.child_cy[L],
                active[L],
            )
            synced[L] = sync_ghosts_within_level(
                synced[L], topology.neighbor_slot[L], topology.neighbor_valid[L],
            )
        return tuple(synced)

    def interiors(blocks):
        return tuple(b[:, :, NG:NG + BS, NG:NG + BS] for b in blocks)

    def set_interiors(blocks, new_int):
        """Write each level's interior from the per-level tuple `new_int`."""
        return tuple(
            b.at[:, :, NG:NG + BS, NG:NG + BS].set(ni)
            for b, ni in zip(blocks, new_int)
        )

    def axpy(b_int, a, k):
        """Per-level  b_int + a*k  (tuples)."""
        return tuple(bi + a * ki for bi, ki in zip(b_int, k))

    @jax.jit
    def step(state: AMRState, topology: AMRTopologyArrays) -> AMRState:
        active = state.active

        b0 = sync_all(state.blocks, active, topology)
        b0_int = interiors(b0)

        k1 = rhs_all(b0)
        b1 = sync_all(set_interiors(b0, axpy(b0_int, 0.5 * dt, k1)), active, topology)
        k2 = rhs_all(b1)
        b2 = sync_all(set_interiors(b0, axpy(b0_int, 0.5 * dt, k2)), active, topology)
        k3 = rhs_all(b2)
        b3 = sync_all(set_interiors(b0, axpy(b0_int, dt, k3)), active, topology)
        k4 = rhs_all(b3)

        # new_int = b0_int + dt/6 (k1 + 2 k2 + 2 k3 + k4), per level.
        new_int = tuple(
            bi + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
            for bi, a, b, c, d in zip(b0_int, k1, k2, k3, k4)
        )
        new_blocks = sync_all(set_interiors(b0, new_int), active, topology)
        return AMRState(blocks=new_blocks, active=active)

    return step


def amr_state_from_global(global_data: jnp.ndarray, nbx: int, nby: int) -> jnp.ndarray:
    """Tile a (NF, nbx*BS, nby*BS) global INTERIOR field into the AMR root-level
    block storage (MAX_BLOCKS, NF, BS+2*NG, BS+2*NG).  Halo cells of each block
    are zero on output; caller should follow with `sync_ghosts_within_level_*`
    to fill them.
    """
    blocks = jnp.zeros((mb(0), NF, BS + 2 * NG, BS + 2 * NG), dtype=global_data.dtype)
    # global_data: (NF, nbx*BS, nby*BS). Reshape → (NF, nbx, BS, nby, BS) → (nbx, nby, NF, BS, BS).
    tiled = global_data.reshape(NF, nbx, BS, nby, BS).transpose(1, 3, 0, 2, 4)
    tiled = tiled.reshape(nbx * nby, NF, BS, BS)
    return blocks.at[:nbx * nby, :, NG:NG + BS, NG:NG + BS].set(tiled)


def amr_state_to_global(blocks: jnp.ndarray, nbx: int, nby: int) -> jnp.ndarray:
    """Inverse of `amr_state_from_global` — gather the interiors into a single
    (NF, nbx*BS, nby*BS) field."""
    interiors = blocks[:nbx * nby, :, NG:NG + BS, NG:NG + BS]   # (nbx*nby, NF, BS, BS)
    interiors = interiors.reshape(nbx, nby, NF, BS, BS)
    return interiors.transpose(2, 0, 3, 1, 4).reshape(NF, nbx * BS, nby * BS)
