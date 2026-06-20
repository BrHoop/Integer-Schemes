"""
Single-dt RK4 evolution over the 3D node-centered AMR hierarchy (one patch).

Phase A scope: one patch (the cube), single shared dt across levels, no
sub-cycling.  Each RHS evaluation re-fills ghosts (within-level + cross-level)
then evaluates the per-block curvilinear RHS + KO, vmapped over slots and masked
by the active flag.  Geometry (per-slot ``jinv``/``d2coef``) is carried as data
(``AMRGeometry``) so the jitted step never recompiles across regrids — only the
geometry *values* change.

The per-block RHS reuses the single-level ``system.rhs_patch`` and
``CurvilinearDerivative`` unchanged (centering is invisible to them); on the
affine cube ``CurvilinearDerivative`` reduces to uniform FD.
"""
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from multipatch.derivative_curvilinear import CurvilinearDerivative
from .state import BS, NG, NF, LEVELS
from .geometry import level_spacing, level_geometry
from .sync import sync_within_level_root, sync_across_levels, sync_within_level

W = BS + 2 * NG


# Compact, recompute-don't-store geometry (no per-slot arrays).  For the affine
# cube jinv = I/world_scale and d2coef = 0 are constant across nodes/blocks/levels,
# so we carry two tiny constants + the per-level FD spacing.  (Curvilinear shells,
# Phase B, will recompute full per-node geometry inside the kernel instead.)
AMRGeometry = namedtuple("AMRGeometry", "jinv d2coef dxi")
# jinv:    (3, 3)        constant inverse Jacobian (broadcasts in CurvilinearDerivative)
# d2coef:  (3, 3, 3)     constant 2nd-derivative connection (zero for affine)
# dxi:     tuple of LEVELS logical-spacing tuples (one per level)

# Duck-typed geometry the CurvilinearDerivative reads (ng, jinv, d2coef, dxi).
_Geom = namedtuple("_Geom", "ng jinv d2coef dxi")


def build_geometry(parent_patch, topo=None, caps=None):
    """Compact per-run geometry (recompute-don't-store).

    Affine cube: a single constant ``jinv = I/world_scale`` + ``d2coef = 0`` for
    every block/level, plus the per-level FD spacing.  Replaces the former
    ``(caps[L],3,3,W,W,W)`` per-slot arrays (~3.6× the field storage) with two
    tiny constants.  ``topo``/``caps`` are accepted for call-site compatibility
    but no longer needed (nothing is stored per slot)."""
    jinv, d2coef = level_geometry(parent_patch)
    dxi = tuple(level_spacing(parent_patch, L) for L in range(LEVELS))
    return AMRGeometry(jinv=jinv, d2coef=d2coef, dxi=dxi)


def _block_rhs(F, jinv, d2coef, dxi, system, order, t, ko_sigma):
    """RHS of one block from its HALOED buffer (F: (NF,W,W,W)) via
    CurvilinearDerivative + KO, cropped to the (NF,BS,BS,BS) interior — the
    interior is what the interiors-only state stores and RK4 updates."""
    d = CurvilinearDerivative(_Geom(NG, jinv, d2coef, dxi), order=order)
    r = system.rhs_patch(F, d, t)
    if ko_sigma > 0.0:
        r = r + jnp.stack([d.ko(F[k], ko_sigma) for k in range(system.NF)])
    return r[:, NG:NG+BS, NG:NG+BS, NG:NG+BS]


def _embed_interiors(interiors_L):
    """Place BS³ interiors into the centre of zeroed (caps, NF, W, W, W) working
    buffers (halo filled by the sync passes)."""
    caps = interiors_L.shape[0]
    z = jnp.zeros((caps, NF, W, W, W), interiors_L.dtype)
    return z.at[:, :, NG:NG+BS, NG:NG+BS, NG:NG+BS].set(interiors_L)


class AMRCubeEvolution:
    """RK4 single-dt evolution of one patch's AMR hierarchy."""

    def __init__(self, parent_patch, system, nb_root, order=6, ko_sigma=0.0,
                 outer_bc=None):
        self.patch = parent_patch
        self.system = system
        self.nb_root = nb_root              # (nbx,nby,nbz) root tiling
        self.order = order
        self.ko_sigma = ko_sigma
        self.outer_bc = outer_bc if outer_bc is not None else (lambda b, t: b)

    # -- transient haloed working buffers from interiors-only storage --------- #
    def build_haloed(self, interiors, active, topo_arr, t):
        """Rebuild the per-level haloed working buffers (caps[L], NF, W,W,W) from
        the stored interiors (caps[L], NF, BS,BS,BS).  Coarse→fine so each level's
        cross-level fill reads its parent's already-haloed buffer.  Transient —
        consumed by the RHS, never stored."""
        nbx, nby, nbz = self.nb_root
        haloed = [None] * LEVELS
        # L0: stitch interiors → haloed (faces+edges+corners exact)
        haloed[0] = sync_within_level_root(interiors[0], nbx, nby, nbz)
        for L in range(1, LEVELS):
            h = _embed_interiors(interiors[L])          # interior in, halo zero
            h = sync_across_levels(                     # halo ← prolongated parent
                h, haloed[L - 1], topo_arr.parent_slot[L],
                topo_arr.child_c[L], active[L])
            h = sync_within_level(                      # face halos ← neighbour interiors
                h, topo_arr.neighbor_slot[L], topo_arr.neighbor_valid[L])
            haloed[L] = h
        haloed = self.outer_bc(haloed, t)               # base-level patch boundary
        return tuple(haloed)

    # -- RHS over the whole hierarchy (interiors in, interior RHS out) -------- #
    def rhs(self, interiors, active, geom, topo_arr, t):
        haloed = self.build_haloed(interiors, active, topo_arr, t)
        out = []
        for L in range(LEVELS):
            dxi = geom.dxi[L]
            # constant geometry (geom.jinv/d2coef) broadcasts inside each block's
            # CurvilinearDerivative — no per-slot geometry to vmap over.
            f = lambda F: _block_rhs(
                F, geom.jinv, geom.d2coef, dxi,
                self.system, self.order, t, self.ko_sigma)
            r = jax.vmap(f)(haloed[L])                   # (caps, NF, BS, BS, BS)
            r = jnp.where(active[L][:, None, None, None, None], r, 0.0)
            out.append(r)
        return tuple(out)

    # -- one RK4 step (single dt) over interiors ----------------------------- #
    def _axpy(self, a, x, y):
        return tuple(yy + a * xx for yy, xx in zip(y, x))

    def step(self, interiors, active, geom, topo_arr, t, dt):
        k1 = self.rhs(interiors, active, geom, topo_arr, t)
        k2 = self.rhs(self._axpy(0.5*dt, k1, interiors), active, geom, topo_arr, t+0.5*dt)
        k3 = self.rhs(self._axpy(0.5*dt, k2, interiors), active, geom, topo_arr, t+0.5*dt)
        k4 = self.rhs(self._axpy(dt, k3, interiors), active, geom, topo_arr, t+dt)
        return tuple(
            b + (dt/6.0)*(a + 2*bb + 2*c + e)
            for b, a, bb, c, e in zip(interiors, k1, k2, k3, k4))

    def make_jit_step(self):
        # active / topo_arr / geom are traced data (fixed per-level shapes) so the
        # compiled step is reused across regrids without retracing.
        return jax.jit(self.step)

    def evolve(self, state, geom, topo_arr, dt, nsteps, t0=0.0):
        step = self.make_jit_step()
        interiors = state.blocks
        t = t0
        for i in range(nsteps):
            interiors = step(interiors, state.active, geom, topo_arr, t, dt)
            t = t0 + (i + 1) * dt
        from .state import AMRState
        return AMRState(blocks=interiors, active=state.active), t
