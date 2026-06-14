"""Llama 7-patch atlas: a central Cartesian cube wrapped by 6 cubed-sphere shells.

Each ``Patch`` carries a uniform *reference* (logical) grid sampled over a box in
``(lx, ly, lz)`` plus ``ng`` ghost layers, and precomputed per-node geometry:

* ``X, Y, Z``  — world coordinates of every node (incl. ghosts), via the analytic
  map in :mod:`coord_maps`.
* ``jinv``     — the inverse Jacobian ``J^{-1}[a, A] = d(xi^a)/d(world_A)`` at every
  node, used by :mod:`derivative_curvilinear` to turn reference-frame finite
  differences into world-Cartesian derivatives.

Grid convention (vertex-centred, endpoints included)
----------------------------------------------------
A reference axis spanning ``[lo, hi]`` with ``N`` interior nodes has spacing
``dxi = (hi - lo) / (N - 1)`` and node positions ``lo + i*dxi`` for
``i = -ng .. N+ng-1`` (the interior is ``i = 0 .. N-1``; the first/last ``ng``
are ghosts). The field arrays in the solver mirror this: shape
``(NF, Nx+2ng, Ny+2ng, Nz+2ng)`` with the interior at ``[ng:N+ng]`` per axis.

Cube vs shells
--------------
* Cube (``PATCH_AFFINE``): all three axes span ``[0, 1]`` and map to world
  ``[-a, a]^3`` (``a = cube_half_width``). ``J^{-1} = I / (2a)``, so the
  curvilinear derivative reduces *exactly* to a uniform Cartesian FD at spacing
  ``2a/(N_cube-1)`` (a regression anchor).
* Shells (``PATCH_CUBED_SPHERE_SHELL``): the two angular axes span the *winged*
  box ``[-wing, 1+wing]`` (so ``mu, nu`` reach past ``±pi/4`` into the neighbour
  shells — the overlap that sources shell<->shell ghost fills); the radial axis
  spans ``[0, 1]`` mapping to ``[r_inner, r_outer]``. The wing is realized purely
  by the angular sampling range, so every patch uses ``origin=(0,0,0)``,
  ``scale=1``, ``wing_angular_fraction=0`` in the map calls.

Overlap requirement (checked by :func:`coverage_report`): the cube and shells
must physically overlap enough that every ghost node of one patch lands in the
*interior* of a donor patch. With ``cube_half_width >= r_inner`` the shells reach
inside the cube faces; ``r_outer`` must exceed the cube-corner reach
(``~cube_half_width*sqrt(3)`` plus ghost margin).
"""
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

from . import coord_maps as cm

_ZERO_ORIGIN = jnp.zeros((3,), dtype=cm.GEO_DTYPE)
_UNIT_SCALE = jnp.asarray(1.0, dtype=cm.GEO_DTYPE)

# Default angular wing (fraction of nominal patch-logical width). Matches the
# dendrojax / Pollney-Llama recommended value.
LLAMA_WING_FRACTION_DEFAULT = 0.167


def _node_axis(lo, hi, N, ng):
    """1D node positions over [lo, hi] with N interior nodes + ng ghosts/side."""
    dxi = (hi - lo) / (N - 1)
    idx = jnp.arange(-ng, N + ng, dtype=cm.GEO_DTYPE)
    return lo + idx * dxi, float(dxi)


@dataclass
class Patch:
    """One logical block + its precomputed world geometry."""
    name: str
    patch_type: int
    patch_params: jnp.ndarray          # (8,)
    N: tuple                            # (Nx, Ny, Nz) interior node counts
    ng: int
    lo: tuple                          # (lo_x, lo_y, lo_z) reference-axis starts
    hi: tuple                          # (hi_x, hi_y, hi_z) reference-axis ends
    # filled in __post_init__:
    lx: jnp.ndarray = field(default=None)   # (Nx+2ng,) reference nodes incl ghosts
    ly: jnp.ndarray = field(default=None)
    lz: jnp.ndarray = field(default=None)
    dxi: tuple = field(default=None)        # (dlx, dly, dlz) logical spacings
    X: jnp.ndarray = field(default=None)    # (shape) world coords
    Y: jnp.ndarray = field(default=None)
    Z: jnp.ndarray = field(default=None)
    jinv: jnp.ndarray = field(default=None)   # (3, 3, *shape): [a, A] = dxi^a/dworld_A
    d2coef: jnp.ndarray = field(default=None)  # (3,3,3,*shape): C[a,A,B]=dJinv[a,A]/dworld_B

    def __post_init__(self):
        ng = self.ng
        self.lx, dlx = _node_axis(self.lo[0], self.hi[0], self.N[0], ng)
        self.ly, dly = _node_axis(self.lo[1], self.hi[1], self.N[1], ng)
        self.lz, dlz = _node_axis(self.lo[2], self.hi[2], self.N[2], ng)
        self.dxi = (dlx, dly, dlz)
        self._build_geometry()

    @property
    def shape(self):
        ng = self.ng
        return (self.N[0] + 2 * ng, self.N[1] + 2 * ng, self.N[2] + 2 * ng)

    @property
    def interior(self):
        """Tuple of slices selecting the interior (drops ghosts) on the 3 axes."""
        ng = self.ng
        return (slice(ng, self.N[0] + ng),
                slice(ng, self.N[1] + ng),
                slice(ng, self.N[2] + ng))

    def _build_geometry(self):
        LX, LY, LZ = jnp.meshgrid(self.lx, self.ly, self.lz, indexing="ij")
        shp = LX.shape
        lxf, lyf, lzf = LX.ravel(), LY.ravel(), LZ.ravel()
        pt, pp = self.patch_type, self.patch_params

        map_v = jax.vmap(lambda a, b, c: cm.dispatch_map(
            pt, pp, a, b, c, _ZERO_ORIGIN, _UNIT_SCALE))
        gx, gy, gz = map_v(lxf, lyf, lzf)
        self.X = gx.reshape(shp)
        self.Y = gy.reshape(shp)
        self.Z = gz.reshape(shp)

        # inverse Jacobian Jinv[a,A] = dxi^a/dx_A and its world-gradient
        #   C[a,A,B] = dJinv[a,A]/dx_B = sum_b Jinv[b,B] * dJinv[a,A]/dxi^b
        # both analytic (autodiff of the closed-form map), precomputed per node.
        def jinv_at(l):
            J = cm.dispatch_jacobian(pt, pp, l[0], l[1], l[2], _ZERO_ORIGIN, _UNIT_SCALE)
            return jnp.linalg.inv(J)                    # (3,3): [a_log, A_world]

        L = jnp.stack([lxf, lyf, lzf], axis=1)         # (Npts, 3)
        Jinv = jax.vmap(jinv_at)(L)                     # (Npts, 3, 3)  [a,A]
        dJinv = jax.vmap(jax.jacfwd(jinv_at))(L)        # (Npts, 3, 3, 3) [a,A,b]
        C = jnp.einsum("nbB,naAb->naAB", Jinv, dJinv)   # (Npts, 3, 3, 3) [a,A,B]

        self.jinv = jnp.moveaxis(Jinv.reshape(shp + (3, 3)), (-2, -1), (0, 1))   # (3,3,*shp)
        self.d2coef = jnp.moveaxis(
            C.reshape(shp + (3, 3, 3)), (-3, -2, -1), (0, 1, 2))                 # (3,3,3,*shp)


@dataclass
class LlamaGrid:
    """The full 7-patch atlas plus the geometry parameters that built it."""
    patches: list
    cube_half_width: float
    r_inner: float
    r_outer: float
    radial_mode: int
    wing: float
    ng: int

    @property
    def cube(self):
        return self.patches[0]

    @property
    def shells(self):
        return self.patches[1:]


def build_llama_grid(
    cube_half_width: float,
    r_inner: float,
    r_outer: float,
    N_cube: int,
    N_ang: int,
    N_rad: int,
    order: int = 6,
    radial_mode: int = cm.RADIAL_MODE_LINEAR,
    wing: float = LLAMA_WING_FRACTION_DEFAULT,
) -> LlamaGrid:
    """Construct the 7-patch Llama atlas.

    Parameters
    ----------
    cube_half_width : float
        Central cube spans ``[-a, a]^3``. Must be ``>= r_inner``.
    r_inner, r_outer : float
        Shared shell radial bounds. ``r_outer`` must exceed the cube-corner
        reach so the cube's corner ghosts are covered by shells.
    N_cube, N_ang, N_rad : int
        Interior node counts: cube per axis; shell angular (per angular axis);
        shell radial.
    order : int
        FD order (sets ghost width ``ng = order // 2``).
    radial_mode : int
        ``coord_maps.RADIAL_MODE_LINEAR`` or ``RADIAL_MODE_LOG``.
    wing : float
        Angular overlap fraction (see module docstring).
    """
    if cube_half_width < r_inner:
        raise ValueError(
            f"cube_half_width ({cube_half_width}) must be >= r_inner ({r_inner}).")
    if r_outer <= r_inner:
        raise ValueError(f"r_outer ({r_outer}) must be > r_inner ({r_inner}).")
    if not (0.0 <= wing < 0.5):
        raise ValueError(f"wing ({wing}) must be in [0, 0.5).")
    ng = order // 2

    patches = []

    # patch 0: central cube
    cube = Patch(
        name="cube",
        patch_type=cm.PATCH_AFFINE,
        patch_params=cm.make_affine_params(
            world_origin=(-cube_half_width,) * 3, world_scale=2.0 * cube_half_width),
        N=(N_cube, N_cube, N_cube),
        ng=ng,
        lo=(0.0, 0.0, 0.0),
        hi=(1.0, 1.0, 1.0),
    )
    patches.append(cube)

    # patches 1..6: cubed-sphere shells, faces +x,-x,+y,-y,+z,-z
    face_names = ["+x", "-x", "+y", "-y", "+z", "-z"]
    for face_id, fname in enumerate(face_names):
        shell = Patch(
            name=f"shell{fname}",
            patch_type=cm.PATCH_CUBED_SPHERE_SHELL,
            patch_params=cm.make_cubed_sphere_params(
                face_id, r_inner, r_outer, radial_mode=radial_mode,
                wing_angular_fraction=0.0),  # wing realized via sampling range
            N=(N_ang, N_ang, N_rad),
            ng=ng,
            lo=(-wing, -wing, 0.0),
            hi=(1.0 + wing, 1.0 + wing, 1.0),
        )
        patches.append(shell)

    return LlamaGrid(
        patches=patches,
        cube_half_width=cube_half_width,
        r_inner=r_inner,
        r_outer=r_outer,
        radial_mode=radial_mode,
        wing=wing,
        ng=ng,
    )


def coverage_report(grid: LlamaGrid, verbose: bool = False) -> dict:
    """For every ghost node of every patch, find which donor patch (if any)
    contains it. Returns a summary dict; an empty ``holes`` list means every
    ghost is either covered by exactly one interior donor or is an outer
    boundary (no donor -> handled by the outer BC).

    A "hole" is a ghost node that lands inside *no* donor patch yet sits at a
    world radius ``<= r_outer`` (i.e. inside the Llama domain, where we expected
    a donor). The only legitimate no-donor ghosts are the outer-radial boundary
    layers at ``r > r_outer`` (handled by the outer BC).
    """
    patches = grid.patches
    r_outer = grid.r_outer
    # Precompute per-patch (type, params) for containment tests.
    summary = {"per_patch": {}, "holes": [], "n_ghost_total": 0,
               "n_covered": 0, "n_boundary": 0}

    for pi, p in enumerate(patches):
        ng = p.ng
        nx, ny, nz = p.shape
        # boolean mask of ghost nodes (any axis index in the ghost band)
        ix = jnp.arange(nx)
        iy = jnp.arange(ny)
        iz = jnp.arange(nz)
        gx_mask = (ix < ng) | (ix >= nx - ng)
        gy_mask = (iy < ng) | (iy >= ny - ng)
        gz_mask = (iz < ng) | (iz >= nz - ng)
        IGX, IGY, IGZ = jnp.meshgrid(gx_mask, gy_mask, gz_mask, indexing="ij")
        is_ghost = IGX | IGY | IGZ                      # (nx,ny,nz)

        # world coords of ghost nodes
        Xg = p.X[is_ghost]
        Yg = p.Y[is_ghost]
        Zg = p.Z[is_ghost]

        # count how many donor patches contain each ghost node
        n_donors = jnp.zeros(Xg.shape, dtype=jnp.int32)
        for di, d in enumerate(patches):
            if di == pi:
                continue
            contains = jax.vmap(lambda a, b, c: cm.dispatch_contains(
                d.patch_type, d.patch_params, a, b, c, _ZERO_ORIGIN, _UNIT_SCALE))
            inside = contains(Xg, Yg, Zg)
            n_donors = n_donors + inside.astype(jnp.int32)

        n_ghost = int(Xg.shape[0])
        covered = int(jnp.sum(n_donors >= 1))
        boundary = int(jnp.sum(n_donors == 0))
        # holes = uncovered ghosts inside the domain (radius <= r_outer)
        Rg = jnp.sqrt(Xg * Xg + Yg * Yg + Zg * Zg)
        hole_mask = (n_donors == 0) & (Rg <= r_outer - 1e-9)
        n_holes = int(jnp.sum(hole_mask))
        if n_holes:
            summary["holes"].append((p.name, n_holes))
        summary["per_patch"][p.name] = {
            "n_ghost": n_ghost, "covered": covered, "boundary": boundary,
            "holes": n_holes,
            "max_donors": int(jnp.max(n_donors)) if n_ghost else 0,
        }
        summary["n_ghost_total"] += n_ghost
        summary["n_covered"] += covered
        summary["n_boundary"] += boundary
        if verbose:
            print(f"{p.name:10s} ghosts={n_ghost:7d} covered={covered:7d} "
                  f"boundary={boundary:6d} max_donors="
                  f"{summary['per_patch'][p.name]['max_donors']}")

    return summary
