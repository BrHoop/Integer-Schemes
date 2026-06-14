"""Analytic coordinate maps for Llama-style multipatch grids.

Each patch carries a reference (logical) cube ``[0, 1]^3`` with coords
``(lx, ly, lz)``. A ``patch_type`` tag selects the logical->world map:

* ``PATCH_AFFINE`` — Cartesian box ``world = world_origin + world_scale * l``.
  Used for the central cube of the Llama grid.

* ``PATCH_CUBED_SPHERE_SHELL`` — Thornburg 2004 equiangular cubed-sphere
  thinshell. Six inflated-cube faces each cover a 1/6 angular sector of a
  spherical shell ``[r_inner, r_outer]``. Two logical coords are angular
  (``tan`` of an equal-angular sweep), one is radial.

For the canonical +x face::

    mu = (pi/4)(2*lx - 1)        nu = (pi/4)(2*ly - 1)
    rho = tan(mu)                sigma = tan(nu)
    r   = radial(lz)                                   # linear or log
    D   = sqrt(1 + rho^2 + sigma^2)
    (cx, cy, cz) = (r/D, rho*r/D, sigma*r/D)

The other 5 faces apply a fixed 3x3 orthogonal matrix ``R`` to the canonical
``(cx, cy, cz)``. ``patch_params`` for a shell packs
``[face_id, radial_mode, r_inner, r_outer, wing_angular_fraction, 0, 0, 0]``
where ``face_id`` is ``0..5`` for ``+x, -x, +y, -y, +z, -z``.

This module is a trimmed, dependency-free port of
``~/Code/dendrojax/src/dendrojax/coord_maps.py`` (the closed-form math is
reused verbatim; only the octree-atlas plumbing and the ``dendrojax`` import
are dropped). ``origin``/``scale`` are retained on every signature so the maps
stay compatible with a future sub-block / refinement layer; root patches pass
``origin=(0,0,0)``, ``scale=1.0`` and these become no-ops.

Cross-validated against dendrojax (which itself matches the C++ ``cubesphere``
reference to ~4e-16) by the geometry tests.
"""
import jax
import jax.numpy as jnp

GEO_DTYPE = jnp.float64

# Patch type tags (dispatch order matches the tuples in dispatch_*).
PATCH_AFFINE = 0
PATCH_CUBED_SPHERE_SHELL = 1

PATCH_PARAMS_LEN = 8

# Radial-mode tags (cubed-sphere shells), stored at patch_params[1].
#   LINEAR : r(lz) = r_inner + lz * (r_outer - r_inner)
#   LOG    : r(lz) = r_inner * (r_outer/r_inner) ** lz   (wave-zone extraction)
RADIAL_MODE_LINEAR = 0
RADIAL_MODE_LOG = 1

# face_id layout: 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z.
# R maps canonical (cx, cy, cz) -> world (x, y, z); canonical x is radial.
FACE_ROTATIONS = jnp.array(
    [
        [[+1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # +x: identity
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # -x: flip x
        [[0.0, 1.0, 0.0], [+1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # +y
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # -y
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [+1.0, 0.0, 0.0]],  # +z
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],  # -z
    ],
    dtype=GEO_DTYPE,
)


def make_affine_params(world_origin=(0.0, 0.0, 0.0), world_scale=1.0):
    """Pack patch_params for a PATCH_AFFINE box: ``[origin_xyz, scale, ...]``."""
    p = jnp.zeros((PATCH_PARAMS_LEN,), dtype=GEO_DTYPE)
    p = p.at[0].set(jnp.asarray(world_origin[0], dtype=GEO_DTYPE))
    p = p.at[1].set(jnp.asarray(world_origin[1], dtype=GEO_DTYPE))
    p = p.at[2].set(jnp.asarray(world_origin[2], dtype=GEO_DTYPE))
    p = p.at[3].set(jnp.asarray(world_scale, dtype=GEO_DTYPE))
    return p


def make_cubed_sphere_params(
    face_id, r_inner, r_outer, radial_mode=RADIAL_MODE_LINEAR,
    wing_angular_fraction=0.0,
):
    """Pack the 8-float patch_params for a cubed-sphere shell.

    ``wing_angular_fraction`` inflates the angular logical box to
    ``[-wing, 1+wing]`` so neighbour-shell ghost fills sample a centred source
    margin (Pollney-Llama II.B). The radial axis is never winged; the forward /
    inverse / jacobian undo any radial inflation via this stored value.
    """
    p = jnp.zeros((PATCH_PARAMS_LEN,), dtype=GEO_DTYPE)
    p = p.at[0].set(jnp.asarray(face_id, dtype=GEO_DTYPE))
    p = p.at[1].set(jnp.asarray(radial_mode, dtype=GEO_DTYPE))
    p = p.at[2].set(jnp.asarray(r_inner, dtype=GEO_DTYPE))
    p = p.at[3].set(jnp.asarray(r_outer, dtype=GEO_DTYPE))
    p = p.at[4].set(jnp.asarray(wing_angular_fraction, dtype=GEO_DTYPE))
    return p


# --------------------------------------------------------------------------- #
# Radial-direction modes (cubed-sphere only)
# --------------------------------------------------------------------------- #


def _radial_forward(lz, r_inner, r_outer, radial_mode):
    r_lin = r_inner + lz * (r_outer - r_inner)
    log_ratio = jnp.log(r_outer / jnp.maximum(r_inner, 1e-30))
    r_log = r_inner * jnp.exp(lz * log_ratio)
    return jnp.where(radial_mode == RADIAL_MODE_LOG, r_log, r_lin)


def _radial_inverse(r, r_inner, r_outer, radial_mode):
    lz_lin = (r - r_inner) / (r_outer - r_inner)
    log_ratio = jnp.log(r_outer / jnp.maximum(r_inner, 1e-30))
    safe_log_ratio = jnp.where(jnp.abs(log_ratio) < 1e-30, 1e-30, log_ratio)
    lz_log = jnp.log(jnp.maximum(r, 1e-30) / jnp.maximum(r_inner, 1e-30)) / safe_log_ratio
    return jnp.where(radial_mode == RADIAL_MODE_LOG, lz_log, lz_lin)


def _radial_dr_dlz(lz, r_inner, r_outer, radial_mode):
    dr_lin = r_outer - r_inner
    log_ratio = jnp.log(r_outer / jnp.maximum(r_inner, 1e-30))
    dr_log = r_inner * jnp.exp(lz * log_ratio) * log_ratio
    return jnp.where(radial_mode == RADIAL_MODE_LOG, dr_log, dr_lin)


# --------------------------------------------------------------------------- #
# Affine (Cartesian) patch
# --------------------------------------------------------------------------- #


def _affine_map(patch_params, lx, ly, lz, origin, scale):
    plx = origin[0] + scale * lx
    ply = origin[1] + scale * ly
    plz = origin[2] + scale * lz
    gx = patch_params[0] + patch_params[3] * plx
    gy = patch_params[1] + patch_params[3] * ply
    gz = patch_params[2] + patch_params[3] * plz
    return gx, gy, gz


def _affine_inverse(patch_params, gx, gy, gz, origin, scale):
    inv_world = 1.0 / patch_params[3]
    plx = (gx - patch_params[0]) * inv_world
    ply = (gy - patch_params[1]) * inv_world
    plz = (gz - patch_params[2]) * inv_world
    inv = 1.0 / scale
    return (plx - origin[0]) * inv, (ply - origin[1]) * inv, (plz - origin[2]) * inv


def _affine_jacobian(patch_params, lx, ly, lz, origin, scale):
    del lx, ly, lz, origin
    eye = jnp.eye(3, dtype=GEO_DTYPE)
    return (patch_params[3] * scale) * eye


def _affine_contains(patch_params, gx, gy, gz, origin, scale):
    eps = 1e-6
    inv_world = 1.0 / patch_params[3]
    plx = (gx - patch_params[0]) * inv_world
    ply = (gy - patch_params[1]) * inv_world
    plz = (gz - patch_params[2]) * inv_world
    lx = (plx - origin[0]) / scale
    ly = (ply - origin[1]) / scale
    lz = (plz - origin[2]) / scale
    return (
        (lx >= -eps) & (lx <= 1.0 + eps)
        & (ly >= -eps) & (ly <= 1.0 + eps)
        & (lz >= -eps) & (lz <= 1.0 + eps)
    )


# --------------------------------------------------------------------------- #
# Cubed-sphere thinshell (Thornburg 2004)
# --------------------------------------------------------------------------- #


def _cubed_sphere_canonical(lx, ly, lz, r_inner, r_outer, radial_mode):
    """Canonical +x-face position; returns intermediates for the Jacobian."""
    mu = (jnp.pi / 4.0) * (2.0 * lx - 1.0)
    nu = (jnp.pi / 4.0) * (2.0 * ly - 1.0)
    rho = jnp.tan(mu)
    sigma = jnp.tan(nu)
    r = _radial_forward(lz, r_inner, r_outer, radial_mode)
    D = jnp.sqrt(1.0 + rho * rho + sigma * sigma)
    cx = r / D
    cy = rho * r / D
    cz = sigma * r / D
    return cx, cy, cz, r, D, rho, sigma


def _cubed_sphere_map(patch_params, lx, ly, lz, origin, scale):
    plx = origin[0] + scale * lx
    ply = origin[1] + scale * ly
    ang_scale = 1.0 + 2.0 * patch_params[4]
    plz = (origin[2] + scale * lz) / ang_scale
    face_id = patch_params[0].astype(jnp.int32)
    radial_mode = patch_params[1].astype(jnp.int32)
    cx, cy, cz, _r, _D, _rho, _sig = _cubed_sphere_canonical(
        plx, ply, plz, patch_params[2], patch_params[3], radial_mode
    )
    R = FACE_ROTATIONS[face_id]
    world = R @ jnp.stack([cx, cy, cz])
    return world[0], world[1], world[2]


def _cubed_sphere_inverse(patch_params, gx, gy, gz, origin, scale):
    face_id = patch_params[0].astype(jnp.int32)
    radial_mode = patch_params[1].astype(jnp.int32)
    R = FACE_ROTATIONS[face_id]
    canonical = R.T @ jnp.stack([gx, gy, gz])
    cx, cy, cz = canonical[0], canonical[1], canonical[2]
    r = jnp.sqrt(cx * cx + cy * cy + cz * cz)
    rho = cy / cx
    sigma = cz / cx
    plx = 0.5 + 2.0 * jnp.arctan(rho) / jnp.pi
    ply = 0.5 + 2.0 * jnp.arctan(sigma) / jnp.pi
    plz = _radial_inverse(r, patch_params[2], patch_params[3], radial_mode)
    ang_scale = 1.0 + 2.0 * patch_params[4]
    inv = 1.0 / scale
    lx = (plx - origin[0]) * inv
    ly = (ply - origin[1]) * inv
    lz = (plz * ang_scale - origin[2]) * inv
    return lx, ly, lz


def _cubed_sphere_jacobian(patch_params, lx, ly, lz, origin, scale):
    """Analytic d(world)/d(local) for thornburg04."""
    plx = origin[0] + scale * lx
    ply = origin[1] + scale * ly
    ang_scale = 1.0 + 2.0 * patch_params[4]
    plz = (origin[2] + scale * lz) / ang_scale
    face_id = patch_params[0].astype(jnp.int32)
    radial_mode = patch_params[1].astype(jnp.int32)
    _cx, _cy, _cz, r, D, rho, sigma = _cubed_sphere_canonical(
        plx, ply, plz, patch_params[2], patch_params[3], radial_mode
    )
    dr = _radial_dr_dlz(plz, patch_params[2], patch_params[3], radial_mode)
    k = jnp.pi / 2.0
    one_p_rho2 = 1.0 + rho * rho
    one_p_sig2 = 1.0 + sigma * sigma
    D3 = D * D * D
    inv_ang = 1.0 / ang_scale

    dcx_dlx = -k * r * rho * one_p_rho2 / D3
    dcx_dly = -k * r * sigma * one_p_sig2 / D3
    dcx_dlz = dr / D * inv_ang

    dcy_dlx = k * r * one_p_rho2 * one_p_sig2 / D3
    dcy_dly = -k * r * rho * sigma * one_p_sig2 / D3
    dcy_dlz = rho * dr / D * inv_ang

    dcz_dlx = -k * r * rho * sigma * one_p_rho2 / D3
    dcz_dly = k * r * one_p_rho2 * one_p_sig2 / D3
    dcz_dlz = sigma * dr / D * inv_ang

    J_canonical = jnp.array(
        [
            [dcx_dlx, dcx_dly, dcx_dlz],
            [dcy_dlx, dcy_dly, dcy_dlz],
            [dcz_dlx, dcz_dly, dcz_dlz],
        ],
        dtype=GEO_DTYPE,
    )
    R = FACE_ROTATIONS[face_id]
    return scale * (R @ J_canonical)


def _cubed_sphere_contains(patch_params, gx, gy, gz, origin, scale):
    face_id = patch_params[0].astype(jnp.int32)
    R = FACE_ROTATIONS[face_id]
    canonical = R.T @ jnp.stack([gx, gy, gz])
    cx, cy, cz = canonical[0], canonical[1], canonical[2]
    r = jnp.sqrt(cx * cx + cy * cy + cz * cz)
    eps = 1e-6
    safe_cx = jnp.where(jnp.abs(cx) < 1e-30, jnp.sign(cx + 1e-30) * 1e-30, cx)
    rho = cy / safe_cx
    sigma = cz / safe_cx
    plx = 0.5 + 2.0 * jnp.arctan(rho) / jnp.pi
    ply = 0.5 + 2.0 * jnp.arctan(sigma) / jnp.pi
    return (
        (cx > eps)
        & (r >= patch_params[2] - eps)
        & (r <= patch_params[3] + eps)
        & (plx >= origin[0] - eps)
        & (plx <= origin[0] + scale + eps)
        & (ply >= origin[1] - eps)
        & (ply <= origin[1] + scale + eps)
    )


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #


def dispatch_map(patch_type, patch_params, lx, ly, lz, origin, scale):
    """Forward logical->world for a single block (or vmapped fleet)."""
    return jax.lax.switch(
        patch_type, (_affine_map, _cubed_sphere_map),
        patch_params, lx, ly, lz, origin, scale,
    )


def dispatch_inverse_map(patch_type, patch_params, gx, gy, gz, origin, scale):
    """Inverse world->logical for a single block, given its face."""
    return jax.lax.switch(
        patch_type, (_affine_inverse, _cubed_sphere_inverse),
        patch_params, gx, gy, gz, origin, scale,
    )


def dispatch_jacobian(patch_type, patch_params, lx, ly, lz, origin, scale):
    """Jacobian d(world)/d(logical) at a logical point. Shape (3, 3)."""
    return jax.lax.switch(
        patch_type, (_affine_jacobian, _cubed_sphere_jacobian),
        patch_params, lx, ly, lz, origin, scale,
    )


def dispatch_contains(patch_type, patch_params, gx, gy, gz, origin, scale):
    """True iff world point (gx, gy, gz) lies inside this patch's region."""
    return jax.lax.switch(
        patch_type, (_affine_contains, _cubed_sphere_contains),
        patch_params, gx, gy, gz, origin, scale,
    )
