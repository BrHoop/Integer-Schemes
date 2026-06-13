"""
Fused FP64 RHS and full-step kernels for Maxwell-Chern-Simons 2D.

Exports
-------
make_fused_rhs(...)   — fused RHS (one HBM read per tile); BCs applied by caller
make_fused_step(...)  — temporally fused full RK4 step (one HBM pass for all 4
                        stages; handles periodic BC internally)

Tile geometry
-------------
RHS kernel   : tile (NF, BS+2*NG, BS+2*NG)          → rhs   (NF, BS, BS)
Step kernel  : tile (NF, BS+2*NG_T, BS+2*NG_T)       → s_new (NF, BS, BS)
               NG_T = 4*NG = 12  (full temporal halo for 4 RK4 stages)

BC notes
--------
make_fused_rhs pads with mode='edge'. For periodic BC the caller must:
  1. Pre-sync ghost zones before calling (vmap(bc_periodic)(data))
  2. Post-sync ghost zones of the returned RHS

make_fused_step works on the physical interior (strips ghost zones before
processing), uses mode='wrap' on the interior for correct periodic wrapping,
and restores ghost zones before returning. No external BC passes needed.
make_fused_step is only valid for periodic BC.
"""

import os
import jax
import jax.numpy as jnp
from jax import lax, vmap

jax.config.update("jax_enable_x64", True)

# Tile edge length.  Override at startup via:  MCS_FUSED_BS=16 python ...
# See fused_rhs_ozaki.py for trade-off discussion.
BS   = int(os.environ.get("MCS_FUSED_BS", 32))
NG   = 3     # 6th-order stencil halo
NG_T = 12    # temporal halo for full 4-stage RK4  (= 4 * NG)
NF   = 10    # number of MCS fields

EX, EY, EZ = 0, 1, 2
BX, BY, BZ = 3, 4, 5
XI, PI, PSI, PHI = 6, 7, 8, 9

_C1  = [-1.0,  9.0, -45.0,   0.0,  45.0,  -9.0,  1.0]
_C2  = [ 2.0, -27.0, 270.0, -490.0, 270.0, -27.0,  2.0]
# Kreiss-Oliger dissipation stencil: the 6th-order central difference δ⁶.
# Added with +ko coefficient it DAMPS grid-scale modes (Nyquist symbol −64).
# The negated form [-1,6,-15,20,-15,6,-1] is anti-dissipative and causes an
# exponential grid-scale instability — do not flip this sign.
_CKO = [ 1.0, -6.0,  15.0, -20.0,  15.0,  -6.0,  1.0]


# ---------------------------------------------------------------------------
# Generic sized stencil + PDE
# ---------------------------------------------------------------------------

def _pde_rhs_sized(tile, inv60dx, inv60dy, inv180dx2, inv180dy2,
                   ko_cx, ko_cy, cs, two_cs_L, two_L, K1, K2, ko_sigma):
    """
    MCS PDE RHS on a tile of any size.
    Input:  (NF, H, W)            — H, W >= 2*NG + 1
    Output: (NF, H-2*NG, W-2*NG)
    """
    h_out = tile.shape[1] - 2 * NG
    w_out = tile.shape[2] - 2 * NG

    def sx(f, coeffs):
        acc = jnp.zeros((h_out, w_out), jnp.float64)
        for k, c in enumerate(coeffs):
            if c != 0.0:
                acc = acc + c * tile[f, k:k + h_out, NG:NG + w_out]
        return acc

    def sy(f, coeffs):
        acc = jnp.zeros((h_out, w_out), jnp.float64)
        for k, c in enumerate(coeffs):
            if c != 0.0:
                acc = acc + c * tile[f, NG:NG + h_out, k:k + w_out]
        return acc

    Ex  = tile[EX,  NG:NG + h_out, NG:NG + w_out]
    Ey  = tile[EY,  NG:NG + h_out, NG:NG + w_out]
    Ez  = tile[EZ,  NG:NG + h_out, NG:NG + w_out]
    Bx  = tile[BX,  NG:NG + h_out, NG:NG + w_out]
    By  = tile[BY,  NG:NG + h_out, NG:NG + w_out]
    Bz  = tile[BZ,  NG:NG + h_out, NG:NG + w_out]
    Pi  = tile[PI,  NG:NG + h_out, NG:NG + w_out]
    Psi = tile[PSI, NG:NG + h_out, NG:NG + w_out]
    Phi = tile[PHI, NG:NG + h_out, NG:NG + w_out]

    dEx_dx  = sx(EX,  _C1) * inv60dx;  dEx_dy  = sy(EX,  _C1) * inv60dy
    dEy_dx  = sx(EY,  _C1) * inv60dx;  dEy_dy  = sy(EY,  _C1) * inv60dy
    dEz_dx  = sx(EZ,  _C1) * inv60dx;  dEz_dy  = sy(EZ,  _C1) * inv60dy
    dBx_dx  = sx(BX,  _C1) * inv60dx;  dBx_dy  = sy(BX,  _C1) * inv60dy
    dBy_dx  = sx(BY,  _C1) * inv60dx;  dBy_dy  = sy(BY,  _C1) * inv60dy
    dBz_dx  = sx(BZ,  _C1) * inv60dx;  dBz_dy  = sy(BZ,  _C1) * inv60dy
    dxi_dx  = sx(XI,  _C1) * inv60dx;  dxi_dy  = sy(XI,  _C1) * inv60dy
    dPsi_dx = sx(PSI, _C1) * inv60dx;  dPsi_dy = sy(PSI, _C1) * inv60dy
    dPhi_dx = sx(PHI, _C1) * inv60dx;  dPhi_dy = sy(PHI, _C1) * inv60dy

    d2xi_dx2 = sx(XI, _C2) * inv180dx2
    d2xi_dy2 = sy(XI, _C2) * inv180dy2

    dt_Ex  = dBz_dy  - dPsi_dx - two_cs_L * (Pi * Bx - Ez * dxi_dy)
    dt_Ey  = -dBz_dx - dPsi_dy - two_cs_L * (Pi * By + Ez * dxi_dx)
    dt_Ez  = dBy_dx  - dBx_dy  - two_cs_L * (Pi * Bz + Ex * dxi_dy - Ey * dxi_dx)
    dt_Bx  = -dEz_dy + dPhi_dx
    dt_By  =  dEz_dx + dPhi_dy
    dt_Bz  = -dEy_dx + dEx_dy
    dt_xi  = -Pi * cs
    dt_Pi  = (-d2xi_dx2 - d2xi_dy2 + two_L * (Bx * Ex + By * Ey + Bz * Ez)) * cs
    dt_Psi = -dEx_dx - dEy_dy - K1 * Psi - two_cs_L * (Bx * dxi_dx + By * dxi_dy)
    dt_Phi =  dBx_dx + dBy_dy - K2 * Phi

    dt_all = [dt_Ex, dt_Ey, dt_Ez, dt_Bx, dt_By, dt_Bz,
              dt_xi, dt_Pi, dt_Psi, dt_Phi]

    if ko_sigma > 0.0:
        dt_all = [
            dt_all[f] + sx(f, _CKO) * ko_cx + sy(f, _CKO) * ko_cy
            for f in range(NF)
        ]

    return jnp.stack(dt_all)  # (NF, h_out, w_out)


# ---------------------------------------------------------------------------
# Per-tile kernel factories
# ---------------------------------------------------------------------------

def _make_kernel_fn(dx, dy, cs, L, K1, K2, ko_sigma):
    """RHS kernel: (NF, BS+2*NG, BS+2*NG) → (NF, BS, BS)."""
    inv60dx   = 1.0 / (60.0 * dx)
    inv60dy   = 1.0 / (60.0 * dy)
    inv180dx2 = 1.0 / (180.0 * dx ** 2)
    inv180dy2 = 1.0 / (180.0 * dy ** 2)
    ko_cx     = ko_sigma / (64.0 * dx)
    ko_cy     = ko_sigma / (64.0 * dy)
    two_cs_L  = 2.0 * cs * L
    two_L     = 2.0 * L

    def kernel_fn(tile):
        return _pde_rhs_sized(tile, inv60dx, inv60dy, inv180dx2, inv180dy2,
                              ko_cx, ko_cy, cs, two_cs_L, two_L,
                              K1, K2, ko_sigma)
    return kernel_fn


def _make_step_kernel_fn(dx, dy, cs, L, K1, K2, ko_sigma, dt):
    """
    Temporally fused RK4 step kernel.
    Input:  tile  (NF, BS+2*NG_T, BS+2*NG_T)
    Output: s_new (NF, BS, BS)

    All four RK4 stages run inside the tile; no HBM access for intermediates.
    """
    inv60dx   = 1.0 / (60.0 * dx)
    inv60dy   = 1.0 / (60.0 * dy)
    inv180dx2 = 1.0 / (180.0 * dx ** 2)
    inv180dy2 = 1.0 / (180.0 * dy ** 2)
    ko_cx     = ko_sigma / (64.0 * dx)
    ko_cy     = ko_sigma / (64.0 * dy)
    two_cs_L  = 2.0 * cs * L
    two_L     = 2.0 * L

    def rhs(t):
        return _pde_rhs_sized(t, inv60dx, inv60dy, inv180dx2, inv180dy2,
                              ko_cx, ko_cy, cs, two_cs_L, two_L,
                              K1, K2, ko_sigma)

    def step_kernel(tile):
        # tile: (NF, BS+24, BS+24)

        # Stage 1 — k1 on (BS+18)×(BS+18)
        k1 = rhs(tile)
        # s1 = s + dt/2 * k1; outer NG cells of s1 are halo for stage 2
        s1 = tile[:, NG:-NG, NG:-NG] + 0.5 * dt * k1

        # Stage 2 — k2 on (BS+12)×(BS+12)
        k2 = rhs(s1)
        # s2 = s + dt/2 * k2 (uses s0 at the (BS+12) inner region)
        s2 = tile[:, 2*NG:-2*NG, 2*NG:-2*NG] + 0.5 * dt * k2

        # Stage 3 — k3 on (BS+6)×(BS+6)
        k3 = rhs(s2)
        # s3 = s + dt * k3
        s3 = tile[:, 3*NG:-3*NG, 3*NG:-3*NG] + dt * k3

        # Stage 4 — k4 on BS×BS
        k4 = rhs(s3)

        # Extract BS×BS centres of k1,k2,k3 for final combination
        k1_c = k1[:, 3*NG:3*NG + BS, 3*NG:3*NG + BS]
        k2_c = k2[:, 2*NG:2*NG + BS, 2*NG:2*NG + BS]
        k3_c = k3[:,   NG:  NG + BS,   NG:  NG + BS]
        s0_c = tile[:, 4*NG:4*NG + BS, 4*NG:4*NG + BS]

        return s0_c + (dt / 6.0) * (k1_c + 2.0*k2_c + 2.0*k3_c + k4)

    return step_kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_fused_rhs(Nx_tot, Ny_tot, dx, dy, cs, L, K1, K2, ko_sigma):
    """
    Build a JIT-compiled fused RHS function.

    Pads with mode='edge'. For periodic BC the caller must pre-sync ghost zones
    before calling and post-sync the returned RHS (vmap(bc_periodic)).

    Returns: fused_rhs(data: (NF,Nx_tot,Ny_tot)) → (NF,Nx_tot,Ny_tot)
    """
    pad_x = (BS - Nx_tot % BS) % BS
    pad_y = (BS - Ny_tot % BS) % BS
    Nx_al = Nx_tot + pad_x
    Ny_al = Ny_tot + pad_y
    nbx   = Nx_al // BS
    nby   = Ny_al // BS
    ph = pw = BS + 2 * NG

    kernel_fn = _make_kernel_fn(dx, dy, cs, L, K1, K2, ko_sigma)

    ii, jj = jnp.meshgrid(jnp.arange(nbx, dtype=jnp.int32),
                           jnp.arange(nby, dtype=jnp.int32), indexing='ij')
    patch_starts = jnp.stack([ii.ravel(), jj.ravel()], axis=1)

    @jax.jit
    def fused_rhs(data):
        aligned = jnp.pad(data,    ((0, 0), (0, pad_x), (0, pad_y)), mode='edge')
        padded  = jnp.pad(aligned, ((0, 0), (NG, NG),   (NG, NG)),   mode='edge')

        def extract_field(u_f):
            def extract_one(ij):
                return lax.dynamic_slice(u_f, (ij[0] * BS, ij[1] * BS), (ph, pw))
            return vmap(extract_one)(patch_starts)

        patches     = vmap(extract_field)(padded).transpose(1, 0, 2, 3)
        rhs_patches = vmap(kernel_fn)(patches)

        rhs_al = (rhs_patches
                  .reshape(nbx, nby, NF, BS, BS)
                  .transpose(2, 0, 3, 1, 4)
                  .reshape(NF, Nx_al, Ny_al))
        return rhs_al[:, :Nx_tot, :Ny_tot]

    return fused_rhs


def make_fused_step(Nx_tot, Ny_tot, ng, dx, dy, cs, L, K1, K2, ko_sigma, dt):
    """
    Build a JIT-compiled temporally fused full RK4 step (periodic BC only).

    Works on the physical interior (strips ghost zones before processing) and
    wraps the interior periodically via mode='wrap' — so no external BC passes
    are needed.  Ghost zones of the returned state are correctly filled.

    Returns: fused_step(data: (NF,Nx_tot,Ny_tot)) → (NF,Nx_tot,Ny_tot)
    """
    Nx = Nx_tot - 2 * ng
    Ny = Ny_tot - 2 * ng

    pad_x = (BS - Nx % BS) % BS
    pad_y = (BS - Ny % BS) % BS
    Nx_al = Nx + pad_x
    Ny_al = Ny + pad_y
    nbx   = Nx_al // BS
    nby   = Ny_al // BS
    ph = pw = BS + 2 * NG_T

    step_kernel = _make_step_kernel_fn(dx, dy, cs, L, K1, K2, ko_sigma, dt)

    ii, jj = jnp.meshgrid(jnp.arange(nbx, dtype=jnp.int32),
                           jnp.arange(nby, dtype=jnp.int32), indexing='ij')
    patch_starts = jnp.stack([ii.ravel(), jj.ravel()], axis=1)

    @jax.jit
    def fused_step(data):
        # Strip ghost zones — work on physical interior only.
        # This ensures mode='wrap' wraps around the actual periodic domain,
        # not around ghost zone copies that would give wrong stencil values.
        interior = data[:, ng:ng + Nx, ng:ng + Ny]   # (NF, Nx, Ny)

        # Align to BS multiple, then add NG_T periodic halo.
        aligned = jnp.pad(interior, ((0, 0), (0, pad_x), (0, pad_y)), mode='wrap')
        padded  = jnp.pad(aligned,  ((0, 0), (NG_T, NG_T), (NG_T, NG_T)), mode='wrap')

        def extract_field(u_f):
            def extract_one(ij):
                return lax.dynamic_slice(u_f, (ij[0] * BS, ij[1] * BS), (ph, pw))
            return vmap(extract_one)(patch_starts)

        patches     = vmap(extract_field)(padded).transpose(1, 0, 2, 3)
        new_patches = vmap(step_kernel)(patches)    # (n_patches, NF, BS, BS)

        interior_new = (new_patches
                        .reshape(nbx, nby, NF, BS, BS)
                        .transpose(2, 0, 3, 1, 4)
                        .reshape(NF, Nx_al, Ny_al))[:, :Nx, :Ny]

        # Reinsert updated interior into the full ghost-zone array and fill
        # ghost zones with periodic copies of the boundary interior cells.
        out = data.at[:, ng:ng + Nx, ng:ng + Ny].set(interior_new)
        out = out.at[:, 0:ng,  :].set(out[:, Nx_tot - 2*ng:Nx_tot - ng, :])
        out = out.at[:, -ng:,  :].set(out[:, ng:2*ng, :])
        out = out.at[:, :, 0:ng].set(out[:, :, Ny_tot - 2*ng:Ny_tot - ng])
        out = out.at[:, :, -ng:].set(out[:, :, ng:2*ng])
        return out

    return fused_step
