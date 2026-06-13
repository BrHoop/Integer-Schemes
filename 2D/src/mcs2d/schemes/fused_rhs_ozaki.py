"""
Fused INT8 Ozaki kernels for MCS 2D.

Exports
-------
make_fused_ozaki_rhs(...)   — fused RHS (one HBM pass); BCs applied by caller
make_fused_ozaki_step(...)  — temporally fused full RK4 step (all 4 stages in
                              one HBM pass); periodic BC only

All derivatives use the INT8 RNS pipeline of ozaki.py (BFP scaling → modular
residues → bias-corrected INT8 GEMM → Garner CRT).  The arithmetic is identical
in float64 to floating_point.py; only the execution path differs.

Temporal fusion
---------------
The full-step kernel loads a tile with halo NG_T = 4*NG = 12 and computes all
four RK4 stages inside the tile.  Each stage's RHS shrinks the valid region by
2*NG, so four banded D-matrices are precomputed — one per stage output size
(BS+18, BS+12, BS+6, BS).  No intermediate stage touches HBM.

Cost note: the temporal tile (BS+2*NG_T)^2 has ~3x the area of the single-pass
RHS tile, so each output cell is recomputed more often.  For the compute-bound
Ozaki pipeline this redundancy may outweigh the HBM savings — benchmark before
assuming it is faster than the 4-call RHS path.
"""

import os
import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np

jax.config.update("jax_enable_x64", True)

# Tile edge length.  Override at startup via:  MCS_FUSED_BS=16 python ...
# Smaller BS → more tiles (better SM occupancy on Hopper), smaller per-tile
# intermediates (more likely to stay in shared memory rather than VRAM), but
# higher halo overhead per output cell.  Recommended values: 16, 32, 64.
BS   = int(os.environ.get("MCS_FUSED_BS", 32))
NG   = 3     # 6th-order halo
NG_T = 12    # temporal halo for full 4-stage RK4  (= 4 * NG)
NF   = 10    # number of MCS fields

# Pad the contraction (K) dimension of every Ozaki GEMM up to this multiple so
# the call hits INT8 tensor cores on Ampere / Hopper.  Padded D residues are 0
# (-128 in bias-shifted i8); the bias formula's `- K*16384` term uses K=padded
# length, exactly cancelling the padded -128*-128=16384 contributions.
_K_MULTIPLE = 16

# INT8 GEMM via cuBLASLt fires Ampere/Hopper tensor cores but can break CUDA
# graph capture on some JAX/CUDA configurations (cuBLASLt does runtime kernel
# autotuning).  Default = False (int32 GEMM, no tensor cores but always
# graph-safe).  Flip to True on hardware/JAX combos that handle INT8 cuBLASLt
# under graph capture — then re-verify with the test suite and benchmark.
USE_INT8_GEMM = False


def _pad_K(k_orig):
    return ((k_orig + _K_MULTIPLE - 1) // _K_MULTIPLE) * _K_MULTIPLE

EX, EY, EZ = 0, 1, 2
BX, BY, BZ = 3, 4, 5
XI, PI, PSI, PHI = 6, 7, 8, 9

# Integer stencil numerators (same as ozaki.py)
_C1_INT  = np.array([-1,  9, -45,   0,  45,  -9,  1], dtype=np.int64)
_C2_INT  = np.array([ 2, -27, 270, -490, 270, -27,  2], dtype=np.int64)
# Kreiss-Oliger 6th-order central difference δ⁶ (damping with +ko coefficient).
# Must NOT be the negated form [-1,6,-15,20,-15,6,-1] — that is anti-dissipative
# and drives an exponential grid-scale instability.  See floating_point.CKO.
_CKO_INT = np.array([ 1, -6,  15, -20,  15,  -6,  1], dtype=np.int64)

# Fields that need d1 derivatives (PI only ever appears as a multiplier).
_D1_IDX = jnp.array([EX, EY, EZ, BX, BY, BZ, XI, PSI, PHI])

from mcs2d.schemes.ozaki import CrtFloatConverter


# ── Precomputed banded D matrices ──────────────────────────────────────────────

def _build_D(coeffs_int, crt, n_out=BS):
    """
    Build the banded stencil matrix mapping (n_out + 2*NG) inputs → n_out outputs.

    K is padded to a multiple of _K_MULTIPLE for INT8 tensor-core alignment.
    Padded columns hold residue 0 (-128 in int8 form); they don't affect D_rsum
    (the sum of *original* residues), and the GEMM bias formula uses
    K = D.shape[2] = K_padded, which cancels their -128*-128 contribution.

    Returns:
      D_mods_i8 : (k_full, n_out, K_padded)  int8
      D_rsum    : (k_full, n_out)             int32
    """
    K_orig = n_out + 2 * NG
    K_pad  = _pad_K(K_orig)
    D = np.zeros((n_out, K_pad), dtype=np.int64)
    for r in range(n_out):
        c = r + NG
        D[r, c - 3 : c + 4] = coeffs_int

    D_mods_i8_list, D_rsum_list = [], []
    for m in crt.mods_full_list:
        D_m = (D % m).astype(np.int32)                  # padded cols are 0
        D_rsum_list.append(D_m.sum(axis=1, dtype=np.int32))
        D_mods_i8_list.append((D_m - 128).astype(np.int8))

    return (
        jnp.stack([jnp.array(d) for d in D_mods_i8_list]),
        jnp.stack([jnp.array(r) for r in D_rsum_list]),
    )


# ── Batched INT8 Ozaki contract (F fields, single tile, any size) ─────────────

def _ozaki_batched(fields, D_mods_i8, D_rsum, crt, axis):
    """
    Ozaki INT8 derivative for F fields over one tile of any size.

    Args:
        fields    : (F, H, W)              float64
        D_mods_i8 : (k_full, H-2*NG, H)    int8   (contracts the derivative axis)
        D_rsum    : (k_full, H-2*NG)        int32
        axis      : 0 (x-derivative) or 1 (y-derivative)

    Returns:
        (F, H-2*NG, W-2*NG) float64 — scaled by BFP exponent only.

    The output shrinks by 2*NG on each axis: the derivative axis is contracted
    by the GEMM (input H → output H-2*NG), the other axis is cropped by NG each
    side to match.  D's input dimension sets the contraction length K.
    """
    K = D_mods_i8.shape[2]  # contraction length = input size along derivative axis

    def scale_one(field):
        max_val = jnp.max(jnp.abs(field))
        m_half  = crt.M_prod_base / 2.0
        s = jnp.where(max_val == 0, 0.0, jnp.floor(jnp.log2(m_half / max_val)))
        return field * (2.0 ** s), -s.astype(jnp.int32)

    scaled, exponents = vmap(scale_one)(fields)
    scaled_ints = jnp.floor(scaled + 0.5).astype(jnp.int64)

    # Base residues with moduli on axis 1 (batched over F on axis 0).
    regs_base = crt.to_base_residues(scaled_ints, axis=1)  # (F, k_base, H, W)
    regs_full = vmap(crt.base_extension)(regs_base)         # (F, k_full, H, W)

    # Bias-shift to signed int8.  If INT8 GEMM is enabled, keep as int8 so
    # XLA emits a real INT8 tensor-core matmul; otherwise cast to int32
    # (fallback for GPUs/configs without INT8 hardware support).
    regs_signed = (regs_full.astype(jnp.int32) - 128)
    if USE_INT8_GEMM:
        regs_signed = regs_signed.astype(jnp.int8)
        D_lhs = D_mods_i8
        accum = jnp.int32
    else:
        D_lhs = D_mods_i8.astype(jnp.int32)
        accum = None

    H_orig  = regs_signed.shape[2]                  # actual data length along x
    pad_amt = K - H_orig                              # cells to append (-128 each)

    def pad_contract(arr, axis):
        if pad_amt == 0:
            return arr
        pw = [(0, 0)] * arr.ndim
        pw[axis] = (0, pad_amt)
        return jnp.pad(arr, pw, constant_values=-128)

    if axis == 0:
        gemm_raw = lax.dot_general(
            D_lhs, pad_contract(regs_signed, axis=2),
            dimension_numbers=(([2], [2]), ([0], [1])),
            preferred_element_type=accum,
        )
        col_sum_r = regs_full.astype(jnp.int32).sum(axis=2).transpose(1, 0, 2)
        result_i32 = (gemm_raw
                      + 128 * D_rsum[:, :, None, None]
                      + 128 * col_sum_r[:, None, :, :]
                      - K * 16384)
        res_mods = crt.reduce_full(result_i32)
        res_mods = res_mods[:, :, :, NG:-NG].transpose(2, 0, 1, 3)

    else:  # axis == 1
        gemm_raw = lax.dot_general(
            pad_contract(regs_signed, axis=3), D_lhs,
            dimension_numbers=(([3], [2]), ([1], [0])),
            preferred_element_type=accum,
        )
        row_sum_r = regs_full.astype(jnp.int32).sum(axis=3).transpose(1, 0, 2)
        result_i32 = (gemm_raw
                      + 128 * D_rsum[:, None, None, :]
                      + 128 * row_sum_r[:, :, :, None]
                      - K * 16384)
        res_mods = crt.reduce_full(result_i32)
        res_mods = res_mods[:, :, NG:-NG, :].transpose(1, 0, 2, 3)

    out_int = vmap(crt.garner_reconstruction)(res_mods)
    return out_int * (2.0 ** exponents)[:, None, None]


# ── Sized PDE assembly (works for any tile size) ──────────────────────────────

def _make_pde_fn(dx, dy, cs, L, K1, K2, ko_sigma, crt):
    """Return pde(tile, n_out, Dset) computing the MCS RHS for any tile size."""
    inv_60dx   = 1.0 / (60.0 * dx)
    inv_60dy   = 1.0 / (60.0 * dy)
    inv_180dx2 = 1.0 / (180.0 * dx ** 2)
    inv_180dy2 = 1.0 / (180.0 * dy ** 2)
    ko_cx      = ko_sigma / (64.0 * dx)
    ko_cy      = ko_sigma / (64.0 * dy)
    two_cs_L   = 2.0 * cs * L
    two_L      = 2.0 * L

    def pde(tile, n_out, Dset):
        """
        Input:  tile (NF, n_out+2*NG, n_out+2*NG) float64
        Output: rhs  (NF, n_out, n_out)           float64
        Dset = (D_d1, D_d1_rsum, D_d2, D_d2_rsum, D_ko, D_ko_rsum) for this size.
        """
        D_d1, D_d1_rsum, D_d2, D_d2_rsum, D_ko, D_ko_rsum = Dset

        d1_in  = tile[_D1_IDX]
        dx_all = _ozaki_batched(d1_in, D_d1, D_d1_rsum, crt, axis=0) * inv_60dx
        dy_all = _ozaki_batched(d1_in, D_d1, D_d1_rsum, crt, axis=1) * inv_60dy

        dEx_dx, dEy_dx, dEz_dx, dBx_dx, dBy_dx, dBz_dx, dxi_dx, dPsi_dx, dPhi_dx = dx_all
        dEx_dy, dEy_dy, dEz_dy, dBx_dy, dBy_dy, dBz_dy, dxi_dy, dPsi_dy, dPhi_dy = dy_all

        xi_tile  = tile[XI:XI+1]
        d2xi_dx2 = _ozaki_batched(xi_tile, D_d2, D_d2_rsum, crt, axis=0)[0] * inv_180dx2
        d2xi_dy2 = _ozaki_batched(xi_tile, D_d2, D_d2_rsum, crt, axis=1)[0] * inv_180dy2

        Ex  = tile[EX,  NG:NG+n_out, NG:NG+n_out]
        Ey  = tile[EY,  NG:NG+n_out, NG:NG+n_out]
        Ez  = tile[EZ,  NG:NG+n_out, NG:NG+n_out]
        Bx  = tile[BX,  NG:NG+n_out, NG:NG+n_out]
        By  = tile[BY,  NG:NG+n_out, NG:NG+n_out]
        Bz  = tile[BZ,  NG:NG+n_out, NG:NG+n_out]
        Pi  = tile[PI,  NG:NG+n_out, NG:NG+n_out]
        Psi = tile[PSI, NG:NG+n_out, NG:NG+n_out]
        Phi = tile[PHI, NG:NG+n_out, NG:NG+n_out]

        dt_Ex  = dBz_dy  - dPsi_dx - two_cs_L * (Pi * Bx - Ez * dxi_dy)
        dt_Ey  = -dBz_dx - dPsi_dy - two_cs_L * (Pi * By + Ez * dxi_dx)
        dt_Ez  = dBy_dx  - dBx_dy  - two_cs_L * (Pi * Bz + Ex * dxi_dy - Ey * dxi_dx)
        dt_Bx  = -dEz_dy + dPhi_dx
        dt_By  =  dEz_dx + dPhi_dy
        dt_Bz  = -dEy_dx + dEx_dy
        dt_xi  = -Pi * cs
        dt_Pi  = (-d2xi_dx2 - d2xi_dy2 + two_L * (Bx*Ex + By*Ey + Bz*Ez)) * cs
        dt_Psi = -dEx_dx - dEy_dy - K1 * Psi - two_cs_L * (Bx*dxi_dx + By*dxi_dy)
        dt_Phi =  dBx_dx + dBy_dy - K2 * Phi

        rhs = jnp.stack([dt_Ex, dt_Ey, dt_Ez, dt_Bx, dt_By, dt_Bz,
                          dt_xi, dt_Pi, dt_Psi, dt_Phi])

        if ko_sigma > 0.0:
            ko_x = _ozaki_batched(tile, D_ko, D_ko_rsum, crt, axis=0) * ko_cx
            ko_y = _ozaki_batched(tile, D_ko, D_ko_rsum, crt, axis=1) * ko_cy
            rhs  = rhs + ko_x + ko_y

        return rhs

    return pde


def _build_Dset(crt, ko_sigma, n_out):
    """Return (D_d1, D_d1_rsum, D_d2, D_d2_rsum, D_ko, D_ko_rsum) for size n_out."""
    D_d1, D_d1_rsum = _build_D(_C1_INT, crt, n_out)
    D_d2, D_d2_rsum = _build_D(_C2_INT, crt, n_out)
    if ko_sigma > 0.0:
        D_ko, D_ko_rsum = _build_D(_CKO_INT, crt, n_out)
    else:
        D_ko = D_ko_rsum = None
    return (D_d1, D_d1_rsum, D_d2, D_d2_rsum, D_ko, D_ko_rsum)


# ── Shared patch extraction ──────────────────────────────────────────────────

def _patch_starts(nbx, nby):
    ii, jj = jnp.meshgrid(jnp.arange(nbx, dtype=jnp.int32),
                           jnp.arange(nby, dtype=jnp.int32), indexing='ij')
    return jnp.stack([ii.ravel(), jj.ravel()], axis=1)


# ── Public API: fused RHS ─────────────────────────────────────────────────────

def make_fused_ozaki_rhs(Nx_tot, Ny_tot, dx, dy, cs, L, K1, K2, ko_sigma,
                         mods_ext=None):
    """
    Build a JIT-compiled fused INT8 Ozaki RHS.  Pads with mode='edge'; the
    caller handles periodic ghost-zone sync (same contract as make_fused_rhs).

    `mods_ext` optionally overrides the extension moduli (see
    CrtFloatConverter docstring for the dynamic-range bound).

    Returns: fused_ozaki_rhs(data: (NF,Nx_tot,Ny_tot)) → (NF,Nx_tot,Ny_tot)
    """
    crt = CrtFloatConverter(mods_ext=mods_ext)
    pde = _make_pde_fn(dx, dy, cs, L, K1, K2, ko_sigma, crt)
    Dset = _build_Dset(crt, ko_sigma, BS)

    pad_x = (BS - Nx_tot % BS) % BS
    pad_y = (BS - Ny_tot % BS) % BS
    Nx_al = Nx_tot + pad_x
    Ny_al = Ny_tot + pad_y
    nbx   = Nx_al // BS
    nby   = Ny_al // BS
    ph = pw = BS + 2 * NG
    starts = _patch_starts(nbx, nby)

    def kernel_fn(tile):
        return pde(tile, BS, Dset)

    @jax.jit
    def fused_ozaki_rhs(data):
        aligned = jnp.pad(data,    ((0, 0), (0, pad_x), (0, pad_y)), mode='edge')
        padded  = jnp.pad(aligned, ((0, 0), (NG, NG),   (NG, NG)),   mode='edge')

        def extract_field(u_f):
            def extract_one(ij):
                return lax.dynamic_slice(u_f, (ij[0]*BS, ij[1]*BS), (ph, pw))
            return vmap(extract_one)(starts)

        patches     = vmap(extract_field)(padded).transpose(1, 0, 2, 3)
        rhs_patches = vmap(kernel_fn)(patches)

        rhs_al = (rhs_patches
                  .reshape(nbx, nby, NF, BS, BS)
                  .transpose(2, 0, 3, 1, 4)
                  .reshape(NF, Nx_al, Ny_al))
        return rhs_al[:, :Nx_tot, :Ny_tot]

    return fused_ozaki_rhs


# ── Public API: temporally fused full RK4 step ────────────────────────────────

def make_fused_ozaki_step(Nx_tot, Ny_tot, ng, dx, dy, cs, L, K1, K2, ko_sigma, dt,
                          mods_ext=None):
    """
    Build a JIT-compiled temporally fused INT8 Ozaki RK4 step (periodic BC only).

    All four RK4 stages run inside a single tiled pass with halo NG_T=12.  Works
    on the physical interior (strips ghost zones, wraps periodically), so no
    external BC passes are needed; ghost zones of the result are refilled.

    `mods_ext` optionally overrides the extension moduli (see
    CrtFloatConverter docstring for the dynamic-range bound).

    Returns: fused_ozaki_step(data: (NF,Nx_tot,Ny_tot)) → (NF,Nx_tot,Ny_tot)
    """
    crt = CrtFloatConverter(mods_ext=mods_ext)
    pde = _make_pde_fn(dx, dy, cs, L, K1, K2, ko_sigma, crt)

    # One D-set per stage output size: BS+18, BS+12, BS+6, BS.
    stage_nouts = [BS + 2 * NG * (3 - i) for i in range(4)]
    stage_Ds    = [_build_Dset(crt, ko_sigma, n) for n in stage_nouts]

    def step_kernel(tile):
        # tile: (NF, BS+2*NG_T, BS+2*NG_T) = (NF, BS+24, BS+24)
        k1 = pde(tile, stage_nouts[0], stage_Ds[0])            # (NF, BS+18, BS+18)
        s1 = tile[:, NG:-NG, NG:-NG] + 0.5 * dt * k1

        k2 = pde(s1, stage_nouts[1], stage_Ds[1])              # (NF, BS+12, BS+12)
        s2 = tile[:, 2*NG:-2*NG, 2*NG:-2*NG] + 0.5 * dt * k2

        k3 = pde(s2, stage_nouts[2], stage_Ds[2])              # (NF, BS+6, BS+6)
        s3 = tile[:, 3*NG:-3*NG, 3*NG:-3*NG] + dt * k3

        k4 = pde(s3, stage_nouts[3], stage_Ds[3])              # (NF, BS, BS)

        k1_c = k1[:, 3*NG:3*NG + BS, 3*NG:3*NG + BS]
        k2_c = k2[:, 2*NG:2*NG + BS, 2*NG:2*NG + BS]
        k3_c = k3[:,   NG:  NG + BS,   NG:  NG + BS]
        s0_c = tile[:, 4*NG:4*NG + BS, 4*NG:4*NG + BS]

        return s0_c + (dt / 6.0) * (k1_c + 2.0*k2_c + 2.0*k3_c + k4)

    Nx = Nx_tot - 2 * ng
    Ny = Ny_tot - 2 * ng
    pad_x = (BS - Nx % BS) % BS
    pad_y = (BS - Ny % BS) % BS
    Nx_al = Nx + pad_x
    Ny_al = Ny + pad_y
    nbx   = Nx_al // BS
    nby   = Ny_al // BS
    ph = pw = BS + 2 * NG_T
    starts = _patch_starts(nbx, nby)

    @jax.jit
    def fused_ozaki_step(data):
        # Strip ghost zones — wrap the true periodic interior, not ghost copies.
        interior = data[:, ng:ng + Nx, ng:ng + Ny]
        aligned = jnp.pad(interior, ((0, 0), (0, pad_x), (0, pad_y)), mode='wrap')
        padded  = jnp.pad(aligned,  ((0, 0), (NG_T, NG_T), (NG_T, NG_T)), mode='wrap')

        def extract_field(u_f):
            def extract_one(ij):
                return lax.dynamic_slice(u_f, (ij[0]*BS, ij[1]*BS), (ph, pw))
            return vmap(extract_one)(starts)

        patches     = vmap(extract_field)(padded).transpose(1, 0, 2, 3)
        new_patches = vmap(step_kernel)(patches)

        interior_new = (new_patches
                        .reshape(nbx, nby, NF, BS, BS)
                        .transpose(2, 0, 3, 1, 4)
                        .reshape(NF, Nx_al, Ny_al))[:, :Nx, :Ny]

        out = data.at[:, ng:ng + Nx, ng:ng + Ny].set(interior_new)
        out = out.at[:, 0:ng,  :].set(out[:, Nx_tot - 2*ng:Nx_tot - ng, :])
        out = out.at[:, -ng:,  :].set(out[:, ng:2*ng, :])
        out = out.at[:, :, 0:ng].set(out[:, :, Ny_tot - 2*ng:Ny_tot - ng])
        out = out.at[:, :, -ng:].set(out[:, :, ng:2*ng])
        return out

    return fused_ozaki_step
