"""
Pallas Ozaki RHS — entire CRT pipeline runs inside a single GPU kernel per tile.

Motivation
----------
The XLA-fused `fused_ozaki` path spends 88% of GPU time in ~190 separate
fused kernels per RK4 step (top 8 are the unrolled Garner CRT levels), with
no shared memory used — intermediate residue tensors round-trip through L2
between every Garner level.

This module replaces that pipeline with ONE Pallas kernel per spatial tile:

    HBM read (BFP48)  ──>  [limb→residues → INT8 GEMM → bias correction
                            → Garner CRT → PDE assembly] ──>  HBM write

All CRT intermediates stay in shared memory / registers.  No L2 traffic
between pipeline stages.

Design notes
------------
* BS=16 by default (override with MCS_PALLAS_BS).  Per-tile working set
  (~70 KB at BS=16) fits comfortably in Hopper's 228 KB/SM shared memory;
  many tiles run concurrently per SM.  XLA's BS=16 was tragic because each
  tile became its own ~190-kernel cascade; here we have ONE kernel per tile.

* BFP48 transfer (MEMORY OPTIMIZATION over fp64).  The kernel consumes the
  field as three int16 mantissa limbs + one shared exponent per haloed tile
  (block floating point, 48-bit signed mantissa), NOT float64.  Transfer is
  3×int16 = 6 bytes/cell vs float64's 8 — a 25% reduction in the kernel's
  input read, and the natural input to RNS (the mantissa IS the scaled
  integer).  Packing happens once in the wrapper (`float64 → BFP48`), which
  also moves the per-tile max-reduce + log2 BFP scaling OUT of the hot kernel.
  NOTE: as wired today fp64 is still the canonical between-step state, so the
  pack reads fp64 each RHS call — the full HBM win lands only when BFP48
  becomes the stored state (canonical-BFP48 / Y3 layout is future work; see
  memory `pallas_pow2_strategy`).  To go below 48 bits (BFP32, 50% off) drop
  base moduli at the cost of dynamic range.

* Residues computed PURELY in int32 from the BFP48 limbs (no fp64, no int64):
    r ≡ p0 + p1·(2^16 mod m) + p2·(2^32 mod m) − is_neg·(2^48 mod m)  (mod m)
  Each term ≤ 2^16·256, sum < 2^26 → int32.  Exact for |mantissa| ≤ 2^47,
  m ≤ 253.  This skips base_extension — every modulus is independent.

* INT8 GEMM is HARDCODED ON.  `jnp.dot(int8, int8, preferred_element_type=int32)`
  lowers to the int8 IMMA/WGMMA tensor cores on Hopper (H200) — the actual
  point of Ozaki (~30× int8/fp64 throughput).  The old `USE_INT8_GEMM=False`
  gate (cuBLASLt + CUDA-graph-capture breakage) was an XLA-`dot_general`
  problem; the Pallas dot lowers to Triton's own MMA, not cuBLASLt, so it does
  not apply here.

Verification
------------
Tests use Pallas's `interpret=True` mode on CPU (correctness only — interpret
does NOT exercise the Triton lowering).  Speed AND Triton-lowering validity are
GPU-only and must be checked on Hopper (H200).

JAX version compatibility — TARGET: 0.9.2
-----------------------------------------
This kernel is written for the JAX 0.9.x Triton lowering, whose supported
primitive set is the binding constraint (verified against
`jax/_src/pallas/triton/lowering.py`):

  * NO `slice_p` / `dynamic_slice_p` / `gather_p` / `scatter_p`.  → indexing is
    done on REFS (load_p), per-modulus/per-field, into Python lists indexed at
    trace time; cropping a computed tile uses `jnp.split` (split_p); the output
    is written PER CHANNEL (no `jnp.stack`, no `.at[].add` scatter).
  * Power-of-2 tile sizes required (`_check_tensor_size`) → `MCS_PALLAS_PAD_POW2`
    (auto-on for any non-CPU backend) pads H 22→32 and NF 10→16.

(0.8.1's older Triton was the *opposite* on indexing — it had no `dynamic_slice`,
which is why the original `dynamic_index_in_dim`-based kernel failed on 0.8.1 GPU.)
AMR does NOT use this kernel (it runs the pure-JAX
`fused_rhs_fp._make_kernel_fn`), so it is unaffected either way.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

jax.config.update("jax_enable_x64", True)

# JAX 0.10 defaults `jax_pallas_use_mosaic_gpu = True`.  Mosaic GPU isn't
# installed in most environments yet, so pallas_call falls through without
# selecting a backend even when Triton IS available.  Force Triton.
# Override globally with `export JAX_PALLAS_USE_MOSAIC_GPU=1` if Mosaic
# becomes available and preferred.
if not os.environ.get("JAX_PALLAS_USE_MOSAIC_GPU"):
    jax.config.update("jax_pallas_use_mosaic_gpu", False)

# ── Tunables (env-overridable) ─────────────────────────────────────────────────

# Tile edge length.  Smaller = more tiles (better SM occupancy), smaller
# per-tile working set (fits in shared memory).  BS=16 is the recommended
# value for Pallas; BS=32 also works if shared memory budget allows.  Must
# be a power of 2 (Pallas-Triton constraint).
BS = int(os.environ.get("MCS_PALLAS_BS", 16))

# 6th-order stencil halo.
NG = 3

# Field count.
NF = 10


def _next_pow2(n):
    return 1 if n <= 1 else 1 << ((n - 1).bit_length())


# Pallas-Triton requires every operation's total array size to be a power of 2
# (`_check_tensor_size`) — confirmed on GPU for BOTH JAX 0.8.1 and 0.9.x, so this
# is NOT version-specific (earlier belief that 0.8.x was permissive was an artifact
# of CPU *interpret* mode, which skips the check).  On a real GPU the (NF=10,
# H=22, H=22) tile (=4840, not pow2) is rejected, so we MUST pad to pow2 (H_PAD=32,
# NF_PAD=16).  Padding wastes ~3x per-tile compute, but it's mandatory on GPU.
# Default: ON for any non-CPU backend (GPU/TPU); OFF on CPU interpret (not needed,
# faster).  Override explicitly with MCS_PALLAS_PAD_POW2=0/1.
if "MCS_PALLAS_PAD_POW2" in os.environ:
    _PAD_POW2 = bool(int(os.environ["MCS_PALLAS_PAD_POW2"]))
else:
    _PAD_POW2 = jax.default_backend() != "cpu"

H_ORIG = BS + 2 * NG          # actual tile width (e.g. 22 at BS=16)
H_PAD  = _next_pow2(H_ORIG) if _PAD_POW2 else H_ORIG
NF_PAD = _next_pow2(NF)       if _PAD_POW2 else NF

# INT8 tensor-core GEMM is now HARDCODED ON in the kernel (the dot keeps int8
# operands with int32 accumulation → Triton IMMA/WGMMA on Hopper).  This flag is
# retained only for reference / parity with ozaki.py and fused_rhs_ozaki.py; it
# is NOT read in this module.  The historical cuBLASLt CUDA-graph-capture issue
# (see feedback_gpu_bugs.md) was an XLA-dot_general problem and does not apply to
# the Triton-lowered Pallas dot.
USE_INT8_GEMM = True  # informational only — see kernel for the hardcoded int8 dot

# Auto-pick interpret mode on CPU (Pallas's kernel-on-CPU simulator).
_INTERPRET = jax.default_backend() == "cpu"

# ── Field indices ──────────────────────────────────────────────────────────────

EX, EY, EZ = 0, 1, 2
BX, BY, BZ = 3, 4, 5
XI, PI, PSI, PHI = 6, 7, 8, 9

_D1_FIELDS = (EX, EY, EZ, BX, BY, BZ, XI, PSI, PHI)  # PI never differentiated

from mcs2d.schemes.ozaki import CrtFloatConverter, _garner_constants


# ── Precomputed banded D matrices ──────────────────────────────────────────────

def _build_D(coeffs_int, mods_full):
    """
    Build the banded stencil matrix at output size BS.  The contraction (K)
    axis is sized to H_PAD (the power-of-2 tile width).  Columns beyond the
    real K_orig = BS + 2*NG hold zeros — these cells get -128 in the
    bias-shifted GEMM and are cancelled by the `-K_pad·128²` term.

    Returns:
      D_i8 : (k_full, BS, H_PAD)  int8   (bias-shifted by -128)
      Drs  : (k_full, BS)          int32  (sum of original D residues per row)
    """
    D = np.zeros((BS, H_PAD), dtype=np.int64)
    for r in range(BS):
        c = r + NG
        D[r, c - 3 : c + 4] = coeffs_int
    D_i8 = np.stack([(D % m - 128).astype(np.int8) for m in mods_full])
    Drs  = np.stack([(D % m).sum(axis=1, dtype=np.int32) for m in mods_full])
    return D_i8, Drs


# ── Public API ─────────────────────────────────────────────────────────────────

def make_pallas_ozaki_rhs(Nx_tot, Ny_tot, dx, dy, cs, L, K1, K2, ko_sigma,
                          mods_ext=None, profile_trunc=None):
    """
    Build a JIT-compiled Pallas-kernel RHS.  Same contract as
    fused_rhs_ozaki.make_fused_ozaki_rhs: caller handles BC ghost-zone sync.

    Returns: pallas_ozaki_rhs(data: (NF,Nx_tot,Ny_tot)) → (NF,Nx_tot,Ny_tot)

    profile_trunc (PROFILING ONLY; default None = full kernel): truncate each
    Ozaki derivative at a pipeline stage so timing differences attribute cost to
    each stage (the Amdahl breakdown):
        0 = limb load + unpack only (no residues/GEMM/CRT)
        1 = + residue conversion          (T1−T0 = residue cost)
        2 = + int8 GEMM + bias + mod-reduce (T2−T1 = GEMM cost)
        None/3 = full (Garner CRT + unscale; full−T2 = CRT cost)
    Each truncation returns a valid (BS,BS) float that flows into the PDE
    assembly, so upstream work is kept alive (not DCE'd).  When None, the code
    path is byte-for-byte the production kernel — profiling adds nothing.
    """
    crt = CrtFloatConverter(mods_ext=mods_ext)
    mods_full = crt.mods_full_py
    k_full = len(mods_full)
    m_prod_base_half = float(crt._m_prod_base_py) / 2.0
    m_prod_full = float(crt._m_prod_full_py)
    basis_full = list(crt._basis_full)
    gdiag, m_acc = _garner_constants(mods_full)

    # Precompute D matrices (per derivative type, all at output size BS)
    D_d1_np, Drs_d1_np = _build_D([-1, 9, -45, 0, 45, -9, 1], mods_full)
    D_d2_np, Drs_d2_np = _build_D([2, -27, 270, -490, 270, -27, 2], mods_full)
    if ko_sigma > 0.0:
        # KO δ⁶ (damping). NOT the negated [-1,6,-15,20,-15,6,-1] — see
        # floating_point.CKO; the wrong sign drives a grid-scale instability.
        D_ko_np, Drs_ko_np = _build_D([1, -6, 15, -20, 15, -6, 1], mods_full)

    # Scalar PDE constants (closed over by the kernel).
    inv_60dx   = 1.0 / (60.0 * dx)
    inv_60dy   = 1.0 / (60.0 * dy)
    inv_180dx2 = 1.0 / (180.0 * dx ** 2)
    inv_180dy2 = 1.0 / (180.0 * dy ** 2)
    ko_cx      = ko_sigma / (64.0 * dx)
    ko_cy      = ko_sigma / (64.0 * dy)
    two_cs_L   = 2.0 * cs * L
    two_L      = 2.0 * L

    # ── Convert constants to jnp arrays (captured as kernel closure) ──────
    D_d1_jnp     = jnp.array(D_d1_np)         # (k, BS, K_pad) int8
    Drs_d1_jnp   = jnp.array(Drs_d1_np)        # (k, BS) int32
    D_d2_jnp     = jnp.array(D_d2_np)
    Drs_d2_jnp   = jnp.array(Drs_d2_np)
    if ko_sigma > 0.0:
        D_ko_jnp   = jnp.array(D_ko_np)
        Drs_ko_jnp = jnp.array(Drs_ko_np)

    # ── Inner Ozaki derivative (runs inside the Pallas kernel) ────────────

    # Cropping the [NG:NG+BS] interior is a GATHER, which 0.9.2 Triton supports
    # via NONE of the obvious tools: no slice_p / dynamic_slice_p / gather_p;
    # split_p needs equal pow2 parts; concatenate is 2-arg only; fp64 `dot` hangs
    # the compiler.  So we express the gather as a one-hot SELECT-AND-REDUCE:
    #   out[..,j,..] = Σ_k onehot[k,j]·x[..,k,..],   onehot[k,j] = (k == NG+j)
    # using only iota / multiply / reduce_sum / broadcast — all well-lowered.
    # Exact (one nonzero term per output), fp64-safe, no dot/slice/concat.
    def _onehot():
        k = jax.lax.broadcasted_iota(jnp.int32, (H_PAD, BS), 0)   # row k
        j = jax.lax.broadcasted_iota(jnp.int32, (H_PAD, BS), 1)   # interior col j
        return (k == j + NG).astype(jnp.float64)                  # (H_PAD, BS)

    def _mid(x, axis):
        """Crop the size-H_PAD `axis` of 2D `x` to the BS interior."""
        C = _onehot()                                             # (H_PAD, BS)
        if axis == 0:        # x (H_PAD, W) → (BS, W)
            return (C[:, :, None] * x[:, None, :]).sum(axis=0)
        return (x[:, :, None] * C[None, :, :]).sum(axis=1)        # x (A, H_PAD) → (A, BS)

    def _crop1(out, axis):
        """Crop the H_PAD halo axis of a derivative tile to the BS interior.
        deriv axis 0: out is (BS, H_PAD) → crop axis 1; axis 1: out is (H_PAD, BS) → axis 0."""
        return _mid(out, 1) if axis == 0 else _mid(out, 0)

    def _crop2(v):
        """Crop both axes of an (H_PAD, H_PAD) tile to (BS, BS)."""
        return _mid(_mid(v, 0), 1)

    def _ozaki_one_field(p0_f, p1_f, p2_f, exp_f, D_list, Drs_list, axis):
        """
        Ozaki derivative of one field tile along `axis`, from its BFP48 mantissa
        limbs.  Returns (BS, BS) float64.

        Inputs: the three int16 limbs (p0/p1/p2) of the signed 48-bit integer
        mantissa, the per-field BFP exponent `exp_f` (= scale_exp), and the
        per-modulus stencil slabs `D_list[k]` (BS,H_PAD int8) / `Drs_list[k]`
        (BS,).  D slabs are Python lists (indexed at trace time) of REF-loaded
        arrays — NOT a value-indexed array, since 0.9.2 Triton lacks slice_p.
        The scaled value is mantissa = p0 + p1·2^16 + p2·2^32 (two's-comp 48-bit);
        the physical value is mantissa · 2^(-exp_f).

        Residues are computed PURELY in int32 from the limbs — no fp64, no int64:
            r ≡ p0 + p1·(2^16 mod m) + p2·(2^32 mod m) − is_neg·(2^48 mod m)  (mod m)
        Each term ≤ 2^16·256, sum < 2^26 — fits int32.  This replaces the old
        in-kernel BFP scaling (max-reduce + log2 + exp2), which now happens once
        in the wrapper at pack time (the per-step "startup" cost moves out of the
        hot kernel).  Cells beyond the real H_ORIG range are zero limbs → residue
        0, cancelled by the bias formula's `-H_PAD·128²` term.

        Pipeline:  limb→residues → bias-shift → int8 GEMM → bias correction
                   → mod-reduce → Garner reconstruction → unscale.
        """
        # Unsigned 16-bit limb values; sign bit is bit 47 = bit 15 of p2.
        p0u = p0_f.astype(jnp.int32) & 0xFFFF
        p1u = p1_f.astype(jnp.int32) & 0xFFFF
        p2u = p2_f.astype(jnp.int32) & 0xFFFF
        is_neg = p2u >> 15                       # 0 or 1

        if profile_trunc == 0:                   # limb load + unpack only
            return _crop2((p0u + p1u + p2u).astype(jnp.float64))

        # Residues straight from the limbs (constant-modulus → multiply-shift).
        # Kept as Python lists indexed at trace time — NOT jnp.stack'd.
        regs_signed_list = []
        col_sum_residues = []
        for k_idx, m in enumerate(mods_full):
            c16 = (1 << 16) % m
            c32 = (1 << 32) % m
            c48 = (1 << 48) % m
            r = (p0u + p1u * c16 + p2u * c32) % m   # mantissa_unsigned mod m
            r = (r - is_neg * c48) % m              # two's-complement sign fix → [0,m)
            regs_signed_list.append((r - 128).astype(jnp.int8))
            col_sum_residues.append(r)              # int32 in [0,m)

        exponent = -exp_f.astype(jnp.float64)      # unscale factor 2^(-scale_exp)

        if profile_trunc == 1:                     # + residue conversion only
            return _crop2(sum(col_sum_residues).astype(jnp.float64))

        # GEMM per modulus.  K = H_PAD (the padded contraction) is power-of-2.
        # Bias formula:  result = (D-128)@(r-128) + 128·rowsum_D + 128·colsum_r - K·128²
        # NOTE: we do NOT crop the halo axis here.  Cropping is a matmul (_crop1),
        # and per-modulus cropping would be k_full matmuls per call; instead we
        # keep the full H_PAD-wide residues and crop ONCE after Garner.  Cost: the
        # Garner CRT then runs on H_PAD-wide tiles (~2× the interior cells) — a
        # known inefficiency (inflates the CRT stage in profiling); revisit if a
        # cheap int crop becomes available.
        result_per_k = []
        for k_idx in range(k_full):
            # int8 operands + int32 accumulation → Triton lowers `jnp.dot` to
            # the int8 IMMA/WGMMA tensor-core path on Hopper (H200).  D_list and
            # regs are ALREADY the bias-shifted (−128) int8 values, so the dot
            # computes (D−128)@(r−128) directly; the bias formula adds the
            # correction.  Operands stay in [−128,124] (no int8 overflow);
            # K=H_PAD=32 terms of ≤128² accumulate well within int32.
            D_k = D_list[k_idx]                                  # (BS, H_PAD) int8
            r_k = regs_signed_list[k_idx]                        # (H_PAD, H_PAD) int8
            r_int_k = col_sum_residues[k_idx]                    # (H_PAD, H_PAD) int32
            D_rsum_k = Drs_list[k_idx]                           # (BS,)
            if axis == 0:
                gemm = jnp.dot(D_k, r_k,
                               preferred_element_type=jnp.int32)  # (BS, H_PAD)
                col_sum = r_int_k.sum(axis=0)                     # (H_PAD,)
                result = (gemm
                          + 128 * D_rsum_k.reshape(BS, 1)
                          + 128 * col_sum.reshape(1, H_PAD)
                          - H_PAD * 16384)                        # (BS, H_PAD)
            else:
                gemm = jnp.dot(r_k, D_k.T,
                               preferred_element_type=jnp.int32)  # (H_PAD, BS)
                row_sum = r_int_k.sum(axis=1)                     # (H_PAD,)
                result = (gemm
                          + 128 * D_rsum_k.reshape(1, BS)
                          + 128 * row_sum.reshape(H_PAD, 1)
                          - H_PAD * 16384)                        # (H_PAD, BS)
            # Mod-reduce by m_k → [0, m_k)  (rem_p; constant modulus).
            result_per_k.append(result % mods_full[k_idx])

        if profile_trunc == 2:                     # + int8 GEMM + bias + mod-reduce
            return _crop1(sum(result_per_k).astype(jnp.float64), axis)

        # Garner CRT reconstruction (sequential over moduli, parallel over BS,BS).
        x = [result_per_k[0]]
        for i in range(1, k_full):
            mi = mods_full[i]
            partial = x[0] % mi
            for jj, j in enumerate(range(1, i)):
                partial = (partial + x[j] * m_acc[i][jj]) % mi
            xi = ((result_per_k[i] - partial) * gdiag[i]) % mi
            x.append(xi)

        # Float64 weighted sum (still H_PAD-wide along the halo axis).
        out = x[0].astype(jnp.float64) * basis_full[0]
        for i in range(1, k_full):
            out = out + x[i].astype(jnp.float64) * basis_full[i]
        out = jnp.where(out > m_prod_full / 2.0, out - m_prod_full, out)

        # Crop the halo axis to the BS interior (matmul) and undo BFP scaling.
        return _crop1(out, axis) * jnp.exp2(exponent)

    # ── The full per-tile Pallas kernel ───────────────────────────────────
    #
    # Pallas in JAX 0.10 forbids closing over jnp arrays; D matrices must be
    # passed as kernel inputs.  We use grid=(n_patches,) with BlockSpec to
    # have each grid block process one patch, with the D matrices broadcast
    # (same data) to every block.
    # ───────────────────────────────────────────────────────────────────────

    # Per-block input shapes:
    #   p0/p1/p2:       slice (1, NF, H, H) int16        BFP48 mantissa limbs
    #   exps:           slice (1, NF)        int32        per-field BFP exponent
    #   D_d1, D_d2:     full (k_full, BS, K_pad)          broadcast to all blocks
    #   Drs_d1, Drs_d2: full (k_full, BS)
    #   (D_ko, Drs_ko optional, only when ko_sigma > 0)

    if ko_sigma > 0.0:
        def kernel(p0_ref, p1_ref, p2_ref, exp_ref,
                   D_d1_ref, Drs_d1_ref, D_d2_ref, Drs_d2_ref,
                   D_ko_ref, Drs_ko_ref, out_ref):
            _kernel_body(p0_ref, p1_ref, p2_ref, exp_ref,
                         D_d1_ref, Drs_d1_ref, D_d2_ref, Drs_d2_ref,
                         D_ko_ref, Drs_ko_ref, out_ref, ko_enabled=True)
    else:
        def kernel(p0_ref, p1_ref, p2_ref, exp_ref,
                   D_d1_ref, Drs_d1_ref, D_d2_ref, Drs_d2_ref, out_ref):
            _kernel_body(p0_ref, p1_ref, p2_ref, exp_ref,
                         D_d1_ref, Drs_d1_ref, D_d2_ref, Drs_d2_ref,
                         None, None, out_ref, ko_enabled=False)

    def _kernel_body(p0_ref, p1_ref, p2_ref, exp_ref,
                     D_d1_ref, Drs_d1_ref, D_d2_ref, Drs_d2_ref,
                     D_ko_ref, Drs_ko_ref, out_ref, ko_enabled):
        # Per-modulus stencil slabs via REF indexing (load_p).  Value-indexing a
        # loaded (k,BS,H_PAD) array would need slice_p — absent in 0.9.2 Triton —
        # so we ref-index per modulus into Python lists indexed at trace time.
        D_d1L   = [D_d1_ref[k]   for k in range(k_full)]   # each (BS, H_PAD) int8
        Drs_d1L = [Drs_d1_ref[k] for k in range(k_full)]   # each (BS,) int32
        D_d2L   = [D_d2_ref[k]   for k in range(k_full)]
        Drs_d2L = [Drs_d2_ref[k] for k in range(k_full)]

        # Per-field derivative: limbs loaded per field via REF indexing.
        def _od(f, DL, DrsL, axis):
            return _ozaki_one_field(p0_ref[0, f], p1_ref[0, f], p2_ref[0, f],
                                    exp_ref[0, f], DL, DrsL, axis)

        # d1 for 9 fields (PI never differentiated).
        dx_all = [_od(f, D_d1L, Drs_d1L, 0) * inv_60dx for f in _D1_FIELDS]
        dy_all = [_od(f, D_d1L, Drs_d1L, 1) * inv_60dy for f in _D1_FIELDS]
        (dEx_dx, dEy_dx, dEz_dx, dBx_dx, dBy_dx, dBz_dx,
         dxi_dx, dPsi_dx, dPhi_dx) = dx_all
        (dEx_dy, dEy_dy, dEz_dy, dBx_dy, dBy_dy, dBz_dy,
         dxi_dy, dPsi_dy, dPhi_dy) = dy_all

        # d2 for xi only
        d2xi_dx2 = _od(XI, D_d2L, Drs_d2L, 0) * inv_180dx2
        d2xi_dy2 = _od(XI, D_d2L, Drs_d2L, 1) * inv_180dy2

        # Interior point VALUES (not derivatives) for the nonlinear PDE algebra.
        # Reconstruct the fp64 field from the BFP48 limbs: value =
        #   (p0 + p1·2^16 + p2·2^32 − is_neg·2^48) · 2^(-exp_f).
        # All terms < 2^48 < 2^52, so the float64 reconstruction is exact; crop
        # the interior with _crop2 (selection matmul, not slice/split).
        def _value_interior(f):
            p0u = p0_ref[0, f].astype(jnp.int32) & 0xFFFF
            p1u = p1_ref[0, f].astype(jnp.int32) & 0xFFFF
            p2u = p2_ref[0, f].astype(jnp.int32) & 0xFFFF
            is_neg = (p2u >> 15).astype(jnp.float64)
            mant = (p0u.astype(jnp.float64)
                    + p1u.astype(jnp.float64) * 65536.0
                    + p2u.astype(jnp.float64) * 4294967296.0
                    - is_neg * 281474976710656.0)        # 2^48
            val = mant * jnp.exp2(-exp_ref[0, f].astype(jnp.float64))
            return _crop2(val)                           # (BS, BS)
        Ex  = _value_interior(EX)
        Ey  = _value_interior(EY)
        Ez  = _value_interior(EZ)
        Bx  = _value_interior(BX)
        By  = _value_interior(BY)
        Bz  = _value_interior(BZ)
        Pi  = _value_interior(PI)
        Psi = _value_interior(PSI)
        Phi = _value_interior(PHI)

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

        chans = [dt_Ex, dt_Ey, dt_Ez, dt_Bx, dt_By, dt_Bz,
                 dt_xi, dt_Pi, dt_Psi, dt_Phi]

        if ko_enabled:
            D_koL   = [D_ko_ref[k]   for k in range(k_full)]
            Drs_koL = [Drs_ko_ref[k] for k in range(k_full)]
            for f in range(NF):
                chans[f] = chans[f] + (_od(f, D_koL, Drs_koL, 0) * ko_cx
                                       + _od(f, D_koL, Drs_koL, 1) * ko_cy)

        # Per-channel stores (swap_p) — no jnp.stack (concat) and no .at[].add
        # scatter (scatter_p absent in 0.9.2 Triton).  Pad channels are zeroed.
        for f in range(NF):
            out_ref[0, f] = chans[f]
        if NF_PAD > NF:
            zero_tile = jnp.zeros((BS, BS), jnp.float64)
            for f in range(NF, NF_PAD):
                out_ref[0, f] = zero_tile

    # ── Outer wrapper: pad → extract patches → pallas_call → reassemble ──

    pad_x = (BS - Nx_tot % BS) % BS
    pad_y = (BS - Ny_tot % BS) % BS
    Nx_al = Nx_tot + pad_x
    Ny_al = Ny_tot + pad_y
    nbx   = Nx_al // BS
    nby   = Ny_al // BS
    n_patches = nbx * nby

    ii, jj = jnp.meshgrid(jnp.arange(nbx, dtype=jnp.int32),
                           jnp.arange(nby, dtype=jnp.int32), indexing='ij')
    patch_starts = jnp.stack([ii.ravel(), jj.ravel()], axis=1)

    # BlockSpec: per-block input is one patch (size 1 along the n_patches axis);
    # D matrices are broadcast (block shape == full shape, index_map → 0s).
    # The patch arrives as three int16 BFP48 limb tiles + a per-field exponent
    # vector instead of one fp64 tile (the memory optimization, see wrapper).
    # All tile shapes stay powers of 2 (Triton requirement): (NF_PAD,H_PAD,H_PAD).
    limb_spec  = pl.BlockSpec((1, NF_PAD, H_PAD, H_PAD), lambda i: (i, 0, 0, 0))
    exp_spec   = pl.BlockSpec((1, NF_PAD),                lambda i: (i, 0))
    D_spec     = pl.BlockSpec((k_full, BS, H_PAD),        lambda i: (0, 0, 0))
    Drs_spec   = pl.BlockSpec((k_full, BS),                lambda i: (0, 0))
    out_spec   = pl.BlockSpec((1, NF_PAD, BS, BS),        lambda i: (i, 0, 0, 0))

    limb_in = [limb_spec, limb_spec, limb_spec, exp_spec]
    if ko_sigma > 0.0:
        in_specs = limb_in + [D_spec, Drs_spec, D_spec, Drs_spec, D_spec, Drs_spec]
    else:
        in_specs = limb_in + [D_spec, Drs_spec, D_spec, Drs_spec]

    pallas_fn = pl.pallas_call(
        kernel,
        grid=(n_patches,),
        in_specs=in_specs,
        out_specs=out_spec,
        # Output is padded to NF_PAD along the field axis; cropped below.
        out_shape=jax.ShapeDtypeStruct((n_patches, NF_PAD, BS, BS), jnp.float64),
        interpret=_INTERPRET,
    )

    nf_pad   = NF_PAD - NF                  # extra zero-fields
    spat_pad = H_PAD - H_ORIG               # extra spatial cells (pow2 pad)

    @jax.jit
    def pallas_ozaki_rhs(data):
        aligned = jnp.pad(data,    ((0, 0), (0, pad_x), (0, pad_y)), mode='edge')
        padded  = jnp.pad(aligned, ((0, 0), (NG, NG),   (NG, NG)),   mode='edge')

        def extract_field(u_f):
            def extract_one(ij):
                return jax.lax.dynamic_slice(
                    u_f, (ij[0] * BS, ij[1] * BS), (H_ORIG, H_ORIG))
            return jax.vmap(extract_one)(patch_starts)

        patches = jax.vmap(extract_field)(padded).transpose(1, 0, 2, 3)
        # patches: (n_patches, NF, H_ORIG, H_ORIG) float64

        # ── Pack each (patch, field) tile to BFP48 (the memory optimization) ──
        # One shared exponent per haloed tile; the 48-bit signed integer mantissa
        # is split into three int16 limbs.  Transfer to the kernel is 3×int16 =
        # 6 bytes/cell vs float64's 8 (25% less).  The per-tile max-reduce + log2
        # that used to run INSIDE the kernel happens here once, at pack time.
        # (int64 math is fine here — this is plain XLA, not the Triton kernel.)
        max_val   = jnp.max(jnp.abs(patches), axis=(2, 3), keepdims=True)
        scale_exp = jnp.where(max_val == 0.0, 0.0,
                              jnp.floor(jnp.log2((2.0 ** 47 - 1) / max_val)))
        scaled_int = jnp.floor(patches * jnp.exp2(scale_exp) + 0.5).astype(jnp.int64)
        p0 = (scaled_int & 0xFFFF).astype(jnp.int16)
        p1 = ((scaled_int >> 16) & 0xFFFF).astype(jnp.int16)
        p2 = ((scaled_int >> 32) & 0xFFFF).astype(jnp.int16)
        exps = scale_exp[:, :, 0, 0].astype(jnp.int32)        # (n_patches, NF)

        # Pad NF + spatial to power-of-2 (Triton).  Zero limbs → residue 0 →
        # cancelled by the bias formula; padded fields carry exponent 0.
        pad4 = ((0, 0), (0, nf_pad), (0, spat_pad), (0, spat_pad))
        p0 = jnp.pad(p0, pad4)
        p1 = jnp.pad(p1, pad4)
        p2 = jnp.pad(p2, pad4)
        exps = jnp.pad(exps, ((0, 0), (0, nf_pad)))
        # limbs now: (n_patches, NF_PAD, H_PAD, H_PAD); exps: (n_patches, NF_PAD)

        if ko_sigma > 0.0:
            rhs_patches = pallas_fn(p0, p1, p2, exps, D_d1_jnp, Drs_d1_jnp,
                                    D_d2_jnp, Drs_d2_jnp, D_ko_jnp, Drs_ko_jnp)
        else:
            rhs_patches = pallas_fn(p0, p1, p2, exps, D_d1_jnp, Drs_d1_jnp,
                                    D_d2_jnp, Drs_d2_jnp)
        # rhs_patches: (n_patches, NF_PAD, BS, BS) — crop the NF axis back.
        rhs_patches = rhs_patches[:, :NF]

        rhs_al = (rhs_patches
                  .reshape(nbx, nby, NF, BS, BS)
                  .transpose(2, 0, 3, 1, 4)
                  .reshape(NF, Nx_al, Ny_al))
        return rhs_al[:, :Nx_tot, :Ny_tot]

    return pallas_ozaki_rhs
