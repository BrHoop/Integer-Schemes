import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np
from functools import reduce

jax.config.update("jax_enable_x64", True)

# Pad the K (contraction) dim of every Ozaki GEMM up to this multiple so the
# call hits INT8 tensor cores on Ampere / Hopper.  Padded D residues are 0
# (-128 in bias-shifted i8); the bias formula's `- K*16384` term uses
# K = D.shape[2] (padded), exactly cancelling padded -128*-128 contributions.
_K_MULTIPLE = 16

# INT8 GEMM via cuBLASLt fires Ampere/Hopper tensor cores but can break CUDA
# graph capture on some JAX/CUDA configurations (cuBLASLt does runtime kernel
# autotuning).  Default = False (int32 GEMM, no tensor cores but always
# graph-safe).  Flip to True on hardware/JAX combos that handle INT8 cuBLASLt
# under graph capture — then re-verify with the test suite and benchmark.
USE_INT8_GEMM = False


def _pad_K(k_orig):
    return ((k_orig + _K_MULTIPLE - 1) // _K_MULTIPLE) * _K_MULTIPLE


# ==============================================================================
# 1. BFP48 COMPRESSION & RECONSTRUCTION
# ==============================================================================
@jax.jit
def float64_to_bfp48(block_32):
    max_val = jnp.max(jnp.abs(block_32))
    scale_exp = jnp.where(max_val == 0, 0.0, jnp.floor(jnp.log2((2.0**47 - 1) / max_val)))
    exponent = scale_exp.astype(jnp.int32)
    scaled = block_32 * (2.0 ** scale_exp)
    scaled_int = jnp.floor(scaled + 0.5).astype(jnp.int64)

    p0 = (scaled_int & 0xFFFF).astype(jnp.int16)
    p1 = ((scaled_int >> 16) & 0xFFFF).astype(jnp.int16)
    p2 = ((scaled_int >> 32) & 0xFFFF).astype(jnp.int16)
    return jnp.stack([p0, p1, p2], axis=-1), exponent

@jax.jit
def bfp48_to_float64(mantissas, exponent):
    p0 = mantissas[..., 0].astype(jnp.int64) & 0xFFFF
    p1 = mantissas[..., 1].astype(jnp.int64) & 0xFFFF
    p2 = mantissas[..., 2].astype(jnp.int64) & 0xFFFF
    combined_int = p0 | (p1 << 16) | (p2 << 32)
    is_neg = combined_int >= (1 << 47)
    combined_int = jnp.where(is_neg, combined_int - (1 << 48), combined_int)
    return combined_int * (2.0 ** (-exponent.astype(jnp.float64)))

# ==============================================================================
# 2. RNS & CRT CORE MATH
# ==============================================================================
def extended_gcd(a, b):
    if a == 0: return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    return d, x, x1

def modinv(a, m):
    d, x, y = extended_gcd(a, m)
    if d != 1: raise ValueError("Modular inverse does not exist")
    return x % m

def _garner_constants(mods):
    """Precompute Garner mixed-radix constants for a list of Python-int moduli.

    Returns (gdiag, m_acc) where:
      gdiag[i]    = modinv(prod(mods[0..i-1]), mods[i])           (Python int)
      m_acc[i][j] = prod(mods[0..j-1]) % mods[i]  for j=1..i-1    (Python ints)
    All are compile-time constants, so the modular reductions that use them
    become multiply-shift instead of runtime integer division.
    """
    n = len(mods)
    gdiag, m_acc = [], []
    for i in range(n):
        p = 1
        for k in range(i):
            p *= mods[k]
        gdiag.append(modinv(p, mods[i]) if i > 0 else 0)
        accs, acc = [], 1
        for j in range(1, i):
            acc = (acc * mods[j - 1]) % mods[i]
            accs.append(acc)
        m_acc.append(accs)
    return gdiag, m_acc


class CrtFloatConverter:
    """RNS/CRT converter with compile-time-constant moduli.

    All modular reductions use Python-int moduli baked into jit closures, so
    XLA lowers `x % m` to multiply-shift rather than emitting a runtime integer
    division (which is what indexing `mods_array[i]` would force).

    Dynamic-range bound for exact reconstruction
    --------------------------------------------
    For a stencil row D with L1 norm ‖D‖₁, the max signed GEMM result is
    ‖D‖₁ · (M_prod_base / 2), so M_prod_full must satisfy
        M_prod_full  >  2 · ‖D‖₁ · M_prod_base / 2  =  ‖D‖₁ · M_prod_base
    i.e. M_prod_ext = M_prod_full / M_prod_base  >  ‖D‖₁ (in bits: log₂‖D‖₁).

    For 6th-order stencils:
        C1  ‖D‖₁ = 110   (≈ 6.8 bits)
        C2  ‖D‖₁ = 1088  (≈ 10.1 bits)   ← tightest constraint
        CKO ‖D‖₁ = 64    (≈ 6.0 bits)

    Default ext moduli [239, 233, 229] give M_prod_ext ≈ 12.7M ≈ 23.6 bits
    (13.5 bits of headroom over C2).  Dropping to [239, 233] still gives
    15.8 bits — safe.  Dropping to just [239] gives 7.9 bits — UNSAFE for C2,
    OK for C1/CKO only.  The full 9-moduli default is recommended unless you
    have measured an actual speedup and your problem stays within the reduced
    bound (small amplitudes far from the BFP scaling ceiling).

    Reducing moduli does NOT cause slow instability; insufficient range causes
    wraparound — large, immediately visible errors — not subtle drift.
    """

    DEFAULT_MODS_BASE = [253, 251, 249, 247, 245, 241]
    # 8 moduli (6 base + 2 ext) — verified safe for all stencils with ~5.7 bits
    # headroom on the tightest constraint (C2).  Also gives *better* float64
    # precision than 9 moduli (smaller basis values → less catastrophic
    # cancellation in Garner).  Reduces unrolled-CRT compile time too.
    DEFAULT_MODS_EXT  = [239, 233]

    def __init__(self, mods_base=None, mods_ext=None):
        self.mods_base_list = list(mods_base) if mods_base is not None else list(self.DEFAULT_MODS_BASE)
        self.mods_ext_list  = list(mods_ext)  if mods_ext  is not None else list(self.DEFAULT_MODS_EXT)
        self.mods_full_list = self.mods_base_list + self.mods_ext_list

        self.k_full = len(self.mods_full_list)
        self.k_base = len(self.mods_base_list)

        # Python-int copies used as compile-time constants in the closures.
        self.mods_base_py = [int(m) for m in self.mods_base_list]
        self.mods_ext_py  = [int(m) for m in self.mods_ext_list]
        self.mods_full_py = [int(m) for m in self.mods_full_list]

        self._m_prod_base_py = reduce(lambda x, y: x * y, self.mods_base_list)
        self._m_prod_full_py = reduce(lambda x, y: x * y, self.mods_full_list)

        # Float/int scalars still needed by callers (BFP scaling, sign tests).
        self.M_prod_base = jnp.array(self._m_prod_base_py, dtype=jnp.int64)
        self.M_prod_full = jnp.array(float(self._m_prod_full_py), dtype=jnp.float64)

        # Cumulative-product basis weights (Python ints / floats).
        self._basis_base = []  # prod(mods_base[0..i-1]), exact int (≤ ~2^47)
        for i in range(self.k_base):
            p = 1
            for k in range(i):
                p *= self.mods_base_py[k]
            self._basis_base.append(p)
        self._basis_full = []  # prod(mods_full[0..i-1]) as float
        for i in range(self.k_full):
            p = 1
            for k in range(i):
                p *= self.mods_full_py[k]
            self._basis_full.append(float(p))
        # basis_mod_ext[k][m] = prod(mods_base[0..k-1]) % mods_ext[m]
        self._basis_mod_ext = [
            [self._basis_base[k] % self.mods_ext_py[m] for m in range(len(self.mods_ext_py))]
            for k in range(self.k_base)
        ]
        self._m_prod_base_mod_ext = [
            self._m_prod_base_py % self.mods_ext_py[m] for m in range(len(self.mods_ext_py))
        ]

        self.base_extension = self._make_base_extension()
        self.garner_reconstruction = self._make_garner_reconstruction()

    # ── Constant-modulus reductions used by callers ───────────────────────────

    def to_base_residues(self, scaled_ints, axis):
        """Map scaled int64 values → base RNS residues, stacking moduli on `axis`."""
        mb = self.mods_base_py
        return jnp.stack([(scaled_ints % mb[k]).astype(jnp.uint8) for k in range(self.k_base)],
                         axis=axis)

    def reduce_full(self, result_i32):
        """Reduce a (k_full, ...) GEMM result by its per-channel modulus (axis 0)."""
        mf = self.mods_full_py
        return jnp.stack([result_i32[k] % mf[k] for k in range(self.k_full)], axis=0)

    # ── Factory: base extension (constant moduli) ─────────────────────────────

    def _make_base_extension(self):
        mb, me = self.mods_base_py, self.mods_ext_py
        kb, ke = self.k_base, len(me)
        gdiag, m_acc = _garner_constants(mb)
        basis_base   = self._basis_base
        basis_mod_ext = self._basis_mod_ext
        m_prod_base_half = self._m_prod_base_py // 2
        m_prod_base_mod_ext = self._m_prod_base_mod_ext

        @jax.jit
        def base_extension(regs_base):
            # regs_base: (k_base, ...) uint8  →  (k_full, ...) uint8
            ri = regs_base.astype(jnp.int32)
            v = [ri[0]]
            for i in range(1, kb):
                mi = mb[i]
                partial = v[0] % mi
                for idx, j in enumerate(range(1, i)):
                    partial = (partial + v[j] * m_acc[i][idx]) % mi
                v.append(((ri[i] - partial) * gdiag[i]) % mi)

            # Sign detection via exact int64 reconstruction of the base value.
            val = v[0].astype(jnp.int64) * basis_base[0]
            for k in range(1, kb):
                val = val + v[k].astype(jnp.int64) * basis_base[k]
            is_neg = (val > m_prod_base_half).astype(jnp.int32)

            ext = []
            for m in range(ke):
                acc = v[0] * basis_mod_ext[0][m]
                for k in range(1, kb):
                    acc = acc + v[k] * basis_mod_ext[k][m]
                ext.append(((acc - is_neg * m_prod_base_mod_ext[m]) % me[m]).astype(jnp.uint8))

            return jnp.concatenate([regs_base, jnp.stack(ext, axis=0)], axis=0)

        return base_extension

    # ── Factory: Garner reconstruction (constant moduli) ──────────────────────

    def _make_garner_reconstruction(self):
        mf = self.mods_full_py
        n  = self.k_full
        gdiag, m_acc = _garner_constants(mf)
        basis = self._basis_full
        m_prod = float(self._m_prod_full_py)  # float64 (exact int exceeds 2^64)

        @jax.jit
        def garner_reconstruction(residues):
            # residues: (k_full, ...) int  →  (...) float64
            ri = residues.astype(jnp.int32)
            x = [ri[0]]
            for i in range(1, n):
                mi = mf[i]
                partial = x[0] % mi
                for idx, j in enumerate(range(1, i)):
                    partial = (partial + x[j] * m_acc[i][idx]) % mi
                x.append(((ri[i] - partial) * gdiag[i]) % mi)

            result = x[0].astype(jnp.float64) * basis[0]
            for i in range(1, n):
                result = result + x[i].astype(jnp.float64) * basis[i]
            return jnp.where(result > m_prod / 2.0, result - m_prod, result)

        return garner_reconstruction

# ==============================================================================
# 3. THE OZAKI DERIVATIVE API
# ==============================================================================
class OzakiDerivative:
    """
    Drop-in replacement for SpatialDerivative.
    Executes exactly via RNS + INT8 GEMM (tensor cores) using Ozaki Scheme II.

    Dtype chain per block:
      float64 → int64 (scaled, range ~2^47) → uint8 (mod residues)
      → int8 (bias-shifted for signed INT8 GEMM) → int32 (GEMM accumulator)
      → int32 (mod-reduced result) → float64 (Garner weighted sum)
    """
    def __init__(self, block_size=64, halo=3, mods_ext=None):
        """Optional `mods_ext` overrides the extension moduli list — see
        CrtFloatConverter docstring for the dynamic-range bound."""
        self.bs = block_size
        self.ng = halo
        self.c = CrtFloatConverter(mods_ext=mods_ext)

        # 6th-Order Integer Coefficients
        self.c1  = [-1, 9, -45, 0, 45, -9, 1]
        self.c2  = [2, -27, 270, -490, 270, -27, 2]
        # KO dissipation: 6th-order central difference δ⁶ (damping with +σ).
        # NOT the negated [-1,6,-15,20,-15,6,-1] — that is anti-dissipative and
        # drives an exponential grid-scale instability.  See floating_point.CKO.
        self.cko = [1, -6, 15, -20, 15, -6, 1]

        self.D_d1_mods, self.D_d1_rsum = self._build_matrix(self.c1)
        self.D_d2_mods, self.D_d2_rsum = self._build_matrix(self.c2)
        self.D_ko_mods, self.D_ko_rsum = self._build_matrix(self.cko)

    def _build_matrix(self, coeffs):
        # Pad K (contraction dim) to a multiple of _K_MULTIPLE for INT8 tensor
        # cores.  Padded columns hold residue 0; the bias formula self-cancels
        # because K = D.shape[2] = K_pad.
        K_orig = self.bs + 2 * self.ng
        K_pad  = _pad_K(K_orig)
        D = np.zeros((self.bs, K_pad), dtype=np.int64)
        for r in range(self.bs):
            c_idx = r + self.ng
            D[r, c_idx-3 : c_idx+4] = coeffs

        D_mods_i8_list, row_sum_list = [], []
        for m in self.c.mods_full_list:
            D_m = (D % m).astype(np.int32)
            row_sum_list.append(D_m.sum(axis=1, dtype=np.int32))
            D_mods_i8_list.append((D_m - 128).astype(np.int8))

        return (
            jnp.stack([jnp.array(d) for d in D_mods_i8_list]),  # (k, bs, K_pad) int8
            jnp.stack([jnp.array(r) for r in row_sum_list]),     # (k, bs)        int32
        )

    def _ozaki_contract(self, block_in, D_mods_i8, D_rsum, axis):
        """Core math engine for a single block.

        Uses signed INT8 GEMM with asymmetric bias correction to map residues
        in [0, 252] into int8 [-128, 127] without changing the mod-m result:
            D @ r  =  (D-128) @ (r-128)  +  128*row_sum(D)  +  128*col_sum(r)  -  K*128^2
        where K is the contraction length. All terms fit comfortably in int32.
        """
        max_val = jnp.max(jnp.abs(block_in))
        m_half = self.c.M_prod_base / 2.0
        scale_exp = jnp.where(max_val == 0, 0.0, jnp.floor(jnp.log2(m_half / max_val)))
        exponent = -scale_exp.astype(jnp.int32)

        # Scale to integers; range is up to M_prod_base/2 ~ 2^47, requires int64
        scaled_block = block_in * (2.0 ** scale_exp)
        scaled_ints = jnp.floor(scaled_block + 0.5).astype(jnp.int64)

        # Reduce to uint8 residues: values [0, m-1] ≤ 252 fit in uint8
        regs_base = self.c.to_base_residues(scaled_ints, axis=0)

        regs_full = self.c.base_extension(regs_base)
        # regs_full: (k_full, patch_x, patch_y) uint8

        # Bias-shift into signed int8.  Keep as int8 for tensor-core GEMM, or
        # cast to int32 as a fallback for hardware/configs without INT8 support.
        regs_signed = (regs_full.astype(jnp.int32) - 128)
        if USE_INT8_GEMM:
            regs_signed = regs_signed.astype(jnp.int8)
            D_lhs = D_mods_i8
            accum = jnp.int32
        else:
            D_lhs = D_mods_i8.astype(jnp.int32)
            accum = None

        K = D_mods_i8.shape[2]                 # padded contraction length
        pad_amt = K - regs_signed.shape[1]     # H_orig = bs + 2*ng

        def pad_contract(arr, axis):
            if pad_amt == 0:
                return arr
            pw = [(0, 0)] * arr.ndim
            pw[axis] = (0, pad_amt)
            return jnp.pad(arr, pw, constant_values=-128)

        if axis == 0:
            # x-derivative: contract D(k,r,c) @ regs(k,c,y) → (k,r,y)
            gemm_raw = lax.dot_general(
                D_lhs, pad_contract(regs_signed, axis=1),
                dimension_numbers=(([2], [1]), ([0], [0])),
                preferred_element_type=accum,
            )
            col_sum_r = regs_full.astype(jnp.int32).sum(axis=1, keepdims=True)
            result_i32 = (gemm_raw
                          + 128 * D_rsum[:, :, None]
                          + 128 * col_sum_r
                          - K * 16384)
            res_mods = self.c.reduce_full(result_i32)[:, :, self.ng:-self.ng]
        else:
            # y-derivative: contract regs(k,x,c) @ D(k,r,c)^T → (k,x,r)
            gemm_raw = lax.dot_general(
                pad_contract(regs_signed, axis=2), D_lhs,
                dimension_numbers=(([2], [2]), ([0], [0])),
                preferred_element_type=accum,
            )
            row_sum_r = regs_full.astype(jnp.int32).sum(axis=2, keepdims=True)
            result_i32 = (gemm_raw
                          + 128 * D_rsum[:, None, :]
                          + 128 * row_sum_r
                          - K * 16384)
            res_mods = self.c.reduce_full(result_i32)[:, self.ng:-self.ng, :]

        out_int = self.c.garner_reconstruction(res_mods)
        return out_int * (2.0 ** exponent)

    def _extract_patches(self, u_pad, nbx, nby):
        """Extract overlapping patches via lax.dynamic_slice (no cuDNN dependency)."""
        patch_size = self.bs + 2 * self.ng
        # Build (n_patches, 2) int32 array of (block_row, block_col) start indices.
        ii, jj = jnp.meshgrid(jnp.arange(nbx, dtype=jnp.int32),
                               jnp.arange(nby, dtype=jnp.int32), indexing='ij')
        starts = jnp.stack([ii.ravel(), jj.ravel()], axis=1)  # (n_patches, 2)
        def extract_one(ij):
            return lax.dynamic_slice(u_pad, (ij[0] * self.bs, ij[1] * self.bs),
                                     (patch_size, patch_size))
        return vmap(extract_one)(starts)  # (n_patches, patch_size, patch_size)

    def _apply_pipeline(self, u, D_mods_i8, D_rsum, divisor, axis):
        """Handles grid patching, the Ozaki math, and unpatching."""
        nx, ny = u.shape

        # Pad to a multiple of block size
        pad_x = (self.bs - (nx % self.bs)) % self.bs
        pad_y = (self.bs - (ny % self.bs)) % self.bs
        u_aligned = jnp.pad(u, ((0, pad_x), (0, pad_y)), mode='edge')

        nbx, nby = u_aligned.shape[0] // self.bs, u_aligned.shape[1] // self.bs

        # Pad for stencil halo, then extract overlapping patches
        u_pad = jnp.pad(u_aligned, ((self.ng, self.ng), (self.ng, self.ng)), mode='edge')
        patches = self._extract_patches(u_pad, nbx, nby)

        wgmma_fn = lambda b: self._ozaki_contract(b, D_mods_i8, D_rsum, axis)
        computed_patches = vmap(wgmma_fn)(patches)

        computed_patches = computed_patches / divisor
        out_aligned = (computed_patches
                       .reshape(nbx, nby, self.bs, self.bs)
                       .transpose(0, 2, 1, 3)
                       .reshape(nbx * self.bs, nby * self.bs))
        return out_aligned[:nx, :ny]

    def _ozaki_contract_batched(self, blocks_in, D_mods_i8, D_rsum, axis):
        """Batched core engine: process F fields for a single spatial block at once.

        Packs all F fields into one INT8 batched GEMM call so XLA emits a single
        cuBLAS kernel instead of F separate ones. Bias correction is the same as
        the single-field case but broadcast over the field dimension.
        """
        # Per-field block-float scaling (independent max per field)
        def scale_one(field):
            max_val = jnp.max(jnp.abs(field))
            m_half = self.c.M_prod_base / 2.0
            scale_exp = jnp.where(max_val == 0, 0.0, jnp.floor(jnp.log2(m_half / max_val)))
            return field * (2.0 ** scale_exp), -scale_exp.astype(jnp.int32)

        scaled_blocks, exponents = vmap(scale_one)(blocks_in)
        # scaled_blocks: (F, patch_x, patch_y) float64 | exponents: (F,) int32

        scaled_ints = jnp.floor(scaled_blocks + 0.5).astype(jnp.int64)

        # RNS: reduce all fields to base residues (moduli on axis 1)
        regs_base = self.c.to_base_residues(scaled_ints, axis=1)
        # regs_base: (F, k_base, patch_x, patch_y) uint8

        regs_full = vmap(self.c.base_extension)(regs_base)
        # regs_full: (F, k_full, patch_x, patch_y) uint8

        regs_signed = (regs_full.astype(jnp.int32) - 128)
        if USE_INT8_GEMM:
            regs_signed = regs_signed.astype(jnp.int8)
            D_lhs = D_mods_i8
            accum = jnp.int32
        else:
            D_lhs = D_mods_i8.astype(jnp.int32)
            accum = None

        K = D_mods_i8.shape[2]                      # padded contraction length
        pad_amt = K - regs_signed.shape[2]          # H_orig = bs + 2*ng

        def pad_contract(arr, axis):
            if pad_amt == 0:
                return arr
            pw = [(0, 0)] * arr.ndim
            pw[axis] = (0, pad_amt)
            return jnp.pad(arr, pw, constant_values=-128)

        if axis == 0:
            # Batched x-derivative: D(k,r,c) @ regs(f,k,c,y)
            gemm_raw = lax.dot_general(
                D_lhs, pad_contract(regs_signed, axis=2),
                dimension_numbers=(([2], [2]), ([0], [1])),
                preferred_element_type=accum,
            )
            # col_sum_r: sum over x (contraction axis, dim 2 of regs_full)
            # (F, k, patch_y) → transpose → (k, F, patch_y) → (k, 1, F, patch_y)
            col_sum_r = regs_full.astype(jnp.int32).sum(axis=2).transpose(1, 0, 2)[:, None, :, :]
            result_i32 = (gemm_raw
                          + 128 * D_rsum[:, :, None, None]   # (k, bs, 1, 1)
                          + 128 * col_sum_r                   # (k, 1, F, patch_y)
                          - K * 16384)
            # Mod-reduce, crop y halo, rearrange to (F, k, bs, bs)
            res_mods = self.c.reduce_full(result_i32)
            res_mods = res_mods[:, :, :, self.ng:-self.ng].transpose(2, 0, 1, 3)
        else:
            # Batched y-derivative: regs(f,k,x,c) @ D(k,r,c)^T
            gemm_raw = lax.dot_general(
                pad_contract(regs_signed, axis=3), D_lhs,
                dimension_numbers=(([3], [2]), ([1], [0])),
                preferred_element_type=accum,
            )
            # row_sum_r: sum over y (contraction axis, dim 3 of regs_full)
            # (F, k, patch_x) → transpose → (k, F, patch_x) → (k, F, patch_x, 1)
            row_sum_r = regs_full.astype(jnp.int32).sum(axis=3).transpose(1, 0, 2)[:, :, :, None]
            result_i32 = (gemm_raw
                          + 128 * D_rsum[:, None, None, :]   # (k, 1, 1, bs)
                          + 128 * row_sum_r                   # (k, F, patch_x, 1)
                          - K * 16384)
            # Mod-reduce, crop x halo, rearrange to (F, k, bs, bs)
            res_mods = self.c.reduce_full(result_i32)
            res_mods = res_mods[:, :, self.ng:-self.ng, :].transpose(1, 0, 2, 3)

        # res_mods: (F, k_full, bs, bs) int32
        out_int = vmap(self.c.garner_reconstruction)(res_mods)   # (F, bs, bs) float64
        return out_int * (2.0 ** exponents)[:, None, None]

    def _apply_pipeline_batched(self, u_batch, D_mods_i8, D_rsum, divisor, axis):
        """Handles grid patching for F fields simultaneously."""
        F, nx, ny = u_batch.shape

        pad_x = (self.bs - (nx % self.bs)) % self.bs
        pad_y = (self.bs - (ny % self.bs)) % self.bs
        u_aligned = jnp.pad(u_batch, ((0, 0), (0, pad_x), (0, pad_y)), mode='edge')

        nbx = u_aligned.shape[1] // self.bs
        nby = u_aligned.shape[2] // self.bs

        u_pad = jnp.pad(u_aligned, ((0, 0), (self.ng, self.ng), (self.ng, self.ng)), mode='edge')

        # Extract overlapping patches for every field
        def extract_patches(u_f):
            return self._extract_patches(u_f, nbx, nby)

        # patches_all: (F, n_blocks, patch_size, patch_size)
        patches_all = vmap(extract_patches)(u_pad)
        # Pivot to (n_blocks, F, patch_size, patch_size) for block-parallel vmap
        patches_blocked = patches_all.transpose(1, 0, 2, 3)

        wgmma_fn = lambda blocks: self._ozaki_contract_batched(blocks, D_mods_i8, D_rsum, axis)
        # computed_patches: (n_blocks, F, bs, bs)
        computed_patches = vmap(wgmma_fn)(patches_blocked)

        computed_patches = computed_patches / divisor
        # Reassemble: (n_blocks, F, bs, bs) → (F, Nx_aligned, Ny_aligned)
        out = (computed_patches
               .reshape(nbx, nby, F, self.bs, self.bs)
               .transpose(2, 0, 3, 1, 4)
               .reshape(F, nbx * self.bs, nby * self.bs))
        return out[:, :nx, :ny]

    # ── Single-field API (drop-in for SpatialDerivative) ──────────────────────

    def compute_d1(self, u, dx, axis):
        return self._apply_pipeline(u, self.D_d1_mods, self.D_d1_rsum, 60.0 * dx, axis)

    def compute_d2(self, u, dx, axis):
        return self._apply_pipeline(u, self.D_d2_mods, self.D_d2_rsum, 180.0 * (dx**2), axis)

    def compute_ko(self, u, dx, sigma, axis):
        # Divisor matches floating_point.py: sigma/dx applied to coeff/64
        # → pipeline divisor = 64*dx/sigma
        return self._apply_pipeline(u, self.D_ko_mods, self.D_ko_rsum, 64.0 * dx / sigma, axis)

    # ── Batched multi-field API (all F fields in one GEMM call) ───────────────

    def compute_d1_batched(self, u_batch, dx, axis):
        """Compute first derivative for all F fields at once. u_batch: (F, Nx, Ny)."""
        return self._apply_pipeline_batched(u_batch, self.D_d1_mods, self.D_d1_rsum, 60.0 * dx, axis)

    def compute_d2_batched(self, u_batch, dx, axis):
        """Compute second derivative for all F fields at once. u_batch: (F, Nx, Ny)."""
        return self._apply_pipeline_batched(u_batch, self.D_d2_mods, self.D_d2_rsum, 180.0 * (dx**2), axis)

    def compute_ko_batched(self, u_batch, dx, sigma, axis):
        """Compute KO dissipation for all F fields at once. u_batch: (F, Nx, Ny)."""
        return self._apply_pipeline_batched(u_batch, self.D_ko_mods, self.D_ko_rsum, 64.0 * dx / sigma, axis)
