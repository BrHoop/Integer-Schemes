import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np
from functools import reduce

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. BFP48 COMPRESSION & RECONSTRUCTION
# ==============================================================================
@jax.jit
def float64_to_bfp48(block_3D):
    max_val = jnp.max(jnp.abs(block_3D))
    scale_exp = jnp.where(max_val == 0, 0.0, jnp.floor(jnp.log2((2.0**47 - 1) / max_val)))
    exponent = scale_exp.astype(jnp.int32)
    scaled = block_3D * (2.0 ** scale_exp)
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
# 2. RNS & CRT CORE MATH (Updated for 3D Tensors)
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

@jax.jit
def base_extension(regs_in, mods_in, mods_out, garner_consts, basis_mod_out, basis_in, m_prod_in, m_prod_in_mod_out):
    num_in = mods_in.shape[0]
    v = jnp.zeros_like(regs_in)
    v = v.at[0].set(regs_in[0])
    for i in range(1, num_in):
        partial_sum = v[0] % mods_in[i]
        m_acc = 1
        for j in range(1, i):
            m_acc = (m_acc * mods_in[j-1]) % mods_in[i]
            term = (v[j] * m_acc) % mods_in[i]
            partial_sum = (partial_sum + term) % mods_in[i]
        diff = (regs_in[i] - partial_sum) % mods_in[i]
        v_i = (diff * garner_consts[i, i]) % mods_in[i]
        v = v.at[i].set(v_i)
        
    # 3D Tensor contraction
    regs_ext = jnp.einsum('kxyz,km->mxyz', v, basis_mod_out)
    val_reconstructed = jnp.sum(v * basis_in.reshape(-1, 1, 1, 1), axis=0)
    
    is_negative = val_reconstructed > (m_prod_in // 2)
    correction = is_negative[None, :, :, :] * m_prod_in_mod_out[:, None, None, None]
    regs_ext = (regs_ext - correction) % mods_out[:, None, None, None]
    
    return jnp.concatenate([regs_in, regs_ext], axis=0)

@jax.jit
def garner_reconstruction(residues, mods, garner_consts, basis, m_prod):
    num_mods = mods.shape[0]
    x = jnp.zeros_like(residues)
    x = x.at[0].set(residues[0])
    for i in range(1, num_mods):
        partial_sum = x[0] % mods[i]
        m_acc = 1
        for j in range(1, i):
            m_acc = (m_acc * mods[j-1]) % mods[i]
            term = (x[j] * m_acc) % mods[i]
            partial_sum = (partial_sum + term) % mods[i]
        diff = (residues[i] - partial_sum) % mods[i]
        v_i = (diff * garner_consts[i, i]) % mods[i]
        x = x.at[i].set(v_i)
        
    # 3D Tensor summation
    result = jnp.sum(x * basis.reshape(-1, 1, 1, 1), axis=0)
    m_half = m_prod / 2.0
    result = jnp.where(result > m_half, result - m_prod, result)
    return result

class CrtFloatConverter:
    # (Unchanged from original)
    def __init__(self):
        self.mods_base_list = [253, 251, 249, 247, 245, 241]
        self.mods_ext_list = [239]
        self.mods_full_list = self.mods_base_list + self.mods_ext_list
        
        self.k_full = len(self.mods_full_list)
        self.k_base = len(self.mods_base_list)

        mods_base_np = np.array(self.mods_base_list, dtype=np.int64)
        mods_ext_np  = np.array(self.mods_ext_list, dtype=np.int64)
        mods_full_np = np.array(self.mods_full_list, dtype=np.int64)
        
        m_prod_base_py = reduce(lambda x, y: x * y, self.mods_base_list)
        m_prod_full_py = reduce(lambda x, y: x * y, self.mods_full_list)
        
        garner_consts = np.zeros((self.k_full, self.k_full), dtype=np.int64)
        for i in range(self.k_full):
            if i == 0: p_val = 1
            else:
                p_val = 1
                for k in range(i): p_val *= int(self.mods_full_list[k])
            for j in range(i, self.k_full):
                inv_p = modinv(p_val, int(self.mods_full_list[j]))
                garner_consts[i, j] = inv_p

        basis_full = np.zeros(self.k_full, dtype=np.float64)
        for i in range(self.k_full):
            prod_val = 1
            for k in range(i): prod_val *= int(self.mods_full_list[k])
            basis_full[i] = float(prod_val)
            
        basis_base = np.zeros(self.k_base, dtype=np.int64)
        for i in range(self.k_base):
            prod_val = 1
            for k in range(i): prod_val *= int(self.mods_base_list[k])
            basis_base[i] = int(prod_val)

        basis_mod_ext = np.zeros((self.k_base, len(mods_ext_np)), dtype=np.int64)
        for i in range(self.k_base):
            weight = 1
            for k in range(i): weight *= int(self.mods_base_list[k])
            for j in range(len(mods_ext_np)):
                basis_mod_ext[i, j] = weight % int(mods_ext_np[j])
            
        m_prod_base_mod_ext = np.zeros(len(mods_ext_np), dtype=np.int64)
        for j in range(len(mods_ext_np)):
            m_prod_base_mod_ext[j] = m_prod_base_py % int(mods_ext_np[j])

        self.mods_base = jnp.array(mods_base_np)
        self.mods_ext  = jnp.array(mods_ext_np)
        self.mods_full = jnp.array(mods_full_np)
        self.garner_consts = jnp.array(garner_consts)
        self.basis_mod_ext = jnp.array(basis_mod_ext)
        self.basis_full = jnp.array(basis_full)
        self.basis_base = jnp.array(basis_base)
        self.M_prod_full = jnp.array(float(m_prod_full_py), dtype=jnp.float64)
        self.M_prod_base = jnp.array(m_prod_base_py, dtype=jnp.int64)
        self.M_prod_base_mod_ext = jnp.array(m_prod_base_mod_ext, dtype=jnp.int64)


# ==============================================================================
# 3. THE 3D OZAKI DERIVATIVE API
# ==============================================================================
class OzakiDerivative3D:
    """
    Executes exactly via RNS math using Ozaki Scheme II over a 3D Grid.
    """
    def __init__(self, block_size=16, halo=4):
        # 16^3 is highly recommended for 3D to prevent VRAM explosion
        self.bs = block_size
        self.ng = halo  
        self.c = CrtFloatConverter()
        
        # 8th-Order Integer Coefficients
        self.c1  = [3, -32, 168, -672, 0, 672, -168, 32, -3]
        self.c2  = [-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]
        self.cko = [-1, 8, -28, 56, -70, 56, -28, 8, -1]

        self.D_d1_mods = self._build_matrix(self.c1)
        self.D_d2_mods = self._build_matrix(self.c2)
        self.D_ko_mods = self._build_matrix(self.cko)

    def _build_matrix(self, coeffs):
        D = np.zeros((self.bs, self.bs + 2 * self.ng), dtype=np.int64)
        for r in range(self.bs):
            c_idx = r + self.ng
            D[r, c_idx-4 : c_idx+5] = coeffs
        mods_np = np.array(self.c.mods_full_list)
        return jnp.stack([jnp.array(D % m) for m in mods_np])

    def _ozaki_contract(self, block_in, D_mods, axis):
        """Core math engine for a single 3D block."""
        max_val = jnp.max(jnp.abs(block_in))
        m_half = self.c.M_prod_base / 2.0
        scale_exp = jnp.where(max_val == 0, 0.0, jnp.floor(jnp.log2(m_half / max_val)))
        exponent = -scale_exp.astype(jnp.int32)
        
        scaled_block = block_in * (2.0 ** scale_exp)
        scaled_ints = jnp.floor(scaled_block + 0.5).astype(jnp.int64) 
        regs_base = scaled_ints[None, :, :, :] % self.c.mods_base[:, None, None, None]
        
        regs_full = base_extension(regs_base, self.c.mods_base, self.c.mods_ext, 
                                   self.c.garner_consts, self.c.basis_mod_ext, 
                                   self.c.basis_base, self.c.M_prod_base, 
                                   self.c.M_prod_base_mod_ext)

        # 3D Orthogonal Contraction Logic
        if axis == 0:
            res_raw = jnp.einsum('krc,kcyz->kryz', D_mods, regs_full)[:, :, self.ng:-self.ng, self.ng:-self.ng]
        elif axis == 1:
            res_raw = jnp.einsum('krc,kxcz->kxrz', D_mods, regs_full)[:, self.ng:-self.ng, :, self.ng:-self.ng]
        elif axis == 2:
            res_raw = jnp.einsum('krc,kxyc->kxyr', D_mods, regs_full)[:, self.ng:-self.ng, self.ng:-self.ng, :]
        else:
            raise ValueError("Axis must be 0, 1, or 2 for 3D derivatives.")
            
        res_mods = res_raw % self.c.mods_full[:, None, None, None]

        out_int = garner_reconstruction(res_mods, self.c.mods_full, self.c.garner_consts, 
                                        self.c.basis_full, self.c.M_prod_full)
        return out_int * (2.0 ** exponent)

    def _extract_patches(self, u_pad, nbx, nby, nbz):
        """Extract overlapping 3D patches via lax.dynamic_slice (no cuDNN dependency)."""
        patch_size = self.bs + 2 * self.ng
        ii, jj, kk = jnp.meshgrid(
            jnp.arange(nbx, dtype=jnp.int32),
            jnp.arange(nby, dtype=jnp.int32),
            jnp.arange(nbz, dtype=jnp.int32),
            indexing='ij'
        )
        starts = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)  # (n_patches, 3)
        def extract_one(ijk):
            return lax.dynamic_slice(
                u_pad,
                (ijk[0] * self.bs, ijk[1] * self.bs, ijk[2] * self.bs),
                (patch_size, patch_size, patch_size)
            )
        return vmap(extract_one)(starts)  # (n_patches, patch_size, patch_size, patch_size)

    def _apply_pipeline(self, u, D_mods, divisor, axis):
        """Handles 3D grid patching, compiling the Ozaki math, and unpatching."""
        nx, ny, nz = u.shape

        # 0. Pad up to the nearest multiple of the block size (bs)
        pad_x = (self.bs - (nx % self.bs)) % self.bs
        pad_y = (self.bs - (ny % self.bs)) % self.bs
        pad_z = (self.bs - (nz % self.bs)) % self.bs
        u_aligned = jnp.pad(u, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='edge')

        nbx = u_aligned.shape[0] // self.bs
        nby = u_aligned.shape[1] // self.bs
        nbz = u_aligned.shape[2] // self.bs

        # 1. Pad for the stencil halo in 3D
        u_pad = jnp.pad(u_aligned, ((self.ng, self.ng), (self.ng, self.ng), (self.ng, self.ng)), mode='edge')

        # 2. Extract 3D patches via lax.dynamic_slice (CUDA-graph-compatible, no cuDNN)
        patches = self._extract_patches(u_pad, nbx, nby, nbz)

        # 3. Map the Ozaki kernel over all blocks
        wgmma_fn = lambda b: self._ozaki_contract(b, D_mods, axis)
        computed_patches = vmap(wgmma_fn)(patches)

        # 4. Scale, Rebuild, and Crop back to original shape
        computed_patches = computed_patches / divisor
        
        # Reshape to (bx, by, bz, px, py, pz) and transpose axes to interleave
        out_aligned = computed_patches.reshape(nbx, nby, nbz, self.bs, self.bs, self.bs)
        out_aligned = out_aligned.transpose(0, 3, 1, 4, 2, 5).reshape(nbx * self.bs, nby * self.bs, nbz * self.bs)
        
        # Slice off the block alignment padding
        return out_aligned[:nx, :ny, :nz]

    # API Hooks
    def compute_d1(self, u, dx, axis):
        return self._apply_pipeline(u, self.D_d1_mods, 840.0 * dx, axis)

    def compute_d2(self, u, dx, axis):
        return self._apply_pipeline(u, self.D_d2_mods, 5040.0 * (dx**2), axis)

    def compute_ko(self, u, dx, sigma, axis):
        # Divisor matches floating_point.py: CKO/256 * sigma/dx → pipeline divisor = 256*dx/sigma
        return self._apply_pipeline(u, self.D_ko_mods, 256.0 * dx / sigma, axis)