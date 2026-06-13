"""
Correctness tests for OzakiDerivative against the floating-point (FP64) reference.

Run with pytest:           pytest 2D/tests/unit/test_ozaki.py -v
Or as a standalone script: python 2D/tests/unit/test_ozaki.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import reduce

from mcs2d.schemes.ozaki import OzakiDerivative, CrtFloatConverter
from mcs2d.schemes.floating_point import SpatialDerivative

# Tolerance for Ozaki vs FP64 comparison.
# Reconstruction error in float64 from large basis values is ~1e-9; 1e-7 gives safe margin.
ATOL = 1e-7


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fp():
    return SpatialDerivative(order=6)

@pytest.fixture(scope="module")
def oz():
    return OzakiDerivative(block_size=64, halo=3)


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth_2d(nx, ny, dx, dy, kx=1, ky=1):
    """sin(kx * 2π x / Lx) * cos(ky * 2π y / Ly) on [0, nx*dx] x [0, ny*dy]."""
    Lx, Ly = nx * dx, ny * dy
    x = jnp.arange(nx, dtype=jnp.float64) * dx
    y = jnp.arange(ny, dtype=jnp.float64) * dy
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    return jnp.sin(kx * 2.0 * jnp.pi * X / Lx) * jnp.cos(ky * 2.0 * jnp.pi * Y / Ly)

def linf(a, b):
    return float(jnp.max(jnp.abs(a - b)))

def l2(a, b):
    return float(jnp.sqrt(jnp.mean((a - b) ** 2)))


# ── Moduli sanity check ───────────────────────────────────────────────────────

class TestModuli:
    def test_product_satisfies_d2_safety_bound(self):
        """For exact RNS reconstruction with the C2 (D2) stencil, we need
        M_prod_full > 2 · ‖C2‖₁ · M_prod_base/2 = 1088 · M_prod_base.
        See CrtFloatConverter docstring for the derivation."""
        c = CrtFloatConverter()
        mods_base = c.mods_base_list
        mods_full = c.mods_base_list + c.mods_ext_list
        m_prod_base = reduce(lambda a, b: a * b, mods_base)
        m_prod_full = reduce(lambda a, b: a * b, mods_full)
        C2_L1_NORM = 2 + 27 + 270 + 490 + 270 + 27 + 2  # = 1088
        required = C2_L1_NORM * m_prod_base
        assert m_prod_full > required, (
            f"M_prod_full = {m_prod_full:.3e} fails the C2 dynamic-range bound "
            f"(needs > 1088 · M_prod_base = {required:.3e}).  Add an ext modulus."
        )

    def test_all_moduli_pairwise_coprime(self):
        """RNS requires that every pair of moduli is coprime."""
        from math import gcd
        c = CrtFloatConverter()
        mods = c.mods_base_list + c.mods_ext_list
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                assert gcd(mods[i], mods[j]) == 1, (
                    f"Moduli {mods[i]} and {mods[j]} are not coprime."
                )

    def test_base_product_fits_int64(self):
        """basis_base values used in sign detection must not overflow int64."""
        c = CrtFloatConverter()
        prod = reduce(lambda a, b: a * b, c.mods_base_list)
        assert prod < 2 ** 63, (
            f"M_prod_base = {prod:.3e} overflows int64 (max {2**63:.3e}). "
            "Move moduli from base to ext."
        )

    def test_all_moduli_fit_uint8(self):
        """All moduli must be <= 253 to fit residues in uint8."""
        c = CrtFloatConverter()
        mods = c.mods_base_list + c.mods_ext_list
        assert all(m <= 253 for m in mods), (
            f"Some moduli exceed 253: {[m for m in mods if m > 253]}"
        )


# ── D1: first derivative ──────────────────────────────────────────────────────

class TestD1:
    def test_axis0_single_block(self, fp, oz):
        """64x64 grid — fits in one block per axis."""
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d1(u, dx, axis=0),
                    fp.compute_d1(u, dx, axis=0)) < ATOL

    def test_axis1_single_block(self, fp, oz):
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d1(u, dy, axis=1),
                    fp.compute_d1(u, dy, axis=1)) < ATOL

    def test_axis0_multi_block(self, fp, oz):
        """128x128 grid — 4 blocks, tests block stitching."""
        nx, ny, dx, dy = 128, 128, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d1(u, dx, axis=0),
                    fp.compute_d1(u, dx, axis=0)) < ATOL

    def test_axis1_multi_block(self, fp, oz):
        nx, ny, dx, dy = 128, 128, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d1(u, dy, axis=1),
                    fp.compute_d1(u, dy, axis=1)) < ATOL

    def test_non_power_of_2_grid(self, fp, oz):
        """Grid that requires padding to nearest block multiple."""
        nx, ny, dx, dy = 100, 80, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d1(u, dx, axis=0),
                    fp.compute_d1(u, dx, axis=0)) < ATOL

    def test_high_wavenumber(self, fp, oz):
        """Higher-frequency field; stresses the scaling and reconstruction."""
        nx, ny, dx, dy = 128, 128, 0.05, 0.05
        u = smooth_2d(nx, ny, dx, dy, kx=4, ky=3)
        assert linf(oz.compute_d1(u, dx, axis=0),
                    fp.compute_d1(u, dx, axis=0)) < ATOL


# ── D2: second derivative ─────────────────────────────────────────────────────

class TestD2:
    def test_axis0_single_block(self, fp, oz):
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d2(u, dx, axis=0),
                    fp.compute_d2(u, dx, axis=0)) < ATOL

    def test_axis1_single_block(self, fp, oz):
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d2(u, dy, axis=1),
                    fp.compute_d2(u, dy, axis=1)) < ATOL

    def test_axis0_multi_block(self, fp, oz):
        nx, ny, dx, dy = 128, 128, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d2(u, dx, axis=0),
                    fp.compute_d2(u, dx, axis=0)) < ATOL

    def test_axis1_multi_block(self, fp, oz):
        nx, ny, dx, dy = 128, 128, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_d2(u, dy, axis=1),
                    fp.compute_d2(u, dy, axis=1)) < ATOL


# ── KO dissipation ────────────────────────────────────────────────────────────

class TestKO:
    def test_axis0(self, fp, oz):
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_ko(u, dx, sigma=0.05, axis=0),
                    fp.compute_ko(u, dx, sigma=0.05, axis=0)) < ATOL

    def test_axis1(self, fp, oz):
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        u = smooth_2d(nx, ny, dx, dy)
        assert linf(oz.compute_ko(u, dy, sigma=0.05, axis=1),
                    fp.compute_ko(u, dy, sigma=0.05, axis=1)) < ATOL


# ── Batched API ───────────────────────────────────────────────────────────────

class TestBatched:
    """Batched API must give the same result as calling single-field API per field."""

    def _batch(self, nx=64, ny=64, dx=0.1, dy=0.1, F=5):
        return jnp.stack([smooth_2d(nx, ny, dx, dy, kx=k+1, ky=k+1) for k in range(F)])

    def test_d1_batched_matches_single_axis0(self, oz):
        u = self._batch()
        batched = oz.compute_d1_batched(u, 0.1, axis=0)
        single = jnp.stack([oz.compute_d1(u[i], 0.1, axis=0) for i in range(u.shape[0])])
        np.testing.assert_allclose(np.array(batched), np.array(single), atol=1e-12)

    def test_d1_batched_matches_single_axis1(self, oz):
        u = self._batch()
        batched = oz.compute_d1_batched(u, 0.1, axis=1)
        single = jnp.stack([oz.compute_d1(u[i], 0.1, axis=1) for i in range(u.shape[0])])
        np.testing.assert_allclose(np.array(batched), np.array(single), atol=1e-12)

    def test_d2_batched_matches_single_axis0(self, oz):
        u = self._batch()
        batched = oz.compute_d2_batched(u, 0.1, axis=0)
        single = jnp.stack([oz.compute_d2(u[i], 0.1, axis=0) for i in range(u.shape[0])])
        np.testing.assert_allclose(np.array(batched), np.array(single), atol=1e-12)

    def test_d2_batched_matches_single_axis1(self, oz):
        u = self._batch()
        batched = oz.compute_d2_batched(u, 0.1, axis=1)
        single = jnp.stack([oz.compute_d2(u[i], 0.1, axis=1) for i in range(u.shape[0])])
        np.testing.assert_allclose(np.array(batched), np.array(single), atol=1e-12)

    def test_d1_batched_matches_fp(self, fp, oz):
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        F = 3
        u = jnp.stack([smooth_2d(nx, ny, dx, dy, kx=k+1, ky=k+1) for k in range(F)])
        batched = oz.compute_d1_batched(u, dx, axis=0)
        expected = jnp.stack([fp.compute_d1(u[i], dx, axis=0) for i in range(F)])
        np.testing.assert_allclose(np.array(batched), np.array(expected), atol=ATOL)

    def test_d2_batched_matches_fp(self, fp, oz):
        nx, ny, dx, dy = 64, 64, 0.1, 0.1
        F = 3
        u = jnp.stack([smooth_2d(nx, ny, dx, dy, kx=k+1, ky=k+1) for k in range(F)])
        batched = oz.compute_d2_batched(u, dx, axis=0)
        expected = jnp.stack([fp.compute_d2(u[i], dx, axis=0) for i in range(F)])
        np.testing.assert_allclose(np.array(batched), np.array(expected), atol=ATOL)


# ── Analytical accuracy ───────────────────────────────────────────────────────

class TestAnalytical:
    """Compare against known analytical derivatives; skip boundary ghost zones."""

    NG = 3  # ghost points to exclude from comparison

    def test_d1_axis0(self, oz):
        nx, ny = 128, 128
        dx = dy = 10.0 / nx
        k = 2.0 * jnp.pi / (nx * dx)
        x = jnp.arange(nx, dtype=jnp.float64) * dx
        y = jnp.arange(ny, dtype=jnp.float64) * dy
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        u = jnp.sin(k * X)

        got = oz.compute_d1(u, dx, axis=0)
        exact = k * jnp.cos(k * X)
        ng = self.NG
        err = float(jnp.max(jnp.abs(got[ng:-ng, ng:-ng] - exact[ng:-ng, ng:-ng])))
        assert err < 1e-7, f"D1 axis=0 vs analytical: max_err={err:.2e}"

    def test_d1_axis1(self, oz):
        nx, ny = 128, 128
        dx = dy = 10.0 / nx
        k = 2.0 * jnp.pi / (ny * dy)
        x = jnp.arange(nx, dtype=jnp.float64) * dx
        y = jnp.arange(ny, dtype=jnp.float64) * dy
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        u = jnp.cos(k * Y)

        got = oz.compute_d1(u, dy, axis=1)
        exact = -k * jnp.sin(k * Y)
        ng = self.NG
        err = float(jnp.max(jnp.abs(got[ng:-ng, ng:-ng] - exact[ng:-ng, ng:-ng])))
        assert err < 1e-7, f"D1 axis=1 vs analytical: max_err={err:.2e}"

    def test_d2_axis0(self, oz):
        nx, ny = 128, 128
        dx = dy = 10.0 / nx
        k = 2.0 * jnp.pi / (nx * dx)
        x = jnp.arange(nx, dtype=jnp.float64) * dx
        y = jnp.arange(ny, dtype=jnp.float64) * dy
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        u = jnp.sin(k * X)

        got = oz.compute_d2(u, dx, axis=0)
        exact = -(k ** 2) * jnp.sin(k * X)
        ng = self.NG
        err = float(jnp.max(jnp.abs(got[ng:-ng, ng:-ng] - exact[ng:-ng, ng:-ng])))
        assert err < 1e-7, f"D2 axis=0 vs analytical: max_err={err:.2e}"

    def test_d2_axis1(self, oz):
        nx, ny = 128, 128
        dx = dy = 10.0 / nx
        k = 2.0 * jnp.pi / (ny * dy)
        x = jnp.arange(nx, dtype=jnp.float64) * dx
        y = jnp.arange(ny, dtype=jnp.float64) * dy
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        u = jnp.cos(k * Y)

        got = oz.compute_d2(u, dy, axis=1)
        exact = -(k ** 2) * jnp.cos(k * Y)
        ng = self.NG
        err = float(jnp.max(jnp.abs(got[ng:-ng, ng:-ng] - exact[ng:-ng, ng:-ng])))
        assert err < 1e-7, f"D2 axis=1 vs analytical: max_err={err:.2e}"


# ── Standalone smoke test ─────────────────────────────────────────────────────

def _smoke():
    from functools import reduce as _reduce
    print("=" * 60)
    print("Ozaki Derivative Smoke Test")
    print("=" * 60)

    c = CrtFloatConverter()
    mods = c.mods_base_list + c.mods_ext_list
    prod = _reduce(lambda a, b: a * b, mods)
    print(f"Moduli:       {mods}")
    print(f"M_prod_full:  {prod:.3e}  (2^{prod.bit_length()-1})")
    print(f"Need > 2^64:  {prod > 2**64}")
    print()

    fp = SpatialDerivative(order=6)
    oz = OzakiDerivative(block_size=64, halo=3)

    cases = [
        ("D1 axis=0, 64x64",   lambda u, d: oz.compute_d1(u, d, axis=0),
                                lambda u, d: fp.compute_d1(u, d, axis=0)),
        ("D1 axis=1, 64x64",   lambda u, d: oz.compute_d1(u, d, axis=1),
                                lambda u, d: fp.compute_d1(u, d, axis=1)),
        ("D2 axis=0, 64x64",   lambda u, d: oz.compute_d2(u, d, axis=0),
                                lambda u, d: fp.compute_d2(u, d, axis=0)),
        ("D2 axis=1, 64x64",   lambda u, d: oz.compute_d2(u, d, axis=1),
                                lambda u, d: fp.compute_d2(u, d, axis=1)),
        ("KO axis=0, 64x64",   lambda u, d: oz.compute_ko(u, d, 0.05, axis=0),
                                lambda u, d: fp.compute_ko(u, d, 0.05, axis=0)),
        ("D1 axis=0, 128x128", lambda u, d: oz.compute_d1(u, d, axis=0),
                                lambda u, d: fp.compute_d1(u, d, axis=0)),
    ]

    all_pass = True
    nx, ny, dx = 64, 64, 0.1

    for label, oz_fn, fp_fn in cases:
        if "128" in label:
            nx, ny = 128, 128
        else:
            nx, ny = 64, 64
        u = smooth_2d(nx, ny, dx, dx)
        oz_r = oz_fn(u, dx)
        fp_r = fp_fn(u, dx)
        err_max = float(jnp.max(jnp.abs(oz_r - fp_r)))
        err_l2  = float(jnp.sqrt(jnp.mean((oz_r - fp_r) ** 2)))
        ok = err_max < ATOL
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:<30}  max={err_max:.2e}  l2={err_l2:.2e}")

    print()
    if all_pass:
        print("All checks passed.")
    else:
        print("SOME CHECKS FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    _smoke()
