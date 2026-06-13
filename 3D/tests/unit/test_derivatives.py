"""
Unit-level audit of the 3D derivative operators and RHS (Phase 1, Step 1.1).

These lock the two things the BSSN RHS will later consume:

  1. The 3D `SpatialDerivative` stencils (C1, C2, CKO at order 6) match the
     validated 2D stencils bit-for-bit, and exactly reproduce a known polynomial
     / Fourier mode along EVERY axis (no axis is privileged).
  2. The 3D MCS RHS reduces EXACTLY to the validated 2D RHS on z-invariant data
     (the z-derivatives vanish, so every 3D equation must collapse onto its 2D
     analogue) -- the term-by-term 2D<->3D correspondence, machine-checked.

CPU-safe, fixed-shape, fast.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs3d.schemes.floating_point import SpatialDerivative as D3
from mcs2d.schemes.floating_point import SpatialDerivative as D2


# ── 1. Stencil coefficients identical to the validated 2D operator ────────────

class TestStencilCoefficients:
    def test_order6_matches_2d_bit_for_bit(self):
        """At order 6 (the production stencil) the 3D operator must be identical
        to the validated 2D operator.  (2D only implements order 6; the 3D
        order-4/8 generality is checked separately below.)"""
        a, b = D3(6), D2(6)
        assert np.array_equal(np.asarray(a.C1), np.asarray(b.C1)), "C1"
        assert np.array_equal(np.asarray(a.C2), np.asarray(b.C2)), "C2"
        assert np.array_equal(np.asarray(a.CKO), np.asarray(b.CKO)), "CKO"

    def test_extra_orders_are_textbook(self):
        """The 3D-only orders 4 and 8 (needed for 8th-order KO and BSSN) must
        carry the standard central-difference coefficients with the dissipative
        KO sign."""
        assert np.allclose(np.asarray(D3(4).C1), np.array([1, -8, 0, 8, -1]) / 12.0)
        assert np.allclose(np.asarray(D3(4).CKO), np.array([-1, 4, -6, 4, -1]) / 16.0)
        assert np.allclose(np.asarray(D3(8).C1),
                           np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0)
        assert np.allclose(np.asarray(D3(8).CKO),
                           np.array([-1, 8, -28, 56, -70, 56, -28, 8, -1]) / 256.0)

    def test_ko_sign_is_dissipative(self):
        """The load-bearing KO sign: the 6th-order kernel is [1,-6,15,-20,15,-6,1]/64
        (the negated form is anti-dissipative and destroys runs)."""
        cko = np.asarray(D3(6).CKO)
        assert np.allclose(cko, np.array([1, -6, 15, -20, 15, -6, 1]) / 64.0)


# ── 2. Derivatives are exact on a known mode along every axis ─────────────────

class TestDerivativeAccuracy:
    N = 32
    L = 2.0 * np.pi          # one full period over the grid

    def _grid_1d(self):
        return np.linspace(0.0, self.L, self.N, endpoint=False)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_d1_d2_on_sine(self, axis):
        """d/dx sin(kx) = k cos(kx); d2/dx2 = -k^2 sin(kx).  The 6th-order stencil
        must reproduce these to ~1e-4 at this resolution, on each axis."""
        d = D3(6)
        x = self._grid_1d(); dx = x[1] - x[0]; k = 1.0
        shape = [1, 1, 1]; shape[axis] = self.N
        f = np.sin(k * x).reshape(shape)
        f = np.broadcast_to(f, (self.N, self.N, self.N)).copy()
        d1 = np.asarray(d.compute_d1(jnp.asarray(f), dx, axis))
        d2 = np.asarray(d.compute_d2(jnp.asarray(f), dx, axis))
        cos = (k * np.cos(k * x)).reshape(shape) * np.ones((self.N, self.N, self.N))
        nsin = (-k**2 * np.sin(k * x)).reshape(shape) * np.ones((self.N, self.N, self.N))
        # interior only (edge padding is one-sided)
        sl = [slice(None)] * 3
        sl[axis] = slice(d.ng, -d.ng)
        sl = tuple(sl)
        assert np.max(np.abs(d1[sl] - cos[sl])) < 1e-4, f"d1 axis {axis}"
        assert np.max(np.abs(d2[sl] - nsin[sl])) < 1e-3, f"d2 axis {axis}"


# ── 3. The 3D RHS reduces to the validated 2D RHS on z-invariant data ─────────

class TestRHSReducesTo2D:
    """On data that is constant in z, every z-derivative vanishes and the 3D MCS
    RHS must collapse onto the 2D RHS field-by-field.  Comparing an arbitrary
    smooth z-invariant state against the 2D solver is the strongest term-by-term
    check that no coupling, sign, or axis was mis-transcribed in the port."""

    def _make_states(self, N=24):
        from mcs2d.main import MaxwellChernSimons2D, load_parameters as lp2
        from mcs3d.main import MaxwellChernSimons3D, load_parameters as lp3
        from mcs_common.wave_state import WaveState
        from pathlib import Path

        pf2 = str(Path(__import__("mcs2d").__file__).resolve().parents[2] / "params.toml")
        pf3 = str(Path(__import__("mcs3d").__file__).resolve().parents[2] / "params.toml")
        over = dict(scheme="floating_point", bc_type="periodic", Lambda=0.4,
                    ko_sigma=0.05, K1=1.0, K2=1.0, enable_cs=1.0)
        p2 = lp2(pf2); p2.update({"Nx": N, "Ny": N, **over})
        p3 = lp3(pf3); p3.update({"Nx": N, "Ny": N, "Nz": N, **over})
        dx = (p2["xmax"] - p2["xmin"]) / N
        sim2 = MaxwellChernSimons2D(dx, dx, 0.4, p2)
        sim3 = MaxwellChernSimons3D(dx, dx, dx, 0.4, p3)

        # An arbitrary smooth, spatially varying 10-field state on the 2D grid.
        rng = np.random.default_rng(0)
        x = np.asarray(sim2.x); y = np.asarray(sim2.y)
        X, Y = np.meshgrid(x, y, indexing="ij")
        d2 = np.zeros((10, sim2.Nx_tot, sim2.Ny_tot))
        for f in range(10):
            a, b = rng.uniform(0.5, 1.5, 2)
            ph = rng.uniform(0, np.pi, 2)
            d2[f] = 0.3 * np.sin(a * X * 2 * np.pi / 10 + ph[0]) \
                        * np.cos(b * Y * 2 * np.pi / 10 + ph[1])
        d2[7] += 1.0   # nonzero Pi background so CS terms are active

        # Embed as z-invariant in 3D (replicate the 2D slice along z).
        d3 = np.broadcast_to(d2[:, :, :, None],
                             (10, sim2.Nx_tot, sim2.Ny_tot, sim3.Nz_tot)).copy()
        return sim2, WaveState(jnp.asarray(d2)), sim3, WaveState(jnp.asarray(d3))

    def test_rhs_matches_2d_on_slice(self):
        sim2, st2, sim3, st3 = self._make_states()
        r2 = np.asarray(sim2.rhs(st2).data)              # (10, Nx, Ny)
        r3 = np.asarray(sim3.rhs(st3).data)              # (10, Nx, Ny, Nz)
        ng = sim2.ng
        # Compare interior; the 3D result must be z-invariant and equal the 2D RHS.
        a = r2[:, ng:-ng, ng:-ng]
        b = r3[:, ng:-ng, ng:-ng, ng:-ng]
        # z-invariance of the 3D RHS interior
        zspread = float(np.max(np.abs(b - b[:, :, :, :1])))
        assert zspread < 1e-12, f"3D RHS not z-invariant on z-invariant data: {zspread:.2e}"
        diff = float(np.max(np.abs(a - b[:, :, :, 0])))
        assert diff < 1e-12, f"3D RHS does not reduce to 2D RHS: max diff = {diff:.2e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
