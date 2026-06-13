"""
Spectral / eigenvalue certificates for the MCS 3D discretization (Phase 1).

The 3D mirror of `2D/tests/validation/test_spectrum.py`.  Every test is
post-processing of the linearized semi-discrete symbol M(kx, ky, kz) built in
`mcs3d.validate`, cross-checked against the exact AD Jacobian of the real RHS:

  * symbol <-> AD-Jacobian agreement (the foundation)  -> TestSymbolMatchesJacobian
  * no growing modes (semi-discrete stability)         -> TestSemiDiscreteStability
  * fully-discrete von Neumann stability at the CFL    -> TestRK4Amplification
  * correct wave propagation (dispersion is 6th-order) -> TestDispersion
  * the exact Kreiss-Oliger dissipation kernel         -> TestKOKernel
  * the physical CFJ tachyon is present & located      -> TestModeStructure
  * constraint damping tracks K                        -> TestConstraintDampingModes
  * strong hyperbolicity (well-posedness)              -> TestStrongHyperbolicity

All CPU-safe: small dense eigensolves and one small AD Jacobian.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from mcs3d.validate import (
    symbol_params, _symbol, grid_wavenumbers, load_parameters,
    ko_symbol_1d, rk4_amplification_radius, em_wave_omega,
    mode_classification, principal_symbol_condition, _build_jacobian,
    _PARAMS_FILE, NF,
)


def _params(lam, **over):
    p = load_parameters(_PARAMS_FILE)
    p.update({"Lambda": lam, "ko_sigma": 0.05, "K1": 1.0, "K2": 1.0, **over})
    return p


def _grid_symbol_params(lam, N=10, **over):
    """symbol_params on an NxNxN grid, plus the discrete grid wavenumbers.
    N=10 keeps the triple loop (N^3 symbol evals) cheap."""
    p = _params(lam, **over)
    dx = (p["xmax"] - p["xmin"]) / N
    dy = (p["ymax"] - p["ymin"]) / N
    dz = (p["zmax"] - p["zmin"]) / N
    sp = symbol_params(p, dx, dy, dz)
    ks = grid_wavenumbers(N, p["xmax"] - p["xmin"])
    return sp, ks, p, dx


# ── 0. Symbol <-> Jacobian consistency (foundation) ───────────────────────────

class TestSymbolMatchesJacobian:
    """The analytic symbol must reproduce the eigenvalues of the EXACT AD Jacobian
    of the real 3D RHS.  This is the decisive correctness gate -- if it passes,
    every other spectral test is trustworthy; if it fails, the symbol is wrong and
    nothing downstream means anything."""

    def test_eigenvalue_multisets_agree(self):
        N = 6
        J, sim, params = _build_jacobian(N, N, N, ko_sigma=0.05, lam=0.4, background="pi0")
        eig_J = np.linalg.eigvals(J)

        sp = symbol_params(params, sim.dx, sim.dy, sim.dz)
        ks = grid_wavenumbers(N, params["xmax"] - params["xmin"])
        eig_S = []
        for kx in ks:
            for ky in ks:
                for kz in ks:
                    eig_S.extend(np.linalg.eigvals(_symbol(kx, ky, kz, sp, with_ko=True)))
        eig_S = np.array(eig_S)

        remaining = list(eig_S)
        worst = 0.0
        for e in eig_J:
            d = np.abs(e - np.array(remaining))
            j = int(np.argmin(d))
            worst = max(worst, float(d[j]))
            remaining.pop(j)
        assert worst < 1e-6, f"symbol vs AD-Jacobian eigenvalue mismatch: {worst:.2e}"


# ── 1. Semi-discrete stability ────────────────────────────────────────────────

class TestSemiDiscreteStability:
    LAM_STABLE = 0.2     # m_cs = 0.4 < smallest grid |k|

    def test_no_growing_modes(self):
        """In the CFJ-stable regime every semi-discrete eigenvalue must have
        Re(lambda) <= 0 (to round-off): no mode grows in time."""
        sp, ks, _, _ = _grid_symbol_params(self.LAM_STABLE)
        max_re = max(np.max(np.linalg.eigvals(_symbol(kx, ky, kz, sp, with_ko=True)).real)
                     for kx in ks for ky in ks for kz in ks)
        assert max_re < 1e-9, f"max Re(lambda) = {max_re:.2e} > 0 (growing mode)"

    def test_ko_only_damps(self):
        """KO must move eigenvalues LEFT (a scalar on the diagonal).  Verify the
        shift is non-positive everywhere -- the load-bearing anti-dissipation
        guard."""
        sp, ks, _, _ = _grid_symbol_params(self.LAM_STABLE)
        worst_increase = -np.inf
        for kx in ks:
            for ky in ks:
                for kz in ks:
                    e_ko = np.sort(np.linalg.eigvals(_symbol(kx, ky, kz, sp, with_ko=True)).real)
                    e_no = np.sort(np.linalg.eigvals(_symbol(kx, ky, kz, sp, with_ko=False)).real)
                    worst_increase = max(worst_increase, float(np.max(e_ko - e_no)))
        assert worst_increase < 1e-9, (
            f"KO increased a Re(lambda) by {worst_increase:.2e} (anti-dissipative!)")


# ── 2. Fully-discrete (von Neumann) stability at the CFL ───────────────────────

class TestRK4Amplification:
    LAM_STABLE = 0.2

    def test_amplification_within_unity(self):
        """The RK4 one-step amplification factor |G(k)| must be <= 1 (to round-off)
        for every grid wavenumber at the operating CFL."""
        sp, ks, p, dx = _grid_symbol_params(self.LAM_STABLE)
        dt = p.get("cfl", 0.02) * dx
        worst = max(rk4_amplification_radius(_symbol(kx, ky, kz, sp, with_ko=True), dt)
                    for kx in ks for ky in ks for kz in ks)
        assert worst < 1.0 + 1e-9, f"max |G(k)| = {worst:.6f} > 1 (CFL too large)"


# ── 3. Dispersion accuracy (6th order) ────────────────────────────────────────

class TestDispersion:
    LAM = 0.2

    def test_dispersion_is_sixth_order(self):
        """The FD wave frequency omega_FD(k) must approach the continuum MCS
        frequency sqrt(k^2 + m_cs k) at 6th order: halving dx cuts the error by
        ~2^6 = 64."""
        p = _params(self.LAM)
        k = 2.0
        m_cs = 2.0 * self.LAM
        omega_exact = np.sqrt(k**2 + m_cs * k)

        def err(dx):
            sp = symbol_params(p, dx, dx, dx)
            return abs(em_wave_omega(k, sp, fd=True) - omega_exact)

        e1, e2 = err(0.05), err(0.025)
        ratio = e1 / e2
        assert 40 < ratio < 90, (
            f"dispersion error ratio under dx-halving = {ratio:.1f} "
            f"(expected ~64 for 6th order); e1={e1:.2e}, e2={e2:.2e}")

    def test_group_velocity_accurate_at_low_k(self):
        """Group velocity v_g = d(omega)/dk must match the continuum value to high
        accuracy at well-resolved wavenumbers."""
        p = _params(self.LAM)
        dx = 0.05
        sp = symbol_params(p, dx, dx, dx)
        m_cs = 2.0 * self.LAM
        k = 1.0
        h = 1e-4
        vg_fd = (em_wave_omega(k + h, sp, fd=True) - em_wave_omega(k - h, sp, fd=True)) / (2 * h)
        omega = np.sqrt(k**2 + m_cs * k)
        vg_exact = (2 * k + m_cs) / (2 * omega)
        rel = abs(vg_fd - vg_exact) / abs(vg_exact)
        assert rel < 1e-3, f"group-velocity rel error = {rel:.2e} at k=1, dx=0.05"


# ── 4. Kreiss-Oliger kernel verification (against the real stencil) ────────────

class TestKOKernel:
    def test_stencil_symbol_is_sin6(self):
        """Apply the SOLVER's actual KO operator to a pure Fourier mode and
        confirm its eigenvalue equals -(sigma/dx) sin^6(k dx/2) to round-off, on
        each axis.  Catches a wrong CKO coefficient or a sign flip."""
        from mcs3d.main import MaxwellChernSimons3D, load_parameters
        p = load_parameters(_PARAMS_FILE)
        N = 16
        p.update({"scheme": "floating_point", "Nx": N, "Ny": N, "Nz": N,
                  "bc_type": "periodic"})
        dx = (p["xmax"] - p["xmin"]) / N
        sim = MaxwellChernSimons3D(dx, dx, dx, p["Lambda"], p)
        sigma = 0.05
        coords = (np.asarray(sim.x), np.asarray(sim.y), np.asarray(sim.z))
        for axis in (0, 1, 2):
            c = coords[axis]
            for m in (1, 3, 7):
                k = 2 * np.pi * m / (p["xmax"] - p["xmin"])
                shape = [1, 1, 1]; shape[axis] = c.size
                line = np.exp(1j * k * c).reshape(shape)
                mode = np.broadcast_to(line, (sim.Nx_tot, sim.Ny_tot, sim.Nz_tot))
                ko = (np.asarray(sim.diff_op.compute_ko(mode.real, dx, sigma, axis))
                      + 1j * np.asarray(sim.diff_op.compute_ko(mode.imag, dx, sigma, axis)))
                expected = ko_symbol_1d(k, dx, sigma)
                # interior sample along this axis
                sl = [sim.ng, sim.ng, sim.ng]; sl[axis] = slice(sim.ng, -sim.ng)
                ratio = ko[tuple(sl)] / mode[tuple(sl)]
                err = float(np.max(np.abs(ratio - expected)))
                assert err < 1e-10, f"axis {axis} m={m}: KO symbol err {err:.2e}"
                assert expected <= 0, f"KO symbol positive (anti-dissipative) m={m}"


# ── 5. Mode structure & the CFJ tachyon ───────────────────────────────────────

class TestModeStructure:
    """The 10-field symbol must have the right physical mode content, and the
    Carroll-Field-Jackiw tachyon must appear exactly where the dispersion relation
    predicts (|k| < m_cs).  Uses Lambda = 0.4 (m_cs = 0.8)."""
    LAM = 0.4

    def test_ten_modes_no_ko(self):
        p = _params(self.LAM, ko_sigma=0.0)
        sp = symbol_params(p, 0.3125, 0.3125, 0.3125)
        cls = mode_classification(2.0, 2.0, 2.0, sp)
        total = cls["oscillatory"] + cls["damped"] + cls["growing"]
        assert total == NF, f"mode count {total} != {NF}"

    def test_cfj_tachyon_below_threshold(self):
        p = _params(self.LAM, ko_sigma=0.0)
        sp = symbol_params(p, 0.3125, 0.3125, 0.3125)
        m_cs = 2.0 * self.LAM
        below = mode_classification(0.5 * m_cs, 0.0, 0.0, sp)
        above = mode_classification(3.0 * m_cs, 0.0, 0.0, sp)
        assert below["growing"] >= 1, (
            f"no growing mode below k=m_cs (CFJ tachyon missing): {below}")
        assert above["growing"] == 0, (
            f"spurious growing mode above k=m_cs: {above}")


# ── 6. Constraint-damping eigenvalues track K ─────────────────────────────────

class TestConstraintDampingModes:
    """The constraint-cleaning subsystem damps at rate K/2 (underdamped Gundlach
    oscillator)."""
    LAM = 0.2

    def test_damping_rate_tracks_K(self):
        kx = ky = kz = 2.0          # CFJ-stable: no tachyon to contaminate min Re
        for K in (0.5, 1.0, 2.0, 5.0):
            p = _params(self.LAM, K1=K, K2=K, ko_sigma=0.0)
            sp = symbol_params(p, 0.3125, 0.3125, 0.3125)
            min_re = float(np.min(np.linalg.eigvals(
                _symbol(kx, ky, kz, sp, with_ko=False)).real))
            assert abs(min_re - (-K / 2.0)) < 1e-6, (
                f"K={K}: most-damped Re = {min_re:.4f}, expected {-K/2:.4f}")

    def test_no_damping_at_zero_K(self):
        p = _params(self.LAM, K1=0.0, K2=0.0, ko_sigma=0.0)
        sp = symbol_params(p, 0.3125, 0.3125, 0.3125)
        min_re = float(np.min(np.linalg.eigvals(
            _symbol(2.0, 2.0, 2.0, sp, with_ko=False)).real))
        assert min_re > -1e-9, f"damping present at K=0: min Re = {min_re:.2e}"


# ── 7. Strong hyperbolicity (well-posedness) ──────────────────────────────────

class TestStrongHyperbolicity:
    def test_principal_symbol_well_conditioned(self):
        """The first-order principal symbol must be diagonalizable with purely
        imaginary eigenvalues and a bounded eigenvector condition number over all
        propagation directions on the unit SPHERE: the definition of strong
        hyperbolicity (well-posedness)."""
        sp = symbol_params(_params(0.4), 0.3125, 0.3125, 0.3125)
        cond = principal_symbol_condition(sp, n_dirs=128)
        assert np.isfinite(cond), "principal symbol has complex wave speeds (not hyperbolic)"
        assert cond < 50.0, f"principal-symbol condition number {cond:.1f} too large"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
