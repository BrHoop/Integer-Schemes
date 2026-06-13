"""
Spectral / eigenvalue certificates for the MCS 2D discretization (Step 1.1).

Every test here is post-processing of the linearized semi-discrete symbol
M(kx, ky) built in `mcs2d.validate`, cross-checked against the exact AD Jacobian
of the real RHS.  Together they certify the properties a numerical relativist
expects of a hyperbolic solver BEFORE trusting it for long evolutions:

  * no growing modes (semi-discrete stability)         -> test_no_growing_modes
  * fully-discrete von Neumann stability at the CFL    -> test_rk4_amplification
  * correct wave propagation (dispersion is 6th-order) -> TestDispersion
  * the exact Kreiss-Oliger dissipation kernel         -> TestKOKernel
  * the physical CFJ tachyon is present & located      -> TestModeStructure
  * constraint damping tracks K                        -> TestConstraintDampingModes
  * strong hyperbolicity (well-posedness)              -> test_strong_hyperbolicity

All CPU-safe: small dense eigensolves and one small AD Jacobian.

Conventions
-----------
* CFJ-STABLE regime (Lambda <= ~0.31 so every grid wavenumber has |k| > m_cs)
  is used for the stability and von Neumann tests, where growth would be a defect.
* CFJ-UNSTABLE probing (Lambda = 0.4) is used for the mode-structure test, where
  a growing mode below k = m_cs is the correct physics.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from mcs2d.validate import (
    symbol_params, _symbol, grid_wavenumbers, load_parameters,
    fd_symbol_d1, fd_symbol_d2, ko_symbol_1d,
    rk4_amplification_radius, em_wave_omega, _symbol_continuum,
    mode_classification, principal_symbol_condition, _build_jacobian,
    _PARAMS_FILE, NF,
)


def _params(lam, **over):
    p = load_parameters(_PARAMS_FILE)
    p.update({"Lambda": lam, "ko_sigma": 0.05, "K1": 1.0, "K2": 1.0, **over})
    return p


def _grid_symbol_params(lam, N=32, **over):
    """symbol_params on an NxN grid, plus the discrete grid wavenumbers."""
    p = _params(lam, **over)
    dx = (p["xmax"] - p["xmin"]) / N
    dy = (p["ymax"] - p["ymin"]) / N
    sp = symbol_params(p, dx, dy)
    kxs = grid_wavenumbers(N, p["xmax"] - p["xmin"])
    kys = grid_wavenumbers(N, p["ymax"] - p["ymin"])
    return sp, kxs, kys, p, dx


# ── 0. Symbol <-> Jacobian consistency (foundation) ───────────────────────────

class TestSymbolMatchesJacobian:
    """The analytic symbol must reproduce the eigenvalues of the EXACT AD Jacobian
    of the real RHS.  If this passes, every other spectral test is trustworthy;
    if it fails, the symbol is wrong and nothing downstream means anything."""

    def test_eigenvalue_multisets_agree(self):
        N = 8
        J, sim, params = _build_jacobian(N, N, ko_sigma=0.05, lam=0.4, background="pi0")
        eig_J = np.linalg.eigvals(J)

        sp = symbol_params(params, sim.dx, sim.dy)
        kxs = grid_wavenumbers(N, params["xmax"] - params["xmin"])
        kys = grid_wavenumbers(N, params["ymax"] - params["ymin"])
        eig_S = []
        for kx in kxs:
            for ky in kys:
                eig_S.extend(np.linalg.eigvals(_symbol(kx, ky, sp, with_ko=True)))
        eig_S = np.array(eig_S)

        # Greedy nearest-neighbour multiset match.
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
    LAM_STABLE = 0.2     # m_cs = 0.4 < smallest grid |k| = 2pi/10 ~ 0.628

    def test_no_growing_modes(self):
        """In the CFJ-stable regime every semi-discrete eigenvalue must have
        Re(lambda) <= 0 (to round-off): no mode grows in time."""
        sp, kxs, kys, _, _ = _grid_symbol_params(self.LAM_STABLE)
        max_re = max(np.max(np.linalg.eigvals(_symbol(kx, ky, sp, with_ko=True)).real)
                     for kx in kxs for ky in kys)
        assert max_re < 1e-9, f"max Re(lambda) = {max_re:.2e} > 0 (growing mode)"

    def test_ko_only_damps(self):
        """KO must move eigenvalues LEFT (it is a scalar on the diagonal, so it
        shifts every eigenvalue by exactly the real KO symbol).  Verify the shift
        is non-positive everywhere and strictly negative at high k."""
        sp, kxs, kys, _, _ = _grid_symbol_params(self.LAM_STABLE)
        worst_increase = -np.inf
        for kx in kxs:
            for ky in kys:
                e_ko = np.sort(np.linalg.eigvals(_symbol(kx, ky, sp, with_ko=True)).real)
                e_no = np.sort(np.linalg.eigvals(_symbol(kx, ky, sp, with_ko=False)).real)
                worst_increase = max(worst_increase, float(np.max(e_ko - e_no)))
        assert worst_increase < 1e-9, (
            f"KO increased a Re(lambda) by {worst_increase:.2e} (anti-dissipative!)")


# ── 2. Fully-discrete (von Neumann) stability at the CFL ───────────────────────

class TestRK4Amplification:
    LAM_STABLE = 0.2

    def test_amplification_within_unity(self):
        """The RK4 one-step amplification factor |G(k)| must be <= 1 (to round-off)
        for every grid wavenumber at the operating CFL.  This is the actual
        guarantee that the scheme does not blow up -- stronger than the
        semi-discrete spectrum, since it includes the time integrator."""
        sp, kxs, kys, p, dx = _grid_symbol_params(self.LAM_STABLE)
        dt = p.get("cfl", 0.05) * dx
        worst = max(rk4_amplification_radius(_symbol(kx, ky, sp, with_ko=True), dt)
                    for kx in kxs for ky in kys)
        assert worst < 1.0 + 1e-9, f"max |G(k)| = {worst:.6f} > 1 (CFL too large)"

    def test_cfl_has_margin(self):
        """Sanity on the CFL choice: at 4x the operating CFL the scheme should
        already be near or past the stability edge, confirming 0.05 is a genuine
        margin and not wildly conservative-by-accident."""
        sp, kxs, kys, p, dx = _grid_symbol_params(self.LAM_STABLE)
        dt_big = 4.0 * p.get("cfl", 0.05) * dx
        worst = max(rk4_amplification_radius(_symbol(kx, ky, sp, with_ko=True), dt_big)
                    for kx in kxs for ky in kys)
        assert worst > 1.0 - 0.05, (
            f"max |G| at 4x CFL = {worst:.4f}: CFL=0.05 is far from the edge "
            f"(harmless, but the margin estimate is off)")


# ── 3. Dispersion accuracy (6th order) ────────────────────────────────────────

class TestDispersion:
    LAM = 0.2

    def test_dispersion_is_sixth_order(self):
        """The FD wave frequency omega_FD(k) must approach the continuum MCS
        frequency sqrt(k^2 + m_cs k) at 6th order: halving dx must cut the error
        by ~2^6 = 64.  Tests that the stencil propagates waves correctly."""
        p = _params(self.LAM)
        k = 2.0                                  # fixed physical wavenumber
        m_cs = 2.0 * self.LAM
        omega_exact = np.sqrt(k**2 + m_cs * k)

        def err(dx):
            sp = symbol_params(p, dx, dx)
            return abs(em_wave_omega(k, sp, fd=True) - omega_exact)

        e1, e2 = err(0.05), err(0.025)
        ratio = e1 / e2
        assert 40 < ratio < 90, (
            f"dispersion error ratio under dx-halving = {ratio:.1f} "
            f"(expected ~64 for 6th order); e1={e1:.2e}, e2={e2:.2e}")

    def test_group_velocity_accurate_at_low_k(self):
        """Group velocity v_g = d(omega)/dk (the speed wave packets actually
        travel) must match the continuum value to high accuracy at well-resolved
        wavenumbers."""
        p = _params(self.LAM)
        dx = 0.05
        sp = symbol_params(p, dx, dx)
        m_cs = 2.0 * self.LAM
        k = 1.0
        h = 1e-4
        vg_fd = (em_wave_omega(k + h, sp, fd=True) - em_wave_omega(k - h, sp, fd=True)) / (2 * h)
        # continuum: omega = sqrt(k^2 + m_cs k) -> vg = (2k + m_cs)/(2 omega)
        omega = np.sqrt(k**2 + m_cs * k)
        vg_exact = (2 * k + m_cs) / (2 * omega)
        rel = abs(vg_fd - vg_exact) / abs(vg_exact)
        assert rel < 1e-3, f"group-velocity rel error = {rel:.2e} at k=1, dx=0.05"


# ── 4. Kreiss-Oliger kernel verification (against the real stencil) ────────────

class TestKOKernel:
    def test_stencil_symbol_is_sin6(self):
        """Apply the SOLVER's actual KO operator to a pure Fourier mode and
        confirm its eigenvalue equals the analytic 6th-order symbol
        -(sigma/dx) sin^6(k dx/2) to round-off.  Catches a wrong CKO coefficient
        or a sign flip (the load-bearing anti-dissipation bug)."""
        from mcs2d.main import MaxwellChernSimons2D, load_parameters
        p = load_parameters(_PARAMS_FILE)
        N = 32
        p.update({"scheme": "floating_point", "Nx": N, "Ny": N, "bc_type": "periodic"})
        dx = (p["xmax"] - p["xmin"]) / N
        sim = MaxwellChernSimons2D(dx, dx, p["Lambda"], p)
        x = np.asarray(sim.x)
        sigma = 0.05
        for m in (1, 3, 7):                       # several wavenumbers
            k = 2 * np.pi * m / (p["xmax"] - p["xmin"])
            mode = np.exp(1j * k * x)[:, None] * np.ones((1, sim.Ny_tot))
            ko_re = np.asarray(sim.diff_op.compute_ko(mode.real, dx, sigma, axis=0))
            ko_im = np.asarray(sim.diff_op.compute_ko(mode.imag, dx, sigma, axis=0))
            applied = ko_re + 1j * ko_im
            expected = ko_symbol_1d(k, dx, sigma)
            # compare in the interior (away from edge padding)
            ratio = applied[sim.ng:-sim.ng, sim.ng] / mode[sim.ng:-sim.ng, sim.ng]
            err = float(np.max(np.abs(ratio - expected)))
            assert err < 1e-10, f"KO symbol at m={m}: stencil={ratio[0]:.4f} vs {expected:.4f}"
            assert expected <= 0, f"KO symbol positive (anti-dissipative) at m={m}"


# ── 5. Mode structure & the CFJ tachyon ───────────────────────────────────────

class TestModeStructure:
    """The 10-field symbol must have the right physical mode content, and the
    Carroll-Field-Jackiw tachyon must appear exactly where the dispersion
    relation predicts (k < m_cs), confirming the CS coupling is wired correctly.
    Uses Lambda = 0.4 (m_cs = 0.8) so the CFJ band is resolvable."""
    LAM = 0.4

    def test_ten_modes_no_ko(self):
        p = _params(self.LAM, ko_sigma=0.0)
        sp = symbol_params(p, 0.3125, 0.3125)
        cls = mode_classification(2.0, 2.0, sp)     # stable k
        total = cls["oscillatory"] + cls["damped"] + cls["growing"]
        assert total == NF, f"mode count {total} != {NF}"

    def test_cfj_tachyon_below_threshold(self):
        p = _params(self.LAM, ko_sigma=0.0)
        sp = symbol_params(p, 0.3125, 0.3125)
        m_cs = 2.0 * self.LAM
        below = mode_classification(0.5 * m_cs, 0.0, sp)
        above = mode_classification(3.0 * m_cs, 0.0, sp)
        assert below["growing"] >= 1, (
            f"no growing mode below k=m_cs (CFJ tachyon missing): {below}")
        assert above["growing"] == 0, (
            f"spurious growing mode above k=m_cs: {above}")


# ── 6. Constraint-damping eigenvalues track K ─────────────────────────────────

class TestConstraintDampingModes:
    """The constraint-cleaning subsystem damps at rate K/2 (underdamped Gundlach
    oscillator).  The most-damped eigenvalue must scale linearly with K, with no
    damping at K=0 -- and the wave modes must be essentially unchanged by K
    (damping must not leak into the principal part)."""
    LAM = 0.2

    def test_damping_rate_tracks_K(self):
        kx = ky = 2.0          # CFJ-stable: no tachyon to contaminate min Re
        for K in (0.5, 1.0, 2.0, 5.0):
            p = _params(self.LAM, K1=K, K2=K, ko_sigma=0.0)
            sp = symbol_params(p, 0.3125, 0.3125)
            min_re = float(np.min(np.linalg.eigvals(_symbol(kx, ky, sp, with_ko=False)).real))
            assert abs(min_re - (-K / 2.0)) < 1e-6, (
                f"K={K}: most-damped Re = {min_re:.4f}, expected {-K/2:.4f}")

    def test_no_damping_at_zero_K(self):
        p = _params(self.LAM, K1=0.0, K2=0.0, ko_sigma=0.0)
        sp = symbol_params(p, 0.3125, 0.3125)
        min_re = float(np.min(np.linalg.eigvals(_symbol(2.0, 2.0, sp, with_ko=False)).real))
        assert min_re > -1e-9, f"damping present at K=0: min Re = {min_re:.2e}"

    def test_wave_modes_unaffected_by_K(self):
        """The oscillatory (wave) frequencies must not shift as K changes -- the
        constraint damping is a lower-order term and must stay out of the
        principal part."""
        kx = ky = 2.0
        def wave_freqs(K):
            p = _params(self.LAM, K1=K, K2=K, ko_sigma=0.0)
            sp = symbol_params(p, 0.3125, 0.3125)
            e = np.linalg.eigvals(_symbol(kx, ky, sp, with_ko=False))
            return np.sort(np.abs(e.imag[np.abs(e.real) < 1e-9]))
        f1, f5 = wave_freqs(1.0), wave_freqs(5.0)
        n = min(len(f1), len(f5))
        assert n > 0 and np.allclose(f1[-n:], f5[-n:], atol=1e-9), (
            "wave frequencies shifted with K (damping leaked into principal part)")


# ── 7. Strong hyperbolicity (well-posedness) ──────────────────────────────────

class TestStrongHyperbolicity:
    def test_principal_symbol_well_conditioned(self):
        """The first-order principal symbol must be diagonalizable with purely
        imaginary eigenvalues and an eigenvector condition number bounded over
        all propagation directions: that is the definition of strong
        hyperbolicity (well-posedness).  An unbounded condition number would mean
        the formulation is only weakly hyperbolic -- ill-posed, the failure mode
        that sinks naive NR formulations."""
        sp = symbol_params(_params(0.4), 0.3125, 0.3125)
        cond = principal_symbol_condition(sp, n_dirs=64)
        assert np.isfinite(cond), "principal symbol has complex wave speeds (not hyperbolic)"
        assert cond < 50.0, f"principal-symbol condition number {cond:.1f} too large"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
