"""Phase-4 (4.C) guard: BSSN semi-discrete spectrum + MSRK ECF machinery.

Pins the canonical pieces: stencil Fourier symbols match the real operator, Minkowski is a fixed
point, the spectrum is left-half-plane stable (no growing mode), and the ECF table is well-formed
with RK4-2(2) the robust MSRK pick at the stability limit (RK4-3 worst). CPU, fast.
"""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from bssn3d.msrk_spectrum import (
    validate_symbols, spectrum, ecf_table, growing_mode_diagnostic,
)


def test_stencil_symbols_match_operator():
    e1, e2 = validate_symbols()
    assert e1 < 1e-12 and e2 < 1e-11


def test_minkowski_is_fixed_point_and_lhp_stable():
    eigs, rhs0 = spectrum(dx=1.0, dt=0.25, ko_sigma=0.0, n_theta=7)
    assert rhs0 < 1e-10                       # RHS(Minkowski) == 0
    assert eigs.real.max() < 1e-6             # no genuine growing mode


def test_no_positive_real_growing_mode():
    gm = growing_mode_diagnostic()
    assert gm[0] < 1e-6                        # most-positive Re ~ 0


def test_ecf_ranking_at_stability_limit():
    eigs, _ = spectrum(dx=1.0, dt=0.25, ko_sigma=0.1, n_theta=7)
    rows, rk4_ecf = ecf_table(eigs, dx=1.0)
    d = {m: (cfl, ecf) for (m, st, buf, cfl, ecf) in rows}
    # every method has a positive CFL (relative-stability criterion didn't collapse to 0)
    assert all(cfl > 0 for (cfl, _ecf) in d.values())
    # RK4-2(2) is the best / robust MSRK; RK4-3 is the worst (smallest CFL) near the limit
    assert d["rk4_2_2"][1] <= d["rk4_2_1"][1]
    assert d["rk4_3"][0] < d["rk4_2_2"][0]
