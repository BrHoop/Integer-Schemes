"""Tier-A6 CAHD damping-rate certificate (see ``docs/BSSN_VALIDATION_PLAN.md``).

The analogue of the MCS -K/2 constraint-damping certificate. The production RHS damps
the Hamiltonian constraint via CAHD:  ``chi_rhs += cahd_c * chi * (dx^2/dt) * H``. This
test seeds a coherent Hamiltonian-constraint violation (a smooth ``hambump``) and evolves
it at several ``cahd_c`` values, confirming the Hamiltonian DECAY RATE strengthens
monotonically with ``cahd_c`` — i.e. the damping is really CAHD doing it, scaling with the
coefficient.

**Why a coherent bump, not robust noise (redesigned 2026-06-17).** The first version used
the ``robust`` noise ID and FAILED non-monotonically: broadband noise lets KO dissipation
and noise dynamics swamp CAHD's effect on ||H||, and strong CAHD destabilises at coarse N
(cahd_c=0.12 -> NaN). A smooth, low-frequency Hamiltonian bump with weak KO isolates CAHD's
damping, so the rate scales cleanly with the coefficient — mirroring the MCS certificate's
use of a clean mode rather than noise. Calibrated rates (N=12, 2 crossings, ko 0.02,
amp 0.005): cahd_c = 0.0 / 0.03 / 0.06  ->  H_exp = +0.001 / -2.14 / -3.05 (e-fold/cross).

CPU-short (N=12) but runs one evolution PER cahd_c (each its own compile), so it is the
heaviest Tier-A CPU test; ``slow``-marked. Run one test per process (each holds a compiled
RHS) to stay under the laptop memory cap.
"""

import jax
jax.config.update("jax_enable_x64", True)

import pytest

from bssn3d.longrun_stability import run, fit_rates

CAHD_VALUES = (0.0, 0.03, 0.06)     # off, half-production, production


@pytest.mark.slow
def test_cahd_damping_scales_with_coefficient():
    """||H|| decay rate must strengthen (become more negative) as cahd_c increases,
    on a coherent Hamiltonian bump where CAHD is the dominant effect on ||H||."""
    rates = {}
    for c in CAHD_VALUES:
        rows = run("hambump", N=12, crossings=2.0, samples=8, ko_sigma=0.02,
                   amp=0.005, out=None, cahd_c=c)
        assert all(r["finite"] for r in rows), f"cahd_c={c} did not stay finite"
        rates[c] = fit_rates(rows)["H_exp"]    # e-folds/crossing; more negative = faster

    print("\n  cahd_c   ||H|| exp decay rate (e-fold/crossing)")
    for c in CAHD_VALUES:
        print(f"   {c:5.3f}   {rates[c]:+.4f}")

    # Without CAHD the bump should not appreciably damp (rate ~ 0, not strongly negative).
    assert rates[0.0] > -0.5, (
        f"||H|| damped even with CAHD off (rate {rates[0.0]:+.4f}) — KO/dynamics, not CAHD, "
        f"is doing the damping; the certificate would be meaningless")
    # Turning CAHD on must damp markedly faster than off, and stronger CAHD faster still —
    # strictly monotone in the coefficient (the certificate). Margins are generous vs the
    # calibrated rates (0.0/0.03/0.06 -> +0.001/-2.14/-3.05).
    assert rates[0.03] < rates[0.0] - 0.5, (
        f"CAHD on did not damp faster than off: {rates[0.03]:+.4f} vs {rates[0.0]:+.4f}")
    assert rates[0.06] < rates[0.03] - 0.1, (
        f"2x CAHD did not damp faster than half: {rates[0.06]:+.4f} vs {rates[0.03]:+.4f}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "slow"]))
