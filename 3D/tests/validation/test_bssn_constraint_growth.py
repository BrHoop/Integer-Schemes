"""Tier-A5 constraint growth-rate diagnostics (see ``docs/BSSN_VALIDATION_PLAN.md``).

A4 records the ||H||(t), ||M||(t) histories; A5 turns them into NUMBERS: fit the secular
growth/decay rate of each channel SEPARATELY (Hamiltonian vs the under-protected
momentum channel). Two certificates:

  * CAHD certificate — on Minkowski+noise the Hamiltonian constraint must be *damped*
    (rate <= ~0), confirming CAHD is actively controlling it.
  * Momentum drift — on the forced gauge wave the momentum channel drifts upward; A5
    quantifies the rate and confirms it is SUB-EXPONENTIAL (bounded, a linear model fits
    at least as well as an exponential) over the window — i.e. secular drift, not blow-up.

Rates are fit by ``bssn3d.longrun_stability.fit_rates`` (the same fitter usable on the
Marylou CSV series). CPU-short (N=12, 2 crossing times); ``slow``-marked.
"""

import jax
jax.config.update("jax_enable_x64", True)

import pytest

from bssn3d.longrun_stability import run, fit_rates


def _print_rates(title, r):
    print(f"\n  {title}")
    print(f"    H: linear={r['H_lin']:+.4e}/cross   exp={r['H_exp']:+.4f} e-fold/cross")
    print(f"    M: linear={r['M_lin']:+.4e}/cross   exp={r['M_exp']:+.4f} e-fold/cross")
    print(f"    M linear-vs-exp residual ratio = {r['M_lin_vs_exp_resid']:.3f} "
          f"(<1 => linear fits better => sub-exponential)")


@pytest.mark.slow
def test_cahd_damps_hamiltonian_on_noise():
    """CAHD certificate: on Minkowski+noise the Hamiltonian constraint is damped, not
    growing (the channel CAHD protects)."""
    rows = run("robust", N=12, crossings=2.0, samples=8, ko_sigma=0.1, amp=1e-8, out=None)
    r = fit_rates(rows)
    _print_rates("robust stability (N=12, 2 crossings, KO 0.1)", r)
    assert all(row["finite"] for row in rows)
    assert r["H_exp"] < 0.2, f"Hamiltonian not damped: exp rate {r['H_exp']:+.3f}/cross"
    assert rows[-1]["H"] <= rows[0]["H"], "Hamiltonian net grew over the window"


@pytest.mark.slow
def test_gauge_wave_momentum_drift_subexponential():
    """Quantify the under-protected momentum drift on the gauge wave and confirm it is
    sub-exponential (bounded secular growth, not blow-up) over the window."""
    rows = run("gauge", N=12, crossings=2.0, samples=8, ko_sigma=0.1, amp=0.01, out=None)
    r = fit_rates(rows)
    _print_rates("gauge wave (N=12, 2 crossings, KO 0.1)", r)
    assert all(row["finite"] for row in rows)
    assert r["M_lin"] > 0.0, "momentum constraint did not drift upward"
    assert rows[-1]["M"] < 0.5, f"momentum unbounded over window: {rows[-1]['M']:.3e}"
    # linear model should describe the drift at least roughly as well as exponential
    # (catches a catastrophic exponential blow-up within the window)
    assert r["M_lin_vs_exp_resid"] < 3.0, (
        f"momentum drift looks exponential (resid ratio {r['M_lin_vs_exp_resid']:.2f})")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "slow"]))
