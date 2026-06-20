"""Step 3.2 gate: is the tensor-core derivative worth the slab-redundancy it forces?

`tc_redundancy_tradeoff.py` reduces the T=13-fp64 vs T=8-TC choice to one number: the algebra
cancels (identical, point-wise), so TC wins iff its effective per-op speedup s < redundancy(13)/
redundancy(8) = 0.652 (TC must be ≥1.53× faster). These tests pin that break-even, its
f-independence, and the sign-flip if a larger WGMMA-aligned slab becomes feasible.
"""

import pytest

from bssn3d import tc_redundancy_tradeoff as TC
from bssn3d.fused_peak_model import redundancy


def test_break_even_threshold():
    """TC must be ≥1.53× faster/op (s<0.652) to overcome the T=8 vs T=13 redundancy."""
    s_star = TC.break_even_s()
    assert s_star == pytest.approx(redundancy(13) / redundancy(8))
    assert s_star == pytest.approx(0.652, abs=1e-3)
    assert 1.0 / s_star == pytest.approx(1.53, abs=1e-2)


def test_break_even_is_f_independent():
    """At s = s*, the whole-step speedup is exactly 1.0 for every derivative-fraction f (the
    algebra cancels), confirming the threshold does not depend on the algebra/deriv split."""
    s_star = TC.break_even_s()
    for f in (0.2, 0.35, 0.5, 0.8):
        assert TC.total_step_speedup(s_star, f) == pytest.approx(1.0, abs=1e-9)


def test_tc_win_requires_below_threshold():
    """Below s* TC is a net win; above it TC loses — monotone in s."""
    f = 0.4
    assert TC.total_step_speedup(0.30, f) > 1.0          # 3.3×/op TC → win
    assert TC.total_step_speedup(0.80, f) < 1.0          # weak TC → loss
    # monotone decreasing speedup as s grows (slower TC)
    sp = [TC.total_step_speedup(s, f) for s in (0.2, 0.4, 0.6, 0.8)]
    assert all(a > b for a, b in zip(sp, sp[1:]))


def test_larger_aligned_slab_flips_the_sign():
    """A WGMMA-aligned slab larger than T=13 has LOWER redundancy than the fp64 slab → break-even
    s exceeds 1 (TC wins for any speedup). The blocker is SMEM, not the redundancy math."""
    assert TC.break_even_s(13, 8) < 1.0                  # T=8 must beat its redundancy tax
    assert TC.break_even_s(13, 16) > 1.0                 # T=16 would be cheaper than fp64 slab
    assert redundancy(16) < redundancy(13)
