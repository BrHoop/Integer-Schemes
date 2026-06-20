"""Step 3.4 item-2 free win: ``pow(x, n)`` -> explicit multiplies in the CUDA codegen.

In CUDA C++ ``pow(double, double)`` is a libcall (``exp(n*log(x))``) — slow and less accurate
than ``x*x``. ``_codegen.lower_pow`` lowers the integer-exponent ``pow`` Dendro's CSE emits. These
tests pin (a) algebraic equivalence to round-off, (b) that EVERY ``pow`` in the real statement set
is consumed, and (c) the drift guard fires on anything it cannot lower.
"""

import numpy as np
import pytest

from bssn3d._codegen import DENDRO_CSE, lower_pow, parse, _POW_ANY_RE, _POW_RE


@pytest.mark.parametrize("n", [2, 3, -2, -3, 1, -1])
def test_lowered_value_matches_pow(n):
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = float(rng.uniform(0.2, 3.0))            # away from 0 (the -n cases divide)
        lowered = lower_pow(f"pow(X, {n})")
        got = eval(lowered, {"X": x})               # noqa: S307 - controlled codegen string
        assert np.isclose(got, x ** n, rtol=1e-15, atol=0.0), (n, x, lowered)


def test_no_pow_call_remains_in_output():
    # the lowered string must contain no residual pow( token
    assert "pow(" not in lower_pow("pow(K, 2)")
    assert "pow(" not in lower_pow("DENDRO_0*pow(chi, -2) + pow(gt4, 2)")


def test_factor_count_is_correct():
    # pow(x, 3) -> three factors of x; pow(x, -2) -> reciprocal of two factors
    assert lower_pow("pow(x, 3)").count("x") == 3
    assert lower_pow("pow(x, 2)") == "(x*x)"
    assert lower_pow("pow(x, -2)") == "(1.0/(x*x))"
    assert lower_pow("pow(x, -3)") == "(1.0/(x*x*x))"
    assert lower_pow("pow(x, 1)") == "x"
    assert lower_pow("pow(x, 0)") == "1.0"


def test_consumes_every_pow_in_real_statements():
    """Across the actual Dendro CSE, lower_pow must lower ALL pow() calls (none left)."""
    statements, _, _ = parse(DENDRO_CSE)
    seen, lowered_total = 0, 0
    for _lhs, rhs in statements:
        seen += len(_POW_RE.findall(rhs))
        out = lower_pow(rhs)
        assert not _POW_ANY_RE.search(out), f"residual pow in: {rhs!r} -> {out!r}"
        lowered_total += 1
    assert seen > 0, "expected the CSE to contain pow() calls to lower"


def test_drift_guard_raises_on_unlowerable_pow():
    # non-identifier base (nested expression) and fractional exponent are NOT the shape
    # Dendro emits; the guard must trip rather than silently emit a slow libcall.
    with pytest.raises(AssertionError):
        lower_pow("pow(x + y, 2)")
    with pytest.raises(AssertionError):
        lower_pow("pow(x, 0.5)")
