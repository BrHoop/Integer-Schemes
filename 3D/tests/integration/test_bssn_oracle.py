"""Phase 2.2 DECISIVE GATE: bit-compare the transliterated BSSN RHS algebra
against Dendro-GR's compiled C++ on identical single-point inputs.

Pure-algebra comparison (derivatives are independent inputs on both sides), so it
isolates the transliteration fidelity. Inputs cross as hex floats (no decimal
round-trip); agreement is limited only by g++/XLA summation order. Requires g++
(CPU-only, no 2FA); skipped otherwise.

Empirically the match is at machine epsilon (~3e-16 max relative diff, many
outputs bit-identical) — comfortably inside the "compare to round-off" bar.
"""

import pytest

from bssn3d import oracle
from bssn3d._codegen import DENDRO_CSE

pytestmark = pytest.mark.skipif(not oracle.have_gpp(),
                                reason="g++ not available to build the Dendro oracle")

TOL = 1e-12   # the plan's round-off bar; actual agreement is ~1e-16


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bitcompare_vs_dendro(seed):
    worst, rows = oracle.compare(seed)
    assert worst < TOL, (
        f"seed {seed}: max relative diff {worst:.3e} exceeds {TOL:.0e}\n" +
        "\n".join(f"  {t:9s} cpp={c:+.16e} jax={j:+.16e} rel={r:.2e}"
                  for t, c, j, r in sorted(rows, key=lambda x: -x[3])[:5])
    )


def test_all_24_outputs_compared():
    _, rows = oracle.compare(0)
    assert len(rows) == 24


def test_oracle_uses_in_repo_source():
    """Self-containment: the oracle compiles the vendored CSE, not an external repo."""
    cpp = oracle.emit_harness(oracle.random_inputs(0))
    assert str(DENDRO_CSE) in cpp
    assert "/vendor/" in str(DENDRO_CSE)
    assert "Dendro-GR" not in str(DENDRO_CSE)   # not the upstream checkout path
