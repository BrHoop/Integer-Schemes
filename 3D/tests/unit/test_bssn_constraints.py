"""Phase 2.3: Hamiltonian + momentum constraint operator (transliterated physcon).

The key reference-free check: the analytic gauge-wave ID is exactly
constraint-satisfying (H = M^i = 0), so the *discrete* constraints are pure
truncation error and must converge at the FD order (6) under refinement. That both
validates the transliteration and is the constraint apples test (momentum is the
under-protected channel). No evolution here → cheap, safe on the laptop.
"""

import numpy as np
import jax
import pytest

from bssn3d.grid import Grid
from bssn3d.constraints import ConstraintSolver
from bssn3d import initial_data as bid
from bssn3d import _constraints_generated as cgen
from bssn3d import _codegen_constraints

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def orders():
    """(order_H, order_M) from the analytic gauge-wave ID at three resolutions."""
    Ns = [24, 32, 48]
    H, M = [], []
    for N in Ns:
        g = Grid.from_domain(N, order=6, lo=-0.5, hi=0.5)
        cs = ConstraintSolver(g, order=6)
        h, m = cs.l2(bid.gauge_wave(g, amplitude=0.01))
        H.append(h)
        M.append(m)
    r = np.log(Ns[-1] / Ns[-2])
    return np.log(H[-2] / H[-1]) / r, np.log(M[-2] / M[-1]) / r


def test_constraints_inventory():
    assert cgen.OUTPUTS == ["H", "M0", "M1", "M2"]      # psi4 dropped by DCE
    assert len(cgen.GRAD1_INPUTS) == 51
    assert len(cgen.GRAD2_INPUTS) == 42


def test_hamiltonian_converges_at_fd_order(orders):
    oH, _ = orders
    assert oH > 5.5, f"Hamiltonian constraint order {oH:.2f} below 5.5"


def test_momentum_converges_at_fd_order(orders):
    _, oM = orders
    assert oM > 5.5, f"momentum constraint order {oM:.2f} below 5.5"


def test_minkowski_constraints_vanish():
    g = Grid.from_domain(16, order=6)
    H, M = ConstraintSolver(g, order=6).l2(bid.minkowski(g))
    assert H < 1e-13 and M < 1e-13      # flat space is exactly constraint-satisfying


def test_constraints_match_regen(tmp_path):
    src = _codegen_constraints.PHYSCON
    if not src.exists():
        pytest.skip("vendored physcon.cpp missing")
    fresh = _codegen_constraints.generate(src=src, out=tmp_path / "regen.py")
    norm = lambda t: "\n".join(l for l in t.splitlines()
                               if not l.strip().startswith("generated"))
    import pathlib
    assert norm(fresh.read_text()) == norm(pathlib.Path(cgen.__file__).read_text())
