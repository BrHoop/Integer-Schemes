"""Phase 2.2 structural tests for the transliterated BSSN RHS.

These are reference-free structural/sanity gates (CPU, fast). The decisive
*fidelity* gate — a bit-compare of the algebra against Dendro-GR C++ on one eval —
is separate (needs the Dendro oracle dump) and is not run here.

Key gate: **Minkowski is a static solution**, so the full RHS must vanish to
round-off. This exercises every one of the 850 SSA statements (CAHD+SSL variant),
all 138 derivative inputs, all 24 outputs, SSA define-before-use, ``pow``/``sqrt``/
``exp`` on arrays, and confirms the physics degenerates correctly at flat space —
including the CAHD term (Hamiltonian constraint H=0) and the SSL term
(``sqrt(chi)-alpha=0`` at flat space), so both vanish there.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from bssn3d.grid import Grid
from bssn3d.state import PhysicsParams, VAR_NAMES
from bssn3d.rhs import BSSNSolver
from bssn3d.derivative_bundle import derivative_bundle
from bssn3d import initial_data as bid
from bssn3d import _bssn_rhs_generated as gen
from bssn3d import _codegen

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grid():
    return Grid.from_domain(16, order=6, lo=-0.5, hi=0.5)


@pytest.fixture(scope="module")
def solver(grid):
    return BSSNSolver(grid, PhysicsParams(), order=6)


# --- generated-module provenance & shape -------------------------------------

def test_generated_inventory():
    assert len(gen.FIELD_INPUTS) == 24
    assert len(gen.GRAD1_INPUTS) == 72
    assert len(gen.GRAD2_INPUTS) == 66
    assert gen.OUTPUT_FIELDS == VAR_NAMES          # output order == state order


def test_derivative_bundle_keys(grid, solver):
    s = bid.gauge_wave(grid, amplitude=0.01)
    D = derivative_bundle(s, solver.diff_op, grid.dx, grid.dy, grid.dz)
    assert len(D) == 138
    # mixed second derivatives are present and indexed (min, max)
    assert "grad2_0_1_gt0" in D and "grad2_0_2_alpha" in D
    assert all(v.shape == grid.shape for v in D.values())


# --- the static-Minkowski gate -----------------------------------------------

def test_minkowski_is_static(solver, grid):
    out = solver.rhs_dict(bid.minkowski(grid))
    worst = max(float(jnp.max(jnp.abs(v))) for v in out.values())
    assert worst < 1e-12, f"flat space not static (worst |RHS| = {worst:.3e})"


def test_rhs_finite_and_shaped(solver, grid):
    dt = solver.rhs(bid.gauge_wave(grid, amplitude=0.01))
    assert dt.data.shape == (24,) + grid.shape
    assert bool(jnp.all(jnp.isfinite(dt.data)))


def test_rhs_deterministic(solver, grid):
    s = bid.robust_stability(grid, amp=1e-3, seed=1)
    a = solver.rhs(s).data
    b = solver.rhs(s).data
    assert jnp.array_equal(a, b)


def test_rhs_jit(solver, grid):
    s = bid.gauge_wave(grid, amplitude=0.01)
    jitted = jax.jit(solver.rhs)
    assert jnp.allclose(jitted(s).data, solver.rhs(s).data)


# --- committed artifact must match a fresh regen (guards against drift) ------

def test_generated_matches_regen(tmp_path):
    src = _codegen.DENDRO_CSE
    if not src.exists():
        pytest.skip(f"Dendro-GR source not present at {src}")
    fresh = _codegen.generate(src=src, out=tmp_path / "regen.py")
    committed = Path(gen.__file__)
    # compare on content with the generation-date line normalized out
    def _norm(text):
        return "\n".join(
            l for l in text.splitlines() if not l.strip().startswith("generated")
        )
    assert _norm(fresh.read_text()) == _norm(committed.read_text())
