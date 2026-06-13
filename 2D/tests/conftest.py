"""Pytest configuration for the 2D MCS test suite.

The mcs2d, mcs3d, and mcs_common packages are expected to be installed
editable (`pip install -e ./common ./2D ./3D` from the repo root), so no
sys.path manipulation is needed here.
"""

from pathlib import Path

import jax
import pytest

# All tests use float64.
jax.config.update("jax_enable_x64", True)


# ── Common fixtures ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 2D/


@pytest.fixture(scope="session")
def params_file() -> str:
    """Path to the canonical 2D params.toml."""
    return str(PROJECT_ROOT / "params.toml")
