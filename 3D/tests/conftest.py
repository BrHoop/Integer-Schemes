"""Pytest configuration for the 3D MCS test suite."""

from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 3D/


@pytest.fixture(scope="session")
def params_file() -> str:
    return str(PROJECT_ROOT / "params.toml")
