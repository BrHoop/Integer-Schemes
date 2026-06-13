"""
CPU-safe plumbing tests for the 3D benchmark/roofline harness (Phase 1, Step 1.4).

The throughput numbers are only meaningful on the H200, but the plumbing -- device
metadata, the HLO cost model (FLOP/byte regime), the oracle, and the metric dict
-- is device-independent and must not rot.  These exercise it on a tiny CPU grid
(timings ignored) so a refactor that breaks the harness is caught without a GPU.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from mcs3d import benchmark as bm
from mcs3d.main import load_parameters


def test_gpu_info_has_fields():
    info = bm.gpu_info()
    for key in ("backend", "device", "peak_bw_GBs", "peak_fp64_GFLOPs"):
        assert key in info


def test_oracle_is_callable_and_finite():
    p = load_parameters(str(bm._find_params()))
    ez = bm._make_oracle(p)
    X, Y, Z = np.meshgrid(*[np.linspace(-1, 1, 4)] * 3, indexing="ij")
    out = ez(X, Y, Z, 0.3)
    assert out.shape == X.shape and np.all(np.isfinite(out))


def test_measure_returns_expected_metrics(monkeypatch):
    """A tiny CPU measurement must populate every metric key and report a valid
    arithmetic-intensity regime ('memory' for the DRAM-bound MCS FD step)."""
    monkeypatch.setattr(bm, "N_SCAN_STEPS", 2)
    monkeypatch.setattr(bm, "N_REPS", 1)
    monkeypatch.setattr(bm, "N_CORRECT", 2)
    monkeypatch.setattr(bm, "N_REPORT_STEPS", 10)

    base = load_parameters(bm.__file__.replace("src/mcs3d/benchmark.py", "params.toml"))
    row = bm.measure("floating_point", 8, base)

    for key in ("scheme", "nx", "compile_s", "per_step_us", "rhs_per_step_us",
                "throughput_Mpts_s", "state_MB", "l2_err"):
        assert key in row, f"missing metric {key}"
    assert row["nx"] == 8
    assert row["per_step_us"] > 0
    # The cost model must classify the MCS FD step as memory-bound (low FLOP/byte).
    assert row.get("bound") in ("memory", "compute")
    if "rhs_flop_per_byte" in row:
        assert row["rhs_flop_per_byte"] > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
