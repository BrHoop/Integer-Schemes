"""M4 gate: the fused CUDA BSSN RHS == verbatim RHS (round-off), the thesis-target kernel.

The kernel is GPU-only, so the round-trip assertion is skipped unless `cuda/rhs_fused.so` is built
and a GPU is present. The scalar-pack / output-mapping invariants are CPU-checkable.
"""

import jax
import pytest

from bssn3d import fused_rhs_cuda as m4


def _has_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def test_scalar_pack_and_output_map():
    from bssn3d.state import PhysicsParams
    s = m4._scalars(PhysicsParams(), 0.06, 0.06, 0.06, 0.01, 0.0)
    assert s.shape[0] == 16 and m4.NOUT == 24          # 13 algebra scalars + dx,dy,dz
    assert m4._OUT_FIELDS[0] == "alpha"                # OUT[k] field-name order


def test_missing_lib_raises_clear_error():
    if m4._LIB.exists():
        pytest.skip("rhs_fused.so is built")
    from bssn3d.grid import Grid
    from bssn3d.state import PhysicsParams
    from bssn3d import initial_data as bid
    m4._registered = False
    g = Grid.from_domain(8, order=6)
    with pytest.raises(FileNotFoundError, match="build_fused.sh"):
        m4.device_rhs_fused(bid.gauge_wave(g, 0.01), PhysicsParams(), g.dx, g.dy, g.dz, 0.01, 0.0)


@pytest.mark.skipif(not (m4._LIB.exists() and _has_gpu()),
                    reason="needs cuda/rhs_fused.so built + a GPU device")
def test_fused_matches_verbatim():
    import jax.numpy as jnp
    from bssn3d.state import BSSNState
    from bssn3d.rhs import BSSNSolver
    g, p, dt, state = m4._setup(16)
    ref = BSSNSolver(g, p, order=6, scheme="verbatim", dt=dt).rhs_dict(state, t=0.0, dt=dt)
    dev = jax.jit(lambda s: m4.device_rhs_fused(BSSNState(s), p, g.dx, g.dy, g.dz, dt, 0.0))(state.data)
    worst = max(float(jnp.max(jnp.abs(jnp.asarray(ref[k]) - dev[k]))) for k in ref)
    assert worst <= 1e-8, f"max|fused - verbatim| = {worst:.3e}"
