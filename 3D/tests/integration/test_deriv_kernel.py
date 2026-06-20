"""M2a gate: the device derivative stage == derivative_bundle (round-off).

The kernel runs on CUDA, so the round-trip assertion is GPU-only (skipped unless
`cuda/deriv_2p5d.so` is built and a GPU is present). The name-mapping invariant — the kernel's
138 outputs map to exactly the derivative_bundle keys — is CPU-checkable and runs everywhere
(it is the link that keeps the device output aligned with the reference).
"""

import jax
import pytest

from bssn3d import deriv_kernel as dk


def _has_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def test_name_mapping_matches_bundle():
    """The 138 kernel outputs map to exactly the derivative_bundle keys, in a fixed order."""
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from bssn3d.grid import Grid
    from bssn3d.state import BSSNState
    from bssn3d.derivative_bundle import derivative_bundle
    from mcs_common.derivatives import SpatialDerivative
    g = Grid.from_domain(8, order=6)
    diff = SpatialDerivative(order=6, ko_order=8)
    state = BSSNState(jnp.zeros((24,) + g.shape))
    ref = derivative_bundle(state, diff, g.dx, g.dy, g.dz)
    assert dk.N_DERIV == 138
    assert set(dk._NAMES) == set(ref) and len(dk._NAMES) == len(ref)


def test_missing_lib_raises_clear_error():
    if dk._LIB.exists():
        pytest.skip("deriv_2p5d.so is built; error-path test is for the unbuilt case")
    import jax.numpy as jnp
    from bssn3d.grid import Grid
    from bssn3d.state import BSSNState
    dk._registered = set()
    g = Grid.from_domain(8, order=6)
    with pytest.raises(FileNotFoundError, match="build_deriv.sh"):
        dk.device_derivative_bundle(BSSNState(jnp.zeros((24,) + g.shape)), g.dx, g.dy, g.dz)


def _variant_built(v):
    from pathlib import Path
    return (dk._CUDA / dk._VARIANTS[v][0]).exists()


@pytest.mark.parametrize("variant", ["global", "smem"])   # M2a, M2b
def test_device_matches_bundle(variant):
    if not (_variant_built(variant) and _has_gpu()):
        pytest.skip(f"needs cuda/{dk._VARIANTS[variant][0]} built + a GPU device")
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import jax.numpy as jnp
    from bssn3d.grid import Grid
    from bssn3d.state import BSSNState
    from bssn3d.derivative_bundle import derivative_bundle
    from mcs_common.derivatives import SpatialDerivative
    g = Grid.from_domain(16, order=6)
    diff = SpatialDerivative(order=6, ko_order=8)
    rng = np.random.default_rng(11)
    state = BSSNState(jnp.array(rng.standard_normal((24,) + g.shape)))
    ref = derivative_bundle(state, diff, g.dx, g.dy, g.dz)
    dev = jax.jit(lambda s: dk.device_derivative_bundle(
        BSSNState(s), g.dx, g.dy, g.dz, variant))(state.data)
    worst = max(float(jnp.max(jnp.abs(jnp.asarray(ref[k]) - dev[k]))) for k in ref)
    assert worst <= 1e-11, f"{variant}: max|device - bundle| = {worst:.3e}"
