"""M0 geometry: coordinate-map round-trip, analytic Jacobian, patch coverage."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from multipatch import atlas as A, coord_maps as cm

_ZERO = jnp.zeros(3)
_ONE = jnp.asarray(1.0)


@pytest.fixture(scope="module")
def grid():
    return A.build_llama_grid(2.0, 1.8, 8.0, N_cube=13, N_ang=13, N_rad=13, order=6)


def test_seven_patches(grid):
    names = [p.name for p in grid.patches]
    assert names == ["cube", "shell+x", "shell-x", "shell+y", "shell-y",
                     "shell+z", "shell-z"]


def test_shell_inverse_roundtrip(grid):
    sh = grid.shells[0]
    ls = jnp.linspace(0.15, 0.85, 4)
    maxerr = 0.0
    for lx in ls:
        for ly in ls:
            for lz in ls:
                gx, gy, gz = cm.dispatch_map(sh.patch_type, sh.patch_params,
                                             lx, ly, lz, _ZERO, _ONE)
                bx, by, bz = cm.dispatch_inverse_map(sh.patch_type, sh.patch_params,
                                                     gx, gy, gz, _ZERO, _ONE)
                maxerr = max(maxerr, float(abs(bx - lx) + abs(by - ly) + abs(bz - lz)))
    assert maxerr < 1e-12


def test_shell_jacobian_matches_autodiff(grid):
    sh = grid.shells[0]

    def mapf(l):
        return jnp.array(cm.dispatch_map(sh.patch_type, sh.patch_params,
                                         l[0], l[1], l[2], _ZERO, _ONE))
    for l0 in [jnp.array([0.3, 0.6, 0.4]), jnp.array([0.7, 0.2, 0.9])]:
        Jad = jax.jacfwd(mapf)(l0)
        Jana = cm.dispatch_jacobian(sh.patch_type, sh.patch_params,
                                    l0[0], l0[1], l0[2], _ZERO, _ONE)
        assert float(jnp.max(jnp.abs(Jad - Jana))) < 1e-12


def test_coverage_no_holes(grid):
    s = A.coverage_report(grid)
    # cube fully donor-covered; shells covered except outer-radial boundary
    assert s["holes"] == []
    assert s["per_patch"]["cube"]["boundary"] == 0
    ng = grid.ng
    nface = (13 + 2 * ng) ** 2
    for name in ["shell+x", "shell-z"]:
        assert s["per_patch"][name]["boundary"] == ng * nface
