"""Ghost-zone sync: root stitch (incl. edges/corners), cross-level halo fill."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from multipatch.amr.state import BS, NG, NF
from multipatch.amr import kernels as K
from multipatch.amr import sync as S

W = BS + 2 * NG


def test_root_stitch_fills_interblock_halos():
    nbx = nby = nbz = 2
    n_root = nbx * nby * nbz
    rng = np.random.default_rng(1)
    gint = rng.standard_normal((NF, nbx*BS, nby*BS, nbz*BS))     # global interior
    gpad = np.pad(gint, ((0, 0), (NG, NG), (NG, NG), (NG, NG)), mode='edge')

    # interiors-only storage (BS³ per slot); the haloed buffer is rebuilt
    interiors = np.zeros((n_root, NF, BS, BS, BS))
    for bi in range(nbx):
        for bj in range(nby):
            for bk in range(nbz):
                slot = (bi*nby + bj)*nbz + bk
                interiors[slot] = \
                    gint[:, bi*BS:bi*BS+BS, bj*BS:bj*BS+BS, bk*BS:bk*BS+BS]

    out = np.asarray(S.sync_within_level_root(jnp.asarray(interiors), nbx, nby, nbz))

    for bi in range(nbx):
        for bj in range(nby):
            for bk in range(nbz):
                slot = (bi*nby + bj)*nbz + bk
                exp = gpad[:, bi*BS:bi*BS+W, bj*BS:bj*BS+W, bk*BS:bk*BS+W]
                # full block (incl. faces/edges/corners) must match edge-padded global
                assert np.max(np.abs(out[slot] - exp)) < 1e-13


def test_across_levels_fills_child_halo():
    rng = np.random.default_rng(2)
    parent = jnp.asarray(rng.standard_normal((NF, W, W, W)))
    corner = (1, 0, 1)
    expected_child = np.asarray(K.prolongate(parent, corner))

    # one child slot whose interior is preset to a sentinel (must be preserved)
    child = np.full((1, NF, W, W, W), -7.0)
    child_blocks = jnp.asarray(child)
    parent_blocks = parent[None]
    parent_slot = jnp.asarray([0], jnp.int32)
    child_c = jnp.asarray([corner], jnp.int32)
    active = jnp.asarray([True])

    out = np.asarray(S.sync_across_levels(
        child_blocks, parent_blocks, parent_slot, child_c, active))[0]

    # interior preserved (sentinel), halo = prolongated parent
    assert np.all(out[:, NG:NG+BS, NG:NG+BS, NG:NG+BS] == -7.0)
    halo = np.ones((W, W, W), bool); halo[NG:NG+BS, NG:NG+BS, NG:NG+BS] = False
    assert np.max(np.abs(out[:, halo] - expected_child[:, halo])) < 1e-13


def test_across_levels_inactive_untouched():
    rng = np.random.default_rng(5)
    parent = jnp.asarray(rng.standard_normal((NF, W, W, W)))
    child = jnp.asarray(rng.standard_normal((1, NF, W, W, W)))
    out = np.asarray(S.sync_across_levels(
        child, parent[None], jnp.asarray([0], jnp.int32),
        jnp.asarray([[0, 0, 0]], jnp.int32), jnp.asarray([False])))
    assert np.array_equal(out, np.asarray(child))
