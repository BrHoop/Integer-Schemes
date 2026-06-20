"""Guard the Step 3.2f microbenchmark reference math (christoffel_ref.py).

The CUDA warp-cooperative kernel is validated against this reference, so the reference
itself must be correct: closed-form inverse == NumPy inverse, the metric/inverse identity
holds, and the Christoffels are jk-symmetric.
"""

import numpy as np
import pytest

from bssn3d.cuda import christoffel_ref as cr


def test_self_test_round_off():
    # closed-form inverse vs np.linalg + the g~^{il} g~_{lj} = delta identity
    assert cr.self_test(n=500, seed=3) < 1e-12


def test_inverse_identity():
    rng = np.random.default_rng(7)
    for _ in range(50):
        gt, _ = cr._random_point(rng)
        ig = cr._mat(cr.inverse_metric(gt))
        assert np.abs(ig @ cr._mat(gt) - np.eye(3)).max() < 1e-12


def test_christoffel_jk_symmetric():
    # Gt^i_{jk} = Gt^i_{kj}: the packed 6 pairs already assume this; verify against a
    # full-index recompute that does NOT assume symmetry.
    rng = np.random.default_rng(11)
    gt, dgt = cr._random_point(rng)
    ig = cr._mat(cr.inverse_metric(gt))
    d = [cr._mat(dgt[m * 6:(m + 1) * 6]) for m in range(3)]
    full = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                full[i, j, k] = 0.5 * sum(
                    ig[i, l] * (d[j][l, k] + d[k][l, j] - d[l][j, k]) for l in range(3))
    assert np.abs(full - np.transpose(full, (0, 2, 1))).max() < 1e-14
    packed = cr.christoffel(gt, dgt)
    for i in range(3):
        for p, (j, k) in enumerate(cr.PAIRS):
            assert packed[i * 6 + p] == pytest.approx(full[i, j, k], abs=1e-14)
