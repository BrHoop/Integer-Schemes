"""Reference math + test-vector generator for the Step 3.2f Increment-1 microbenchmark.

The microbenchmark computes the **conformal inverse metric** g~^{ij} and the **conformal
Christoffel symbols** Gt^i_{jk} from the 6 metric components g~_{ij} and their 18 first
derivatives d_m g~_{ij} — the densest first-order block of the BSSN trunk. The CUDA kernel
(`microbench_christoffel.cu`) must reproduce these to round-off.

This module is the CPU ground truth (it runs locally; CUDA does not). It:
  * computes the inverse + Christoffels two independent ways and cross-checks them,
  * verifies the algebraic identities (g~^{il} g~_{lj} = delta, jk-symmetry),
  * writes raw-float64 binary test vectors the CUDA program reads back.

Index convention (matches `evolve.py`): the 6 symmetric components are
  gt0=g_xx gt1=g_xy gt2=g_xz gt3=g_yy gt4=g_yz gt5=g_zz
i.e. the matrix [[gt0,gt1,gt2],[gt1,gt3,gt4],[gt2,gt4,gt5]]. The 18 derivatives are
  dgt[m*6 + c]  =  d_m (gt_c)   for direction m in {0,1,2}, component c in {0..5}.
The 18 Christoffel outputs are ordered
  gam[i*6 + p]  =  Gt^i_{jk}     for i in {0,1,2}, p indexing the 6 symmetric (j,k) pairs
  PAIRS = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)].

Run:  `python -m bssn3d.cuda.christoffel_ref`   (self-test + writes test_vectors.bin)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# symmetric (j,k) pairs and the 3x3 component<->pair lookup
PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
_SYM = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (0, 2): 2, (2, 0): 2,
        (1, 1): 3, (1, 2): 4, (2, 1): 4, (2, 2): 5}


def _mat(gt: np.ndarray) -> np.ndarray:
    """6-vector -> symmetric 3x3."""
    g0, g1, g2, g3, g4, g5 = gt
    return np.array([[g0, g1, g2], [g1, g3, g4], [g2, g4, g5]])


def inverse_metric(gt: np.ndarray) -> np.ndarray:
    """Closed-form inverse of the symmetric 3x3 metric (the form the CUDA kernel ports).

    General determinant (NOT assuming det=1) so the kernel is correct on un-rescaled
    data. Returns the 6 independent components of g~^{ij} in the same packing as gt."""
    g0, g1, g2, g3, g4, g5 = gt
    # cofactors (adjugate of the symmetric matrix)
    c0 = g3 * g5 - g4 * g4          # (0,0)
    c1 = g2 * g4 - g1 * g5          # (0,1)
    c2 = g1 * g4 - g2 * g3          # (0,2)
    c3 = g0 * g5 - g2 * g2          # (1,1)
    c4 = g1 * g2 - g0 * g4          # (1,2)
    c5 = g0 * g3 - g1 * g1          # (2,2)
    det = g0 * c0 + g1 * c1 + g2 * c2
    idet = 1.0 / det
    return np.array([c0, c1, c2, c3, c4, c5]) * idet


def christoffel(gt: np.ndarray, dgt: np.ndarray) -> np.ndarray:
    """Gt^i_{jk} = 1/2 g~^{il} (d_j g~_{lk} + d_k g~_{lj} - d_l g~_{jk}).

    `dgt` is the 18-vector d_m gt_c (m*6+c). Returns the 18 Christoffels (i*6 + pair)."""
    igt = _mat(inverse_metric(gt))
    # d[m] = symmetric 3x3 of the m-direction derivative
    d = [_mat(dgt[m * 6:(m + 1) * 6]) for m in range(3)]

    def dg(m, a, b):                                # d_m g~_{ab}
        return d[m][a, b]

    out = np.zeros(18)
    for i in range(3):
        for p, (j, k) in enumerate(PAIRS):
            s = 0.0
            for l in range(3):
                s += igt[i, l] * (dg(j, l, k) + dg(k, l, j) - dg(l, j, k))
            out[i * 6 + p] = 0.5 * s
    return out


# ---------------------------------------------------------------------------
# Validation (independent computations must agree)
# ---------------------------------------------------------------------------
def _christoffel_numpy(gt: np.ndarray, dgt: np.ndarray) -> np.ndarray:
    """Same quantity via `np.linalg.inv` + dense loops — an independent oracle."""
    g = _mat(gt)
    ig = np.linalg.inv(g)
    d = [_mat(dgt[m * 6:(m + 1) * 6]) for m in range(3)]
    out = np.zeros(18)
    for i in range(3):
        for p, (j, k) in enumerate(PAIRS):
            s = 0.0
            for l in range(3):
                s += 0.5 * ig[i, l] * (d[j][l, k] + d[k][l, j] - d[l][j, k])
            out[i * 6 + p] = s
    return out


def _random_point(rng: np.random.Generator):
    """A physically-plausible conformal metric (SPD, det~1) + small derivatives."""
    a = rng.normal(scale=0.15, size=(3, 3))
    sym = 0.5 * (a + a.T)
    g = np.eye(3) + sym
    # ensure SPD; nudge toward det~1 like the conformal metric
    while np.linalg.eigvalsh(g).min() <= 0.05:
        g = 0.5 * (g + np.eye(3))
    g = g / np.cbrt(np.linalg.det(g))               # det -> 1 (conformal)
    gt = np.array([g[0, 0], g[0, 1], g[0, 2], g[1, 1], g[1, 2], g[2, 2]])
    dgt = rng.normal(scale=0.2, size=18)
    return gt, dgt


def self_test(n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n):
        gt, dgt = _random_point(rng)
        # identity: g~^{il} g~_{lj} = delta
        ig = _mat(inverse_metric(gt))
        ident_err = np.abs(ig @ _mat(gt) - np.eye(3)).max()
        # closed-form vs numpy oracle
        c_closed = christoffel(gt, dgt)
        c_numpy = _christoffel_numpy(gt, dgt)
        err = max(ident_err, np.abs(c_closed - c_numpy).max())
        max_err = max(max_err, err)
    return max_err


# ---------------------------------------------------------------------------
# Test-vector file (raw little-endian float64, the CUDA program reads it back)
# ---------------------------------------------------------------------------
#   layout:  int64 npts | npts*(6 gt + 18 dgt) inputs | npts*18 expected gam
OUT = Path(__file__).resolve().parent / "test_vectors.bin"


def write_test_vectors(npts: int = 1 << 20, seed: int = 1, out: Path = OUT) -> Path:
    rng = np.random.default_rng(seed)
    gt = np.empty((npts, 6))
    dgt = np.empty((npts, 18))
    gam = np.empty((npts, 18))
    for n in range(npts):
        g, d = _random_point(rng)
        gt[n], dgt[n] = g, d
        gam[n] = christoffel(g, d)
    with open(out, "wb") as f:
        np.array([npts], dtype=np.int64).tofile(f)
        np.hstack([gt, dgt]).astype(np.float64).tofile(f)   # (npts, 24)
        gam.astype(np.float64).tofile(f)                    # (npts, 18)
    return out


def main() -> None:
    err = self_test()
    print(f">> christoffel_ref self-test (closed-form vs numpy + delta identity)")
    print(f">> max error over 2000 random conformal points: {err:.2e}")
    assert err < 1e-12, "reference math disagrees — fix before generating vectors"
    # a small vector file for quick CUDA iteration; bump npts for the timing run.
    p = write_test_vectors(npts=1 << 18)
    sz = p.stat().st_size / 1e6
    print(f">> wrote {p.name}: {1 << 18} points, {sz:.1f} MB "
          f"(int64 count + (6+18) inputs + 18 expected, float64)")


if __name__ == "__main__":
    main()
