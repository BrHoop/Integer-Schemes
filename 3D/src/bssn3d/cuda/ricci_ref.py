"""Reference math + test-vector generator for the Step 3.2f Increment-1b microbenchmark.

1b extends 1a (inverse metric -> Christoffel) to the **conformal Ricci tensor** R~_{ij} =
Ricci(g~) — the first-AND-second-order trunk. This is the working set big enough to push a
naive 1-thread layout over the 255-register file (~128 co-resident fp64 temps: inverse,
18 Christoffels, 54 Christoffel derivatives, the second-derivative contractions, products),
so it is the TRUE register-fit go/no-go: does warp-cooperative g=4 stay register-resident
(0 spill) where naive spills?

Inputs per point (60 doubles):
  gt[6]                 g~_{ij}                          (xx xy xz yy yz zz)
  dgt[m*6 + c]          d_m g~_c          m in 0..2      (18)
  ddgt[symidx(m,n)*6+c] d_m d_n g~_c      mn symmetric   (36)
Output: ric[6] = R~_{ij} (symmetric, packed like gt).

R~_{ij} = d_l G^l_{ij} - d_j G^l_{il} + G^l_{lm} G^m_{ij} - G^l_{jm} G^m_{il}, with
G^l_{ij} = 1/2 g~^{lm}(d_i g~_{mj} + d_j g~_{mi} - d_m g~_{ij}) the conformal Christoffel,
d_k G^l_{ij} expanded via d_k g~^{lm} = -g~^{la} g~^{mb} d_k g~_{ab} and the 2nd derivs.

This module is the CPU ground truth (runs locally). It validates the closed-form pointwise
Ricci against an INDEPENDENT SymPy symbolic Ricci of an analytic metric, then writes raw-
float64 test vectors for `microbench_ricci.cu`.

Run:  `python -m bssn3d.cuda.ricci_ref`   (SymPy self-test + writes ricci_vectors.bin)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
_TAB = ((0, 1, 2), (1, 3, 4), (2, 4, 5))     # symidx(a,b)


def symidx(a: int, b: int) -> int:
    return _TAB[a][b]


def _mat(v6: np.ndarray) -> np.ndarray:
    g0, g1, g2, g3, g4, g5 = v6
    return np.array([[g0, g1, g2], [g1, g3, g4], [g2, g4, g5]])


def inverse_metric(gt: np.ndarray) -> np.ndarray:
    g0, g1, g2, g3, g4, g5 = gt
    c0 = g3 * g5 - g4 * g4
    c1 = g2 * g4 - g1 * g5
    c2 = g1 * g4 - g2 * g3
    c3 = g0 * g5 - g2 * g2
    c4 = g1 * g2 - g0 * g4
    c5 = g0 * g3 - g1 * g1
    idet = 1.0 / (g0 * c0 + g1 * c1 + g2 * c2)
    return np.array([c0, c1, c2, c3, c4, c5]) * idet


def ricci(gt: np.ndarray, dgt: np.ndarray, ddgt: np.ndarray) -> np.ndarray:
    """Conformal Ricci R~_{ij} from the metric + its 1st/2nd derivatives (the form the
    CUDA kernel ports). Returns the 6 independent components packed like gt."""
    ig = _mat(inverse_metric(gt))                       # g~^{lm}

    def dg(m, a, b):                                    # d_m g~_{ab}
        return dgt[m * 6 + symidx(a, b)]

    def ddg(m, n, a, b):                                # d_m d_n g~_{ab}
        return ddgt[symidx(m, n) * 6 + symidx(a, b)]

    # d_k g~^{lm} = - g~^{la} g~^{mb} d_k g~_{ab}
    dig = np.zeros((3, 3, 3))                            # [k][l][m]
    for k in range(3):
        for l in range(3):
            for m in range(3):
                s = 0.0
                for a in range(3):
                    for b in range(3):
                        s += ig[l, a] * ig[m, b] * dg(k, a, b)
                dig[k, l, m] = -s

    # Christoffel G^l_{ij}
    G = np.zeros((3, 3, 3))                              # [l][i][j]
    for l in range(3):
        for i in range(3):
            for j in range(3):
                s = 0.0
                for m in range(3):
                    s += ig[l, m] * (dg(i, m, j) + dg(j, m, i) - dg(m, i, j))
                G[l, i, j] = 0.5 * s

    # d_k G^l_{ij}
    dG = np.zeros((3, 3, 3, 3))                          # [k][l][i][j]
    for k in range(3):
        for l in range(3):
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for m in range(3):
                        s += dig[k, l, m] * (dg(i, m, j) + dg(j, m, i) - dg(m, i, j))
                        s += ig[l, m] * (ddg(k, i, m, j) + ddg(k, j, m, i) - ddg(k, m, i, j))
                    dG[k, l, i, j] = 0.5 * s

    out = np.zeros(6)
    for p, (i, j) in enumerate(PAIRS):
        r = 0.0
        for l in range(3):
            r += dG[l, l, i, j] - dG[j, l, i, l]
            for m in range(3):
                r += G[l, l, m] * G[m, i, j] - G[l, j, m] * G[m, i, l]
        out[p] = r
    return out


# ---------------------------------------------------------------------------
# SymPy oracle: symbolic Ricci of an analytic metric, evaluated at a point.
# Independent of the closed-form above (symbolic diff vs chain-rule expansion).
# ---------------------------------------------------------------------------
def _sympy_check(seed: int = 0, n: int = 6) -> float:
    import sympy as sp
    x = sp.symbols("x0 x1 x2")
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n):
        # analytic SPD-near-identity metric with non-trivial 2nd derivatives
        A = rng.normal(scale=0.12, size=(3, 3, 3))      # [comp-pair builder]
        def comp():
            a = rng.normal(scale=0.08, size=3)
            return (a[0] * sp.sin(x[0]) + a[1] * sp.cos(x[1]) + a[2] * sp.sin(x[2]))
        pert = [[comp() for _ in range(3)] for _ in range(3)]
        g = sp.Matrix(3, 3, lambda i, j: (1 if i == j else 0)
                      + (pert[i][j] + pert[j][i]) / 2)
        ig = g.inv()
        # symbolic Christoffel + Ricci
        def d(expr, k):
            return sp.diff(expr, x[k])
        Gam = [[[sp.Rational(1, 2) * sum(ig[l, m] * (d(g[m, j], i) + d(g[m, i], j)
                 - d(g[i, j], m)) for m in range(3)) for j in range(3)]
                for i in range(3)] for l in range(3)]
        Ric = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                r = 0
                for l in range(3):
                    r += d(Gam[l][i][j], l) - d(Gam[l][i][l], j)
                    for m in range(3):
                        r += Gam[l][l][m] * Gam[m][i][j] - Gam[l][j][m] * Gam[m][i][l]
                Ric[i, j] = r
        pt = {x[k]: float(rng.uniform(-1, 1)) for k in range(3)}
        # evaluate g, dg, ddg at the point -> feed the numpy reference
        gv = np.array([float(g[a, b].subs(pt)) for a, b in PAIRS])
        dgv = np.array([float(d(g[a, b], m).subs(pt))
                        for m in range(3) for a, b in PAIRS])
        ddgv = np.array([float(d(d(g[a, b], mn[0]), mn[1]).subs(pt))
                         for mn in PAIRS for a, b in PAIRS])
        ric_num = ricci(gv, dgv, ddgv)
        ric_sym = np.array([float(Ric[a, b].subs(pt)) for a, b in PAIRS])
        max_err = max(max_err, np.abs(ric_num - ric_sym).max())
    return max_err


# ---------------------------------------------------------------------------
# Test vectors. Generating valid (g, dg, ddg) that are a CONSISTENT jet of a real
# field matters for physical magnitudes, so we sample an analytic field and read off
# its derivatives (random independent ddg would be unphysically large).
# ---------------------------------------------------------------------------
OUT = Path(__file__).resolve().parent / "ricci_vectors.bin"


def _field_jet(rng):
    """Sample g~, d g~, dd g~ as the 2-jet of an analytic conformal metric at a point."""
    # g(x) = I + sum_k S_k * basis_k(x), basis with known 1st/2nd derivatives
    freqs = rng.uniform(0.5, 1.5, size=(3, 3))
    phase = rng.uniform(0, 2 * np.pi, size=(3, 3))
    amp = rng.normal(scale=0.06, size=(3, 6))           # per-direction, per-comp
    xpt = rng.uniform(-1, 1, size=3)
    gt = np.zeros(6); dgt = np.zeros(18); ddgt = np.zeros(36)
    base = np.array([1.0, 0, 0, 1.0, 0, 1.0])           # identity metric
    for c in range(6):
        val = base[c]; dval = np.zeros(3); ddval = np.zeros((3, 3))
        for m in range(3):
            w = freqs[m % 3, c % 3]
            s = np.sin(w * xpt[m] + phase[m % 3, c % 3])
            cs = np.cos(w * xpt[m] + phase[m % 3, c % 3])
            a = amp[m, c]
            val += a * s
            dval[m] += a * w * cs
            ddval[m, m] += -a * w * w * s
        gt[c] = val
        for m in range(3):
            dgt[m * 6 + c] = dval[m]
        for q, (m, n) in enumerate(PAIRS):
            ddgt[q * 6 + c] = ddval[m, n]
    # guard SPD
    if np.linalg.eigvalsh(_mat(gt)).min() <= 0.1:
        return _field_jet(rng)
    return gt, dgt, ddgt


def write_test_vectors(npts: int = 1 << 18, seed: int = 1, out: Path = OUT) -> Path:
    rng = np.random.default_rng(seed)
    gt = np.empty((npts, 6)); dgt = np.empty((npts, 18))
    ddgt = np.empty((npts, 36)); ric = np.empty((npts, 6))
    for n in range(npts):
        g, d, dd = _field_jet(rng)
        gt[n], dgt[n], ddgt[n] = g, d, dd
        ric[n] = ricci(g, d, dd)
    with open(out, "wb") as f:
        np.array([npts], dtype=np.int64).tofile(f)
        np.hstack([gt, dgt, ddgt]).astype(np.float64).tofile(f)   # (npts, 60)
        ric.astype(np.float64).tofile(f)                          # (npts, 6)
    return out


def main() -> None:
    print(">> ricci_ref self-test (closed-form pointwise Ricci vs SymPy symbolic Ricci)")
    err = _sympy_check(seed=2, n=6)
    print(f">> max error over 6 analytic metrics: {err:.2e}")
    assert err < 1e-9, "Ricci reference disagrees with SymPy — fix before vectors"
    p = write_test_vectors(npts=1 << 18)
    sz = p.stat().st_size / 1e6
    print(f">> wrote {p.name}: {1 << 18} points, {sz:.1f} MB "
          f"(int64 count + 60 inputs + 6 Ricci, float64)")


if __name__ == "__main__":
    main()
