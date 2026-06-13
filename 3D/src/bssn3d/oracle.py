"""Dendro-GR bit-compare oracle for the transliterated BSSN RHS (the decisive
Phase-2.2 fidelity gate) — fully CPU-side, no octree/MPI, no 2FA.

The vendored CSE (``vendor/bssneqs_sympy_cse_wo_derivs.cpp``) takes the 138
derivatives **as inputs**, so its only external symbol is ``pow``. We therefore
compile it standalone: a generated C++ harness declares all 162 inputs (24 fields
+ 138 derivatives) + the gauge params, sets them to chosen values, ``#include``s
the CSE, and prints the 24 ``*_rhs`` outputs. The *same* input values are fed to
the JAX algebra (``_bssn_rhs_generated.bssn_rhs_algebra``) and the two are diffed.

This is a pure-algebra comparison (no finite differencing on either side):
derivative inputs are independent random scalars, so it isolates the
transliteration. Values cross the boundary as hex floats (``float.hex()`` /
``%a``) so there is no decimal round-trip; agreement is then limited only by
floating-point summation-order differences between g++ and XLA (~1e-12 relative),
which is the "compare to round-off" bar the plan calls for.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ._codegen import DENDRO_CSE, RHS_TO_FIELD
from ._bssn_rhs_generated import (
    FIELD_INPUTS, GRAD1_INPUTS, GRAD2_INPUTS, bssn_rhs_algebra,
)

GRAD1_NAMES = [f"grad_{a}_{f}" for a, f in GRAD1_INPUTS]
GRAD2_NAMES = [f"grad2_{i}_{j}_{f}" for i, j, f in GRAD2_INPUTS]
DERIV_NAMES = GRAD1_NAMES + GRAD2_NAMES
OUTPUT_TOKENS = list(RHS_TO_FIELD.keys())


def have_gpp() -> bool:
    return shutil.which("g++") is not None


# ---------------------------------------------------------------------------
# input generation
# ---------------------------------------------------------------------------

def random_inputs(seed: int = 0) -> Dict:
    """Random but *valid* single-point inputs (non-degenerate conformal metric).

    Returns ``{"fields": {...}, "derivs": {...}, "eta": ..., "lmbda": (...),
    "lambda_f": (...)}``. gt is a small perturbation of delta_ij and chi, alpha
    stay positive so the CSE's inverse-metric / 1-over-det terms are finite.
    """
    rng = np.random.default_rng(seed)

    fields: Dict[str, float] = {}
    for name in FIELD_INPUTS:
        if name in ("gt0", "gt3", "gt5"):          # diagonal conformal metric
            fields[name] = 1.0 + 0.1 * rng.uniform(-1, 1)
        elif name in ("gt1", "gt2", "gt4"):        # off-diagonal (small)
            fields[name] = 0.05 * rng.uniform(-1, 1)
        elif name == "chi":
            fields[name] = 1.0 + 0.2 * rng.uniform(0, 1)
        elif name == "alpha":
            fields[name] = 1.0 + 0.1 * rng.uniform(-1, 1)
        else:                                       # K, beta, At, Gt, B
            fields[name] = 0.1 * rng.uniform(-1, 1)

    derivs = {name: 0.05 * rng.uniform(-1, 1) for name in DERIV_NAMES}

    # distinct gauge / CAHD / SSL values so an index or param mix-up shows up. dt and
    # t are nonzero (CAHD divides by dt; t feeds the SSL Gaussian) so neither term is
    # accidentally masked in the bit-compare.
    return {
        "fields": fields,
        "derivs": derivs,
        "eta": 2.13,
        "lmbda": (1.10, 0.90, 1.20, 0.80),
        "lambda_f": (1.00, 0.50),
        "BSSN_CAHD_C": 0.06,
        "dt": 0.013,
        "dx_i": 0.027,
        "h_ssl": 0.6,
        "sig_ssl": 20.0,
        "t": 1.7,
    }


SCALARS = ["BSSN_CAHD_C", "dt", "dx_i", "h_ssl", "sig_ssl", "t"]


# ---------------------------------------------------------------------------
# C++ oracle
# ---------------------------------------------------------------------------

def _hex(x: float) -> str:
    return float(x).hex()


def emit_harness(values: Dict, cse_path: Path = DENDRO_CSE) -> str:
    """Generate the standalone C++ harness with inputs baked as hex literals."""
    f = values["fields"]
    d = values["derivs"]
    lm = values["lmbda"]
    lf = values["lambda_f"]

    L: List[str] = ["#include <cmath>", "#include <cstdio>", "", "int main() {",
                    "  const int pp = 0;"]
    for name in FIELD_INPUTS:
        L.append(f"  double {name}[1]; {name}[0] = {_hex(f[name])};")
    # The CAHD+SSL CSE indexes derivative inputs as grad_..._f[pp] (the earlier
    # no-CAHD variant used them bare), so declare them as length-1 arrays too.
    for name in DERIV_NAMES:
        L.append(f"  double {name}[1]; {name}[0] = {_hex(d[name])};")
    L.append(f"  double lambda[4] = {{{_hex(lm[0])}, {_hex(lm[1])}, "
             f"{_hex(lm[2])}, {_hex(lm[3])}}};")
    L.append(f"  double lambda_f[2] = {{{_hex(lf[0])}, {_hex(lf[1])}}};")
    L.append(f"  double eta = {_hex(values['eta'])};")
    for name in SCALARS:
        L.append(f"  double {name} = {_hex(values[name])};")
    for tok in OUTPUT_TOKENS:
        L.append(f"  double {tok}[1];")
    L.append(f'  #include "{cse_path}"')
    for tok in OUTPUT_TOKENS:
        L.append(f'  printf("{tok} %a\\n", {tok}[0]);')
    L.append("  return 0;")
    L.append("}")
    return "\n".join(L) + "\n"


def run_cpp_oracle(values: Dict, cse_path: Path = DENDRO_CSE) -> Dict[str, float]:
    """Compile + run the harness; return ``{output_token: value}``."""
    if not have_gpp():
        raise RuntimeError("g++ not found; cannot build the Dendro oracle.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "harness.cpp"
        exe = tmp / "harness"
        src.write_text(emit_harness(values, cse_path))
        # C++17: hex-float literals (bit-exact value exchange) are standard there.
        subprocess.run(["g++", "-O0", "-std=c++17", str(src), "-o", str(exe)],
                       check=True, capture_output=True, text=True)
        proc = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    out: Dict[str, float] = {}
    for line in proc.stdout.splitlines():
        tok, hexval = line.split()
        out[tok] = float.fromhex(hexval)
    return out


# ---------------------------------------------------------------------------
# JAX algebra (same inputs)
# ---------------------------------------------------------------------------

def run_jax_algebra(values: Dict) -> Dict[str, float]:
    """Evaluate the transliterated JAX algebra at the same single point."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    F = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["fields"].items()}
    D = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["derivs"].items()}
    out = bssn_rhs_algebra(
        F, D, values["eta"], values["lmbda"], values["lambda_f"],
        values["BSSN_CAHD_C"], values["dt"], values["dx_i"],
        values["h_ssl"], values["sig_ssl"], values["t"],
    )
    # map back to Dendro output tokens for a token-by-token comparison
    field_to_tok = {field: tok for tok, field in RHS_TO_FIELD.items()}
    return {field_to_tok[field]: float(np.asarray(arr)[0]) for field, arr in out.items()}


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare(seed: int = 0) -> Tuple[float, List[Tuple[str, float, float, float]]]:
    """Run both sides at ``seed``; return (max_rel_diff, per-output rows).

    Each row is ``(token, cpp, jax, rel_diff)`` with
    ``rel_diff = |cpp - jax| / max(|cpp|, |jax|, 1e-12)``.
    """
    cpp = run_cpp_oracle(seed_or_values(seed))
    jx = run_jax_algebra(seed_or_values(seed))
    rows = []
    worst = 0.0
    for tok in OUTPUT_TOKENS:
        c, j = cpp[tok], jx[tok]
        rel = abs(c - j) / max(abs(c), abs(j), 1e-12)
        rows.append((tok, c, j, rel))
        worst = max(worst, rel)
    return worst, rows


def seed_or_values(seed_or_dict):
    return seed_or_dict if isinstance(seed_or_dict, dict) else random_inputs(seed_or_dict)


def _main():
    print(f"Dendro-GR BSSN RHS bit-compare  (g++ available: {have_gpp()})\n")
    overall = 0.0
    for seed in range(3):
        worst, rows = compare(seed)
        overall = max(overall, worst)
        print(f"seed {seed}:  max relative diff = {worst:.3e}")
        # show the 3 worst outputs for this seed
        for tok, c, j, rel in sorted(rows, key=lambda r: -r[3])[:3]:
            print(f"    {tok:9s} cpp={c:+.16e}  jax={j:+.16e}  rel={rel:.2e}")
    print(f"\nOVERALL max relative diff across seeds: {overall:.3e}")


if __name__ == "__main__":
    _main()
