"""Transliterate Dendro-GR's generated BSSN CSE C++ -> a JAX algebra module.

Source (Phase-3 production variant): ``Dendro-GR/CodeGen/bssneqs_SSL_HD_dxsq.cpp``
— the **CAHD + SSL** RHS (Hamiltonian-constraint damping on chi, spatial
slice-locking on the lapse), a flat SSA block of ``const double DENDRO_N =
<expr>;`` statements over fields, the 138 derivative inputs, the gauge params and
the CAHD/SSL scalars, ending in 24 ``*_rhs[pp] = ...`` assignments. Derivatives are
*inputs* (the ``_wo_derivs`` design), so this transform is pure algebra — no FD,
no SymPy.

(Phase 2 used the simpler no-CAHD ``bssneqs_sympy_cse_wo_derivs.cpp``. Phase 3.0
locks the production variant to CAHD+SSL — what long runs actually need — and
re-validates it through this same pipeline + the oracle + apples gates.)

Why textual (not a full parser): after stripping ``//`` comments the body is pure
arithmetic — ``+ - * /``, balanced parens, integer/float literals, and three
functions: ``pow(x, y)`` (Python's builtin ``pow`` evaluates it on JAX arrays),
``sqrt`` and ``exp`` (mapped to ``jnp.sqrt`` / ``jnp.exp``). The only token that is
not already a valid Python identifier is the keyword ``lambda``; everything else
(``DENDRO_*``, ``grad_*``, ``grad2_*``, field names, the scalar params) is verbatim.
So the transform is:

    1. strip ``//`` comments,
    2. drop ``[pp]`` indexing (each ``field[pp]`` becomes a JAX array variable),
    3. drop ``const double`` (every temp is a Python local),
    4. rename the keyword ``lambda`` -> ``lmbda`` (``lambda_f`` is untouched —
       no word boundary splits it),
    5. ``sqrt(`` -> ``jnp.sqrt(``, ``exp(`` -> ``jnp.exp(``.

emitting one Python statement per SSA line, in order (SSA => define-before-use is
already satisfied). The generated module is committed and reviewable.

Run:  ``python -m bssn3d._codegen``  (regenerates ``_bssn_rhs_generated.py``).
"""

import hashlib
import re
from datetime import date
from pathlib import Path

# Vendored Dendro-GR source (in-repo, so nothing reads ~/Code/Dendro-GR at
# runtime). See vendor/README.md for provenance / how to refresh.
DENDRO_CSE = Path(__file__).resolve().parent / "vendor" / \
    "bssneqs_SSL_HD_dxsq.cpp"

OUT_PATH = Path(__file__).resolve().parent / "_bssn_rhs_generated.py"

# Scalar parameters the CAHD+SSL variant consumes, in RHS-signature order after
# (eta, lmbda, lambda_f). eta/lmbda/lambda_f are the original gauge knobs; the rest
# are new in this variant: CAHD strength + the dx^2/dt damping factor, and the SSL
# amplitude/width + current time t (the Gaussian slice-locking ramp). The
# drift-guard in ``generate`` asserts the vendored file references exactly these.
SCALAR_PARAMS = ["eta", "BSSN_CAHD_C", "dt", "dx_i", "h_ssl", "sig_ssl", "t"]
INDEXED_PARAMS = ["lmbda", "lambda_f"]   # lambda[0..3] (renamed) and lambda_f[0..1]

# Dendro RHS output token -> our state field name (b_rhs = shift beta; B_rhs = aux B).
RHS_TO_FIELD = {
    "a_rhs": "alpha", "chi_rhs": "chi", "K_rhs": "K",
    "gt_rhs00": "gt0", "gt_rhs01": "gt1", "gt_rhs02": "gt2",
    "gt_rhs11": "gt3", "gt_rhs12": "gt4", "gt_rhs22": "gt5",
    "b_rhs0": "beta0", "b_rhs1": "beta1", "b_rhs2": "beta2",
    "At_rhs00": "At0", "At_rhs01": "At1", "At_rhs02": "At2",
    "At_rhs11": "At3", "At_rhs12": "At4", "At_rhs22": "At5",
    "Gt_rhs0": "Gt0", "Gt_rhs1": "Gt1", "Gt_rhs2": "Gt2",
    "B_rhs0": "B0", "B_rhs1": "B1", "B_rhs2": "B2",
}

FIELD_INPUTS = [
    "alpha", "chi", "K",
    "gt0", "gt1", "gt2", "gt3", "gt4", "gt5",
    "beta0", "beta1", "beta2",
    "At0", "At1", "At2", "At3", "At4", "At5",
    "Gt0", "Gt1", "Gt2",
    "B0", "B1", "B2",
]

_GRAD1_RE = re.compile(r"\bgrad_(\d)_([A-Za-z][A-Za-z0-9]*)\b")
_GRAD2_RE = re.compile(r"\bgrad2_(\d)_(\d)_([A-Za-z][A-Za-z0-9]*)\b")
_LAMBDA_RE = re.compile(r"\blambda\b")   # bare keyword only; 'lambda_f' has no \b split
_SQRT_RE = re.compile(r"\bsqrt\(")
_EXP_RE = re.compile(r"\bexp\(")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

# pow(<identifier>, <integer>) — the only pow shape Dendro's CSE emits (exponents {2,-2,-3}
# on bare temps/fields). Anything else trips the guard in `lower_pow`.
_POW_RE = re.compile(r"\bpow\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(-?\d+)\s*\)")
_POW_ANY_RE = re.compile(r"\bpow\(")


def lower_pow(expr: str) -> str:
    """Rewrite ``pow(x, n)`` (integer n) into explicit multiplies / reciprocals.

    Step 3.4 item-2 free win. In CUDA C++ ``pow(double, double)`` is a libcall computed as
    ``exp(n*log(x))`` — ~20-50x a multiply *and* less accurate than ``x*x``; lowering is both
    faster and more accurate. Dendro's CSE only emits the simple form
    ``pow(<identifier>, {2,-2,-3})``; this raises if it ever sees a ``pow`` it cannot lower
    (a nested-arg or fractional exponent) so we never silently emit a slow libcall.

        pow(x,  n>0) -> (x*x*...*x)         [n factors]
        pow(x,  n<0) -> (1.0/(x*x*...*x))   [|n| factors]

    n==0 would be a constant 1.0 (not expected from Dendro; handled for completeness).
    """
    def _repl(m: re.Match) -> str:
        base, n = m.group(1), int(m.group(2))
        if n == 0:
            return "1.0"
        factors = "*".join([base] * abs(n))
        body = factors if abs(n) == 1 else f"({factors})"
        return body if n > 0 else f"(1.0/{body})"

    out = _POW_RE.sub(_repl, expr)
    if _POW_ANY_RE.search(out):
        raise AssertionError(
            f"lower_pow: unlowerable pow() left in expression (non-identifier base or "
            f"fractional/non-integer exponent): {out!r}"
        )
    return out


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _statements(code: str):
    """Yield (lhs, rhs) for each ``;``-terminated assignment, in order."""
    for raw in code.split(";"):
        s = raw.strip()
        if not s:
            continue
        s = s.replace("[pp]", "")            # field[pp]/out[pp] -> field/out
        s = s.replace("const double ", "")   # every temp is a Python local
        lhs, rhs = s.split("=", 1)
        yield lhs.strip(), rhs.strip()


def parse(src: Path = DENDRO_CSE):
    """Return (statements, grad1, grad2) parsed from the Dendro CSE file.

    grad1 = sorted list of (axis, field); grad2 = sorted list of (i, j, field).
    """
    text = src.read_text()
    code = _strip_comments(text)
    statements = list(_statements(code))

    grad1 = sorted({(int(a), f) for a, f in _GRAD1_RE.findall(text)})
    grad2 = sorted({(int(i), int(j), f) for i, j, f in _GRAD2_RE.findall(text)})
    return statements, grad1, grad2


def _translate_rhs(expr: str) -> str:
    """C++ arithmetic expression -> Python/JAX.

    ``lambda`` (keyword) -> ``lmbda``; ``sqrt(``/``exp(`` -> ``jnp.sqrt(``/``jnp.exp(``.
    ``pow`` is left as-is (Python's builtin evaluates it on JAX arrays).
    """
    expr = _LAMBDA_RE.sub("lmbda", expr)
    expr = _SQRT_RE.sub("jnp.sqrt(", expr)
    expr = _EXP_RE.sub("jnp.exp(", expr)
    return expr


def _validate_scalar_params(src: Path, statements, grad1, grad2):
    """Drift guard: confirm the vendored file references exactly the scalar params
    we hard-code in ``SCALAR_PARAMS``/``INDEXED_PARAMS`` — nothing more, nothing
    less. A refreshed/changed CSE that introduces a new scalar (or drops one) trips
    this immediately, so the transliteration can never silently ignore a parameter.
    """
    text = _strip_comments(src.read_text())
    idents = set(_IDENT_RE.findall(text))

    known = set()
    known |= {f"DENDRO_{i}" for i in range(100000)
              if f"DENDRO_{i}" in idents}            # CSE temps
    known |= set(FIELD_INPUTS)                        # 24 fields
    known |= set(RHS_TO_FIELD)                        # 24 output tokens
    known |= {f"grad_{a}_{f}" for a, f in grad1}
    known |= {f"grad2_{i}_{j}_{f}" for i, j, f in grad2}
    known |= {"pow", "sqrt", "exp", "const", "double", "pp", "lambda", "lambda_f"}

    leftover = {i for i in idents if i not in known and not _GRAD1_RE.fullmatch(i)
                and not _GRAD2_RE.fullmatch(i)}
    # the bare scalar params we expect (eta + CAHD/SSL scalars; lambda/lambda_f are
    # in `known`). `lmbda` is our rename, not in the source.
    expected = set(SCALAR_PARAMS)
    if leftover != expected:
        raise AssertionError(
            f"scalar-param drift in {src.name}: "
            f"unexpected={sorted(leftover - expected)} "
            f"missing={sorted(expected - leftover)}"
        )


def _emit_module(statements, grad1, grad2, src_hash, *, src_name: str,
                 cut_set=None) -> str:
    """Emit the JAX RHS module text.

    ``cut_set=None`` → the verbatim module (the bit-validated oracle anchor; its
    bytes must never change). A non-None ``cut_set`` (a set of DENDRO_* temp
    names) emits the **staged** variant: identical algebra, but each cut temp is
    pinned with ``jax.lax.optimization_barrier`` immediately after its definition
    so XLA cannot re-CSE/re-fuse it into the giant spilling pointwise kernel —
    the Step 3.1 controllability probe. See ``bssn3d.staging.generate_staged``.
    """
    staged = cut_set is not None
    grad1_names = [f"grad_{a}_{f}" for a, f in grad1]
    grad2_names = [f"grad2_{i}_{j}_{f}" for i, j, f in grad2]

    L = []
    w = L.append
    w('"""AUTO-GENERATED by bssn3d._codegen — DO NOT EDIT BY HAND.')
    w("")
    if staged:
        w(f"STAGED variant — {len(cut_set)} optimization_barrier cut points pinned")
        w("on the high-fan-out tensor-hierarchy temps (Step 3.1 probe: can XLA be")
        w("forced register-bounded?). Same algebra as the verbatim module; barriers")
        w("only constrain fusion. Verbatim stays the oracle reference.")
        w("")
    w(f"Transliterated from Dendro-GR  {src_name}")
    w(f"  sha256[:16] = {src_hash}")
    w(f"  generated   = {date.today().isoformat()}")
    w(f"  statements  = {len(statements)}   grad1 = {len(grad1)}   grad2 = {len(grad2)}")
    w("")
    if staged:
        w("Regenerate:  python -m bssn3d.staging")
    else:
        w("Regenerate:  python -m bssn3d._codegen")
    w('"""')
    w("")
    if staged:
        w("import jax")
    w("import jax.numpy as jnp")
    w("")
    w("# Field inputs, derivative inputs, and the output field order, exported so")
    w("# the derivative-bundle builder and the RHS wrapper stay in lockstep.")
    w(f"FIELD_INPUTS = {FIELD_INPUTS!r}")
    w(f"GRAD1_INPUTS = {[list(g) for g in grad1]!r}")
    w(f"GRAD2_INPUTS = {[list(g) for g in grad2]!r}")
    w(f"OUTPUT_FIELDS = {[RHS_TO_FIELD[t] for t in RHS_TO_FIELD]!r}")
    w(f"SCALAR_PARAMS = {SCALAR_PARAMS!r}")
    if staged:
        w(f"CUT_SET = {sorted(cut_set)!r}")
    w("")
    w("")
    w("def bssn_rhs_algebra(F, D, eta, lmbda, lambda_f, "
      "BSSN_CAHD_C, dt, dx_i, h_ssl, sig_ssl, t):")
    w('    """Pure BSSN RHS algebra, CAHD+SSL variant (derivatives supplied as inputs).')
    w("")
    w("    F: dict field-name -> array (24 entries, FIELD_INPUTS).")
    w("    D: dict deriv-name -> array (GRAD1_INPUTS + GRAD2_INPUTS).")
    w("    eta: float;  lmbda: indexable len-4 (lambda[0..3]);")
    w("    lambda_f: indexable len-2 (lambda_f[0..1]).")
    w("    BSSN_CAHD_C: Hamiltonian-constraint-damping strength (chi RHS).")
    w("    dt: time step;  dx_i: grid spacing (the dx^2/dt CAHD factor).")
    w("    h_ssl, sig_ssl: SSL amplitude + Gaussian time-width;  t: current time")
    w("        (the lapse slice-locking ramp exp(-t^2 / 2 sig_ssl^2)).")
    w("    Returns dict field-name -> d/dt array (OUTPUT_FIELDS).")
    w('    """')
    # bind field inputs
    for name in FIELD_INPUTS:
        w(f'    {name} = F["{name}"]')
    # bind derivative inputs
    for name in grad1_names + grad2_names:
        w(f'    {name} = D["{name}"]')
    w("")
    # the SSA body + outputs, in file order
    for lhs, rhs in statements:
        py = _translate_rhs(rhs)
        w(f"    {lhs} = {py}")
        if staged and lhs in cut_set:
            w(f"    {lhs} = jax.lax.optimization_barrier({lhs})")
    w("")
    w("    return {")
    for tok, field in RHS_TO_FIELD.items():
        w(f'        "{field}": {tok},')
    w("    }")
    return "\n".join(L) + "\n"


def generate(src: Path = DENDRO_CSE, out: Path = OUT_PATH) -> Path:
    statements, grad1, grad2 = parse(src)
    _validate_scalar_params(src, statements, grad1, grad2)
    src_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    out.write_text(_emit_module(statements, grad1, grad2, src_hash,
                                src_name=src.name, cut_set=None))
    return out


if __name__ == "__main__":
    p = generate()
    print(f">> wrote {p}")
