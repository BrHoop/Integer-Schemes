"""Step 3.4 / C1 — e-graph saturation PROBE (the go/no-go before the intensive item-5 build).

Question: does equality saturation expose a lower-register-pressure form of the BSSN algebra, on the
broad-plateau DAG the fusion model found? This runs ONE output cone through egglog and measures peak
register liveness BEFORE vs AFTER saturate + (op-count) DAG-extraction — the Architecture-A pipeline
(B1) on a single cone:

  cone SSA -> egglog e-graph -> saturate (float-safe ruleset) -> serialize -> my own min-op-count
  DAG-extraction WITH sharing (egglog's tree-extract would explode) -> re-CSE to flat SSA ->
  score peak-live with the same staging machinery as the baseline.

Decision rule: if extracted peak-live drops meaningfully (toward the 255-reg cliff in the full DAG),
GREEN for item 5; if the plateau resists, that is the negative result -> pivot to MSRK + Ozaki.

Run:  python -m bssn3d.egraph_probe --output K_rhs --iters 12
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

CONST_MAP: Dict[str, str] = {}   # c_<sanitized> -> original numeric literal (for emission/eval)

from egglog import EGraph, Expr, StringLike, rewrite, ruleset, vars_

from ._codegen import parse, DENDRO_CSE, lower_pow, _LAMBDA_RE
from .staging import build_dag, min_liveness_order, straight_line_liveness


# --- egglog term algebra -------------------------------------------------------------------
class M(Expr):
    @classmethod
    def var(cls, name: StringLike) -> "M": ...
    def __add__(self, o: "M") -> "M": ...
    def __sub__(self, o: "M") -> "M": ...
    def __mul__(self, o: "M") -> "M": ...
    def __truediv__(self, o: "M") -> "M": ...
    def __neg__(self) -> "M": ...


def _float_safe_ruleset():
    a, b, c = vars_("a b c", M)
    return ruleset(
        rewrite(a + b).to(b + a),                       # commutativity
        rewrite(a * b).to(b * a),
        rewrite((a + b) + c).to(a + (b + c)),           # associativity (both directions)
        rewrite(a + (b + c)).to((a + b) + c),
        rewrite((a * b) * c).to(a * (b * c)),
        rewrite(a * (b * c)).to((a * b) * c),
        rewrite(a * (b + c)).to(a * b + a * c),         # distribute / factor (exposes sharing)
        rewrite(a * b + a * c).to(a * (b + c)),
        rewrite(a * (b - c)).to(a * b - a * c),
        rewrite(a * b - a * c).to(a * (b - c)),
    )
    # NB: NO x*0->0 / no reassoc-across-cancellation (float-unsafe); consts are opaque leaves.


# --- cone extraction -----------------------------------------------------------------------
def cone_statements(dag, stmt: Dict[str, str], output: str) -> List[Tuple[str, str]]:
    """All transitive DENDRO-temp ancestors of `output` + output, in file (topological) order."""
    seen, stack = set(), [output]
    while stack:
        n = stack.pop()
        for d in dag.deps[n]:
            if d in stmt and d not in seen:
                seen.add(d)
                stack.append(d)
    seen.add(output)
    return [(n, stmt[n]) for n in dag.order if n in seen]


# --- SSA rhs string -> egglog M expr -------------------------------------------------------
def _to_egg(node, env: Dict[str, M]) -> M:
    if isinstance(node, ast.Expression):
        return _to_egg(node.body, env)
    if isinstance(node, ast.BinOp):
        l, r = _to_egg(node.left, env), _to_egg(node.right, env)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r
        raise ValueError(f"unsupported binop {ast.dump(node)}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_to_egg(node.operand, env)
    if isinstance(node, ast.Name):
        e = env.get(node.id)                             # cone temp -> its expr; else leaf var
        return e if e is not None else M.var(node.id)
    if isinstance(node, ast.Constant):
        name = "c_" + str(node.value).replace(".", "p").replace("-", "m")
        CONST_MAP[name] = repr(node.value)
        return M.var(name)
    raise ValueError(f"unsupported node {ast.dump(node)}")


def build_root(cone: List[Tuple[str, str]]) -> M:
    env: Dict[str, M] = {}
    for lhs, rhs in cone:
        tree = ast.parse(lower_pow(_LAMBDA_RE.sub("lmbda", rhs)), mode="eval")
        env[lhs] = _to_egg(tree, env)
    return env[cone[-1][0]]


# --- my own min-op-count DAG extraction over the serialized e-graph -------------------------
_BINOPS = {" + ": "+", " - ": "-", " * ": "*", " / ": "/"}


def _node_kind(op: str, nchildren: int):
    """Return ('leaf', name) | ('binop', sym) | ('neg', None) | ('str', literal)."""
    if op == "M.var":
        return ("var", None)
    if op.startswith('"'):
        return ("str", op.strip('"'))
    if nchildren == 1:
        return ("neg", None)
    for token, sym in _BINOPS.items():
        if token in op:
            return ("binop", sym)
    raise ValueError(f"unknown op {op!r} (n={nchildren})")


def extract(graph: dict, root_eclass: str, mode: str = "ops"):
    """Bottom-up additive extraction, DAG with sharing, fixpoint relaxation (handles cycles).
    mode='ops' -> min total op-count (cost = 1 + Σ children); mode='depth' -> min critical-path
    (cost = 1 + max children) -> balanced trees, shorter dependency chains. Returns chosen map."""
    nodes = graph["nodes"]
    by_class: Dict[str, List[str]] = {}
    for nid, nd in nodes.items():
        by_class.setdefault(nd["eclass"], []).append(nid)

    INF = float("inf")
    cost = {ec: INF for ec in by_class}
    chosen: Dict[str, str] = {}
    for _ in range(400):                                # relaxation sweeps
        changed = False
        for ec, nids in by_class.items():
            best, best_c = None, INF
            for nid in nids:
                nd = nodes[nid]
                kind, _ = _node_kind(nd["op"], len(nd["children"]))
                base = 0.0 if kind in ("var", "str") else 1.0
                child_costs, ok = [], True
                for ch in nd["children"]:
                    cc = cost[nodes[ch]["eclass"]]
                    if cc == INF:
                        ok = False
                        break
                    child_costs.append(cc)
                if not ok:
                    continue
                c = base + (sum(child_costs) if mode == "ops" else max(child_costs, default=0.0))
                if c < best_c:
                    best, best_c = nid, c
            if best is not None and best_c < cost[ec]:
                cost[ec], chosen[ec] = best_c, best
                changed = True
        if not changed:
            break
    if cost.get(root_eclass, INF) == INF:
        raise RuntimeError("extraction failed to reach the root eclass")
    return chosen


def extract_min_ops(graph: dict, root_eclass: str):   # back-comat alias used by probe()
    return extract(graph, root_eclass, "ops")


def _leaf_name(graph: dict, eclass: str, chosen) -> str:
    nd = graph["nodes"][chosen[eclass]]
    # M.var -> its String child literal
    return graph["nodes"][nd["children"][0]]["op"].strip('"')


def reemit_cse(graph: dict, root_eclass: str, chosen, threshold: int = 2) -> List[Tuple[str, str]]:
    """Materialize the chosen DAG as flat SSA. CSE GRANULARITY = `threshold`: an op-eclass with
    fan-out >= threshold becomes a named temp; below it is inlined. threshold=1 -> every op is a
    temp (3-address / instruction-level); threshold=2 -> Dendro-like (share only reused); large ->
    aggressive inlining (wide trees). Varying it sweeps source structure at FIXED dataflow."""
    nodes = graph["nodes"]

    # fan-out of each op-eclass within the chosen DAG (reachable from root)
    fan: Dict[str, int] = {}
    seen = set()
    stack = [root_eclass]
    while stack:
        ec = stack.pop()
        nd = nodes[chosen[ec]]
        kind, _ = _node_kind(nd["op"], len(nd["children"]))
        if kind in ("var", "str"):
            continue
        first = ec not in seen
        seen.add(ec)
        for ch in nd["children"]:
            cec = nodes[ch]["eclass"]
            ckind, _ = _node_kind(nodes[chosen[cec]]["op"], len(nodes[chosen[cec]]["children"]))
            if ckind not in ("var", "str"):
                fan[cec] = fan.get(cec, 0) + 1
                if first:
                    stack.append(cec)
    fan[root_eclass] = threshold                          # force the output to a named temp

    name: Dict[str, str] = {}
    stmts: List[Tuple[str, str]] = []
    counter = [0]

    def expr_of(ec: str) -> str:
        nd = nodes[chosen[ec]]
        kind, info = _node_kind(nd["op"], len(nd["children"]))
        if kind == "var":
            return _leaf_name(graph, ec, chosen)
        if kind == "neg":
            return f"-({ref(nd['children'][0])})"            # parens: avoid `--` on nested neg
        cec = [nodes[ch]["eclass"] for ch in nd["children"]]
        # spaces are load-bearing: `a - -x` must NOT collapse to `a--x` (C++ decrement).
        return f"({ref_ec(cec[0])} {info} {ref_ec(cec[1])})"

    def ref_ec(ec: str) -> str:
        nd = nodes[chosen[ec]]
        kind, _ = _node_kind(nd["op"], len(nd["children"]))
        if kind in ("var", "str"):
            return _leaf_name(graph, ec, chosen) if kind == "var" else nd["op"].strip('"')
        if fan.get(ec, 0) >= threshold:                    # >= granularity -> materialized temp
            return materialize(ec)
        return expr_of(ec)                                 # below -> inline

    def ref(child_nid: str) -> str:
        return ref_ec(nodes[child_nid]["eclass"])

    def materialize(ec: str) -> str:
        if ec in name:
            return name[ec]
        t = f"DENDRO_{counter[0]}"
        counter[0] += 1
        name[ec] = t                                       # name first (DAG self-ref safe)
        rhs = expr_of(ec)
        stmts.append((t, rhs))
        return t

    materialize(root_eclass)
    return stmts


# --- scoring (same machinery as the baseline) ----------------------------------------------
def peak_live(statements: List[Tuple[str, str]]) -> int:
    dag = build_dag(statements)
    return straight_line_liveness(dag, min_liveness_order(dag), dtype="fp32")  # raw value count


def root_eclass_of(graph: dict) -> str:
    """The unique sink eclass (referenced by no node's children) — the cone's single output."""
    referenced = set()
    for nd in graph["nodes"].values():
        for ch in nd["children"]:
            referenced.add(graph["nodes"][ch]["eclass"])
    sinks = [ec for ec in graph["class_data"] if ec not in referenced]
    op_sinks = [ec for ec in sinks
                if any(_node_kind(graph["nodes"][n]["op"], len(graph["nodes"][n]["children"]))[0]
                       not in ("var", "str")
                       for n in graph["nodes"] if graph["nodes"][n]["eclass"] == ec)]
    if len(op_sinks) != 1:
        raise RuntimeError(f"expected 1 output sink, got {len(op_sinks)}: {op_sinks}")
    return op_sinks[0]


def _saturate_extract(cone, iters):
    """Build e-graph from the cone, run `iters` rule-rounds (0 = control: my-CSE, no rewrites),
    extract op-count-min DAG, re-CSE. Returns (statements, n_nodes, n_classes)."""
    egraph = EGraph()
    egraph.register(build_root(cone))
    if iters:
        egraph.run(_float_safe_ruleset() * iters)        # bounded (full saturate blows up)
    graph = json.loads(egraph._serialize().to_json())
    rec = root_eclass_of(graph)
    chosen = extract_min_ops(graph, rec)
    return reemit_cse(graph, rec, chosen), len(graph["nodes"]), len(graph["class_data"])


def probe(output: str = "K_rhs", iters: int = 12, verbose: bool = True):
    stmts, _, _ = parse(DENDRO_CSE)
    dag = build_dag(stmts)
    cone = cone_statements(dag, dict(stmts), output)
    base_peak = peak_live(cone)                           # Dendro's own CSE
    base_ops = sum(dag.op_cost[n] for n, _ in cone)

    ctrl, _, _ = _saturate_extract(cone, 0)               # CONTROL: my re-CSE, NO rewrites
    ctrl_peak = peak_live(ctrl)
    ex, n_nodes, n_classes = _saturate_extract(cone, iters)  # saturated
    ex_peak = peak_live(ex)

    if verbose:
        print(f">> e-graph saturation probe (C1) :: output cone = {output}")
        print(f"   cone: {len(cone)} temps, ~{base_ops} ops")
        print(f"   after {iters} rule iters: e-graph = {n_nodes} e-nodes / {n_classes} e-classes")
        print(f"   {'':28}{'peak-live':>10}{'temps':>8}")
        print(f"   {'baseline (Dendro CSE)':28}{base_peak:>10}{len(cone):>8}")
        print(f"   {'control (my re-CSE, 0 rules)':28}{ctrl_peak:>10}{len(ctrl):>8}")
        print(f"   {'e-graph saturated+extracted':28}{ex_peak:>10}{len(ex):>8}")
        # the HONEST e-graph contribution = saturated vs the SAME-policy control:
        d_eg = (ctrl_peak - ex_peak) / ctrl_peak * 100 if ctrl_peak else 0.0
        d_tot = (base_peak - ex_peak) / base_peak * 100 if base_peak else 0.0
        print(f"   e-graph contribution (vs control): {ex_peak - ctrl_peak:+d} values ({d_eg:+.1f}%)"
              f"  {'<-- saturation HELPS' if ex_peak < ctrl_peak else '<-- e-graph adds nothing'}")
        print(f"   total vs Dendro baseline:          {ex_peak - base_peak:+d} values ({d_tot:+.1f}%)")
    return dict(output=output, base_peak=base_peak, ctrl_peak=ctrl_peak, ex_peak=ex_peak,
                n_nodes=n_nodes, n_classes=n_classes)


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _dendro_form(cone):
    return [(l, lower_pow(_LAMBDA_RE.sub("lmbda", r))) for l, r in cone]


def collect_leaves(stmts) -> List[str]:
    """Distinct input identifiers (fields/derivs/scalars) — NOT temps, NOT c_ constants."""
    defined = {l for l, _ in stmts}
    seen, leaves = set(), []
    for _, rhs in stmts:
        for tok in _IDENT.findall(rhs):
            if tok not in defined and tok not in seen and not tok.startswith("c_"):
                seen.add(tok)
                leaves.append(tok)
    return leaves


def _eval_ssa(stmts, leafvals: Dict[str, float]) -> float:
    env = {**leafvals, **{c: float(v) for c, v in CONST_MAP.items()}}
    for lhs, rhs in stmts:
        env[lhs] = eval(rhs, {"__builtins__": {}}, env)   # noqa: S307 - controlled codegen
    return env[stmts[-1][0]]


def check_equivalence(output="K_rhs", trials=200, seed=0) -> float:
    """CPU: verify the e-graph re-CSE is mathematically equal to Dendro's form (round-off)."""
    import random
    stmts, _, _ = parse(DENDRO_CSE)
    dag = build_dag(stmts)
    cone = cone_statements(dag, dict(stmts), output)
    dendro = _dendro_form(cone)
    recse, _, _ = _saturate_extract(cone, 0)              # populates CONST_MAP via build_root
    leaves = collect_leaves(dendro)
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(trials):
        lv = {leaf: rng.uniform(0.5, 1.5) for leaf in leaves}   # away from 0 (denominators)
        a, b = _eval_ssa(dendro, lv), _eval_ssa(recse, lv)
        worst = max(worst, abs(a - b) / (abs(a) + 1e-30))
    return worst


def _emit_kernel(name, stmts, leaves, inline_const) -> List[str]:
    idx = {leaf: k for k, leaf in enumerate(leaves)}
    L = [f"__global__ void {name}(const double* __restrict__ IN, "
         "double* __restrict__ OUT, int N) {",
         "  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= N) return;"]
    for leaf, k in idx.items():
        L.append(f"  const double {leaf} = IN[(size_t){k}*N + i];")
    declared = set(leaves)
    for lhs, rhs in stmts:
        r = rhs
        if inline_const:                                  # c_<x> leaves -> numeric literals
            r = re.sub(r"\bc_[A-Za-z0-9_]+\b",
                       lambda m: CONST_MAP.get(m.group(0), m.group(0)), r)
        kw = "" if lhs in declared else "double "
        L.append(f"  {kw}{lhs} = {r};")
        declared.add(lhs)
    L.append(f"  OUT[i] = {stmts[-1][0]};")
    L.append("}")
    return L


def emit_4b_cuda(output="K_rhs", path=None) -> Path:
    """Emit a .cu with TWO kernels — Dendro CSE vs e-graph re-CSE of `output` — for a
    `nvcc -c -Xptxas -v` register comparison (the 4b decisive test: does the CPU-liveness
    reduction translate to fewer ptxas registers, or does ptxas already capture it?)."""
    stmts, _, _ = parse(DENDRO_CSE)
    dag = build_dag(stmts)
    cone = cone_statements(dag, dict(stmts), output)
    dendro = _dendro_form(cone)
    recse, _, _ = _saturate_extract(cone, 0)
    leaves = collect_leaves(dendro)

    out = ["// AUTO-GENERATED 4b probe (bssn3d.egraph_probe --emit-4b). Compare ptxas -v registers:",
           f"//   nvcc -arch=sm_90a -c -Xptxas -v {Path(path).name if path else 'krhs_4b.cu'}",
           f"// Two equivalent forms of {output}: Dendro CSE ({len(dendro)} temps) vs e-graph",
           f"// re-CSE ({len(recse)} temps). Same {len(leaves)} leaf inputs (IN[k*N+i]).", ""]
    out += _emit_kernel(f"{output.lower()}_dendro", dendro, leaves, inline_const=False)
    out.append("")
    out += _emit_kernel(f"{output.lower()}_recse", recse, leaves, inline_const=True)
    out.append("")
    p = Path(path) if path else Path(__file__).resolve().parent / "cuda" / f"{output.lower()}_4b.cu"
    p.write_text("\n".join(out) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser(description="C1 e-graph saturation probe + 4b emitter.")
    ap.add_argument("--output", default="K_rhs", help="output cone to probe")
    ap.add_argument("--iters", type=int, default=12, help="bounded rule iterations")
    ap.add_argument("--check", action="store_true", help="CPU: verify re-CSE == Dendro (round-off)")
    ap.add_argument("--emit-4b", action="store_true",
                    help="emit the Dendro-vs-reCSE CUDA kernel pair for ptxas -v")
    a = ap.parse_args()
    if a.check:
        print(f">> re-CSE vs Dendro equivalence ({a.output}): max rel err = "
              f"{check_equivalence(a.output):.2e}")
    elif a.emit_4b:
        p = emit_4b_cuda(a.output)
        print(f">> wrote {p}")
    else:
        probe(a.output, a.iters)


if __name__ == "__main__":
    main()
