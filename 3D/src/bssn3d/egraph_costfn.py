"""Step 3.4 / B — cost-function EXISTENCE experiment.

C1/4b showed our named-temp liveness proxy ANTI-predicts ptxas (re-CSE 194 vs Dendro 158). Before
declaring "no constructible GPU cost function," this asks the question head-on: across a SPREAD of
equivalent forms of a cone, does ANY static feature predict ptxas registers (and wall-clock)?

It generates ~12 equivalent forms spanning two axes:
  * CSE granularity (threshold 1..8) — same dataflow, different source structure (the axis that
    fooled us: more inlining -> wider trees). Varies temp_count, named_liveness; FIXES the
    dataflow features (instr_liveness, crit_path).
  * Reassociation (e-graph, op-count vs depth extraction) — changes the dataflow. Varies
    instr_liveness, crit_path, op_count.
plus the true Dendro baseline (the 158-reg reference).

For each form it computes the static features and emits a CUDA kernel; you compile the whole .cu
with `nvcc -c -Xptxas -v` and report the per-kernel register counts. Pair those with the printed
feature table and the answer falls out:
  - some feature correlates with ptxas regs -> THAT is the cost function -> e-graph is back on.
  - nothing correlates -> ptxas needs measurement, not static prediction (-> learned model / pivot).

Run:  python -m bssn3d.egraph_costfn --output K_rhs
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from egglog import EGraph

from ._codegen import parse, DENDRO_CSE
from .staging import build_dag, min_liveness_order, straight_line_liveness
from . import egraph_probe as ep


def _peak(stmts) -> int:
    dag = build_dag(stmts)
    return straight_line_liveness(dag, min_liveness_order(dag), dtype="fp32")


def _longest_chain(stmts) -> int:
    """Critical path = longest dependency chain (in ops), over the SSA in topological order."""
    dag = build_dag(stmts)
    depth: Dict[str, int] = {}
    best = 0
    for n in dag.order:
        d = 1
        for dep in dag.deps[n]:
            if dep in depth:
                d = max(d, depth[dep] + 1)
        depth[n] = d
        best = max(best, d)
    return best


def _build_saturate(cone, iters: int, mode: str):
    eg = EGraph()
    eg.register(ep.build_root(cone))
    if iters:
        eg.run(ep._float_safe_ruleset() * iters)
    graph = json.loads(eg._serialize().to_json())
    rec = ep.root_eclass_of(graph)
    chosen = ep.extract(graph, rec, mode)
    return graph, rec, chosen


def _features(form_stmts, instr_stmts) -> dict:
    return dict(
        temps=len(form_stmts),
        named_live=_peak(form_stmts),          # source-structure (our old proxy)
        instr_live=_peak(instr_stmts),         # dataflow (CSE-invariant) register pressure
        ops=len(instr_stmts),                  # canonical op count
        crit_path=_longest_chain(instr_stmts), # dataflow critical path
    )


def generate_forms(output: str, reassoc_iters: int = 0):
    """Return [(label, statements, features, inline_const, rel_err)] — the form spread.

    The granularity sweep (iters=0, bit-identical) is the CORE decisive test and is memory-cheap.
    `reassoc_iters > 0` ALSO saturates to get dataflow-varying forms — but saturation explodes
    (~1.1M e-nodes by iters=5 on a 169-temp cone) and can OOM a small box; keep it <=2."""
    stmts, _, _ = parse(DENDRO_CSE)
    dag = build_dag(stmts)
    cone = ep.cone_statements(dag, dict(stmts), output)
    dendro = ep._dendro_form(cone)
    leaves = ep.collect_leaves(dendro)

    # iters=0 e-graph -> bit-identical granularity sweep
    g0, r0, ch0 = _build_saturate(cone, 0, "ops")
    instr0 = ep.reemit_cse(g0, r0, ch0, threshold=1)

    rng = random.Random(7)
    samples = [{lf: rng.uniform(0.5, 1.5) for lf in leaves} for _ in range(64)]

    def rel_err(form_stmts):
        worst = 0.0
        for lv in samples:
            a = ep._eval_ssa(dendro, lv)
            b = ep._eval_ssa(form_stmts, lv)
            worst = max(worst, abs(a - b) / (abs(a) + 1e-30))
        return worst

    forms = []
    # true Dendro baseline (the reference 158-reg form)
    forms.append(("dendro", dendro, _features(dendro, instr0), False, 0.0))
    # granularity sweep (bit-identical, op-count extraction)
    for t in (1, 2, 3, 5, 8):
        f = ep.reemit_cse(g0, r0, ch0, threshold=t)
        forms.append((f"gran_t{t}", f, _features(f, instr0), True, rel_err(f)))
    # reassociation variants (change the dataflow) — equivalent within round-off. OPT-IN +
    # low-iters: saturation explodes and can OOM. Skipped by default (granularity is the core test).
    if reassoc_iters:
        for mode in ("ops", "depth"):
            g, r, ch = _build_saturate(cone, reassoc_iters, mode)
            f = ep.reemit_cse(g, r, ch, threshold=2)
            instr = ep.reemit_cse(g, r, ch, threshold=1)
            forms.append((f"reassoc_i{reassoc_iters}_{mode}", f, _features(f, instr), True, rel_err(f)))
    return forms, leaves


def emit_cuda(output: str, forms, leaves, path=None) -> Path:
    out = ["// AUTO-GENERATED cost-fn experiment (bssn3d.egraph_costfn). Compile + read regs:",
           f"//   nvcc -arch=sm_90a -c -Xptxas -v {output.lower()}_costfn.cu",
           f"// {len(forms)} equivalent forms of {output}; pair the per-kernel regs with the",
           "// feature table printed by the tool.", ""]
    for i, (label, stmts, _feat, inline_const, _re) in enumerate(forms):
        out += ep._emit_kernel(f"f{i:02d}_{label}", stmts, leaves, inline_const)
        out.append("")
    p = Path(path) if path else Path(ep.__file__).resolve().parent / "cuda" / f"{output.lower()}_costfn.cu"
    p.write_text("\n".join(out) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser(description="B: GPU cost-function existence experiment.")
    ap.add_argument("--output", default="K_rhs")
    ap.add_argument("--reassoc-iters", type=int, default=0,
                    help="ALSO add dataflow-varying reassoc forms (saturates; keep <=2; can OOM)")
    a = ap.parse_args()

    forms, leaves = generate_forms(a.output, a.reassoc_iters)
    p = emit_cuda(a.output, forms, leaves)

    print(f">> cost-fn experiment :: {a.output}  ({len(forms)} equivalent forms, {len(leaves)} leaves)")
    print(f"   kernel      {'temps':>6}{'named_live':>11}{'instr_live':>11}{'ops':>6}"
          f"{'crit_path':>10}{'rel_err':>10}")
    for i, (label, _s, f, _ic, re) in enumerate(forms):
        print(f"   f{i:02d}_{label:11}{f['temps']:>6}{f['named_live']:>11}{f['instr_live']:>11}"
              f"{f['ops']:>6}{f['crit_path']:>10}{re:>10.1e}")
    print(f"\n>> wrote {p}")
    print(">> NEXT (Marylou): nvcc -arch=sm_90a -c -Xptxas -v "
          f"src/bssn3d/cuda/{a.output.lower()}_costfn.cu 2>&1 | grep -iE 'entry function|registers'")
    print(">> Then pair each kernel's ptxas registers with the row above; report which feature")
    print(">> (if any) tracks the register count. That feature == the GPU cost function we need.")


if __name__ == "__main__":
    main()
