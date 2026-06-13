"""Dataflow analysis of the transliterated BSSN CSE RHS (Phase 3.1).

The verbatim RHS (`_bssn_rhs_generated.py`, from `bssneqs_SSL_HD_dxsq.cpp`) is a
flat SSA block of ~850 `DENDRO_* = <expr>` statements. XLA fuses it into a few
huge pointwise kernels whose *static* live set (peak ~584 temps) overruns the
255-register file → spill (measured: see `[[bssn-codegen-staging]]`). Phase 3
makes it register-bounded by **staging**: store a small set of high-fan-out
tensor-hierarchy temps (inverse metric, Christoffels, Ricci, CalGt) as per-point
scalars and recompute the cheap leaves, so no single fused kernel holds the whole
584-wide live set at once.

This module is the dataflow front-end for that:

  * `build_dag(statements)` — parse the SSA into a DAG (deps, op-cost, fan-out).
  * `cone_cost(dag)` — transitive recompute cost of each node (store-vs-recompute).
  * `rank_candidates(dag)` — fan-out × cone-cost ranking → the materialization
    (cut/barrier) candidates that Step 3.1's staged emission pins, and that 3.2's
    automatic cut-set generator consumes.

It reads the *same* vendored CSE the committed `_bssn_rhs_generated.py` is built
from (via `_codegen.parse`), so the analysis can never drift from the emitted RHS.

Run:  `python -m bssn3d.staging`  (prints the fan-out / recompute report).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import hashlib
from pathlib import Path

from ._codegen import (
    parse, _emit_module, _translate_rhs, DENDRO_CSE, FIELD_INPUTS, RHS_TO_FIELD,
    SCALAR_PARAMS,
)

STAGED_OUT = Path(__file__).resolve().parent / "_bssn_rhs_staged.py"

# Default cut-set size: pin the top-K materialization candidates as
# optimization_barrier stage boundaries. Tuned against the measured spill map;
# the ~69 figure tracks the tensor-hierarchy temp count (igt/Christoffel/Ricci/CalGt).
DEFAULT_K = 69

# A DENDRO_* CSE temp (the internal nodes of the DAG); outputs are the 24 *_rhs.
_TEMP_RE = re.compile(r"\bDENDRO_\d+\b")
# Arithmetic-operator proxy for op cost. We count binary/unary +,-,*,/ and the
# three transcendental calls; constant-fold ratios like `3.0/4.0` slightly
# over-count but only as a uniform bias — fine for a *relative* ranking.
_OP_RE = re.compile(r"[-+*/]")
_CALL_RE = re.compile(r"\b(pow|sqrt|exp)\(")

OUTPUT_TOKENS = list(RHS_TO_FIELD.keys())
_OUTPUT_SET = set(OUTPUT_TOKENS)


@dataclass
class DAG:
    """Static dataflow of the SSA RHS.

    order:    statement LHS names in file (define-before-use) order.
    deps:     node -> set of DENDRO_* temps it directly references.
    op_cost:  node -> arithmetic-op count of its defining expression.
    fanout:   temp -> number of *downstream statements* that reference it.
    is_output: node -> True if it is one of the 24 `*_rhs` outputs.
    """

    order: List[str]
    deps: Dict[str, Set[str]] = field(default_factory=dict)
    op_cost: Dict[str, int] = field(default_factory=dict)
    fanout: Dict[str, int] = field(default_factory=dict)
    is_output: Dict[str, bool] = field(default_factory=dict)

    @property
    def temps(self) -> List[str]:
        return [n for n in self.order if not self.is_output[n]]


def op_cost(expr: str) -> int:
    """Cheap arithmetic-op proxy for the cost of recomputing one expression."""
    return len(_OP_RE.findall(expr)) + len(_CALL_RE.findall(expr))


def build_dag(statements: List[Tuple[str, str]] | None = None) -> DAG:
    """Build the SSA dataflow DAG from the vendored CSE statements.

    Each statement `lhs = rhs`; an edge `lhs -> t` for every DENDRO_* temp `t`
    referenced in `rhs`. Leaves (fields, `grad_*`/`grad2_*`, scalar params,
    literals) are *not* nodes — they are the recompute floor.
    """
    if statements is None:
        statements, _, _ = parse(DENDRO_CSE)

    dag = DAG(order=[lhs for lhs, _ in statements])
    fan: Dict[str, int] = {}
    for lhs, rhs in statements:
        refs = set(_TEMP_RE.findall(rhs))
        dag.deps[lhs] = refs
        dag.op_cost[lhs] = op_cost(rhs)
        dag.is_output[lhs] = lhs in _OUTPUT_SET
        for t in refs:
            fan[t] = fan.get(t, 0) + 1
    # every temp gets a fan-out entry (0 if only feeds an output via itself)
    dag.fanout = {n: fan.get(n, 0) for n in dag.order}
    return dag


def cone_cost(dag: DAG) -> Dict[str, int]:
    """Transitive recompute cost of each node's dependency cone.

    `cone[n]` = op_cost[n] + sum of op_cost over all *distinct* transitive temp
    ancestors of `n`. This is the work to recompute `n` from leaves if nothing
    below it is stored — the "recompute" side of the store-vs-recompute call.
    Computed in topological (file) order; ancestor sets are memoised.
    """
    ancestors: Dict[str, Set[str]] = {}
    cone: Dict[str, int] = {}
    for n in dag.order:                      # file order == topological order
        anc: Set[str] = set()
        for d in dag.deps[n]:
            anc.add(d)
            anc |= ancestors.get(d, set())
        ancestors[n] = anc
        cone[n] = dag.op_cost[n] + sum(dag.op_cost.get(a, 0) for a in anc)
    return cone


@dataclass
class Candidate:
    name: str
    fanout: int
    op_cost: int
    cone_cost: int
    score: int          # fanout * cone_cost — recompute saved by storing it


def rank_candidates(dag: DAG | None = None) -> List[Candidate]:
    """Rank temps as materialization (store/barrier) candidates.

    Heuristic: a temp is worth storing when it is reused a lot (high fan-out) AND
    expensive to rebuild (large cone) — storing it once avoids `fanout` repeats of
    its whole cone. Score = `fanout * cone_cost`. Outputs are excluded (they are
    the sinks, already materialised). This is the seed the 3.2 automatic cut-set
    generator refines against the measured spill map.
    """
    if dag is None:
        dag = build_dag()
    cone = cone_cost(dag)
    cands = [
        Candidate(n, dag.fanout[n], dag.op_cost[n], cone[n],
                  dag.fanout[n] * cone[n])
        for n in dag.order if not dag.is_output[n]
    ]
    cands.sort(key=lambda c: (-c.score, -c.fanout))
    return cands


def select_cut_set(k: int = DEFAULT_K, dag: DAG | None = None) -> List[str]:
    """The top-``k`` materialization candidates, as a sorted list of temp names."""
    return sorted(c.name for c in rank_candidates(dag)[:k])


# ---------------------------------------------------------------------------
# 3.2b — the materialize/recompute schedule (the generator core)
# ---------------------------------------------------------------------------
#
# A schedule partitions the temps into a **materialize set** M (stored as per-point
# register scalars) and a **recompute set** R = temps \ M (re-derived inline at each
# use). The two design metrics:
#
#   * persistent_liveness(M) — peak simultaneously-live M-temps under the realized
#     schedule (M-temps in SSA order, then the 24 outputs). This is the dominant term
#     of register pressure; the staging target is to keep it well under the 255-reg
#     budget (leaving headroom for the recompute-tree transients + derivative loads,
#     which this model treats as reserve — the authoritative check is 3.2c's ptxas).
#   * recompute_ops(M) — total emitted arithmetic ops once R-temps are inlined per use
#     (an R-temp used by k realized nodes is recomputed k times). The multiplier vs the
#     verbatim 4527-op CSE is the price paid to shrink liveness.
#
# Sweeping M (top-K by fan-out×cone) traces the store-vs-recompute Pareto curve; the
# selected schedule is the largest K whose persistent liveness fits the budget (most
# materialization → least recompute, subject to registers).


def _mdeps_map(dag: DAG, M) -> Dict[str, Set[str]]:
    """For each node, the set of **M-temps** reached when its expression is expanded
    with R-temps inlined (M-temps are stored leaves; R-temps recurse). Iterative in
    SSA order (deps precede uses), so no recursion-limit risk."""
    Mset = set(M)
    memo: Dict[str, Set[str]] = {}
    for n in dag.order:
        res: Set[str] = set()
        for d in dag.deps[n]:
            if d in Mset:
                res.add(d)            # stored leaf — don't expand
            else:
                res |= memo[d]        # R-temp — its inlined M-leaves
        memo[n] = res
    return memo


def persistent_liveness(dag: DAG, M) -> int:
    """Peak count of simultaneously-live M-temps under the realized schedule."""
    Mset = set(M)
    if not Mset:
        return 0
    mdeps = _mdeps_map(dag, Mset)
    realized = [n for n in dag.order if n in Mset or dag.is_output[n]]
    pos = {n: i for i, n in enumerate(realized)}
    last = {m: pos[m] for m in Mset}             # live at least at its definition
    for n in realized:
        for m in mdeps[n]:
            last[m] = max(last[m], pos[n])
    # peak interval overlap: +1 at def, -1 just past last use (defs before frees)
    events = []
    for m in Mset:
        events.append((pos[m], 0))               # 0 sorts before 1 → def first
        events.append((last[m] + 1, 1))
    events.sort()
    live = peak = 0
    for _, kind in events:
        live += 1 if kind == 0 else -1
        peak = max(peak, live)
    return peak


def recompute_ops(dag: DAG, M) -> int:
    """Total emitted ops once R-temps are inlined (R used k× → recomputed k×)."""
    Mset = set(M)
    inl: Dict[str, int] = {}                     # inlined-cone op cost of each R-temp
    for n in dag.order:
        if n in Mset or dag.is_output[n]:
            continue
        c = dag.op_cost[n]
        for d in dag.deps[n]:
            if d not in Mset:
                c += inl[d]
        inl[n] = c
    total = 0
    for n in dag.order:                          # realized nodes: M-temps + outputs
        if n in Mset or dag.is_output[n]:
            c = dag.op_cost[n]
            for d in dag.deps[n]:
                if d not in Mset:
                    c += inl[d]
            total += c
    return total


@dataclass
class Schedule:
    materialize: List[str]      # SSA-ordered M-temps (stored as register scalars)
    peak_live: int              # persistent peak of simultaneously-live M-temps
    recompute_ops: int          # total emitted ops (R inlined per use)
    multiplier: float           # recompute_ops / verbatim CSE ops
    budget: int


def liveness_cost_curve(dag: DAG, Ks, ranked=None):
    """For each K, materialize the top-K by score; return (K, peak_live, multiplier)."""
    ranked = ranked or [c.name for c in rank_candidates(dag)]
    base = sum(dag.op_cost.values())             # verbatim CSE op total
    rows = []
    for k in Ks:
        M = ranked[:k]
        rows.append((k, persistent_liveness(dag, M), recompute_ops(dag, M) / base))
    return rows


def select_schedule(dag: DAG | None = None, budget: int = 200) -> Schedule:
    """Largest top-K materialization whose persistent liveness fits ``budget``.

    ``budget`` targets the *persistent* peak; the 255-register file keeps headroom for
    recompute-tree transients + derivative loads. Most materialization under the budget
    = least recompute. (Final word is 3.2c's ptxas, not this estimate.)"""
    if dag is None:
        dag = build_dag()
    ranked = [c.name for c in rank_candidates(dag)]
    base = sum(dag.op_cost.values())
    # scan K upward; persistent liveness is ~monotone in K → take the last fit.
    best_k = 0
    for k in range(0, len(ranked) + 1):
        if persistent_liveness(dag, ranked[:k]) <= budget:
            best_k = k
        else:
            break
    M = ranked[:best_k]
    M_ordered = [n for n in dag.order if n in set(M)]
    ops = recompute_ops(dag, M)
    return Schedule(M_ordered, persistent_liveness(dag, M), ops, ops / base, budget)


def output_fanout(dag: DAG | None = None) -> Dict[str, int]:
    """``temp -> number of the 24 outputs it is a transitive ancestor of``.

    The register-pressure analysis (2026-06-12, ``docs/algebra.md``) found the spill
    floor is a *shared first-order tensor trunk* — temps consumed by many outputs stay
    co-live across the whole output phase. Output-fanout is the discriminator that
    isolates that trunk (the 128 temps feeding >=12 outputs are 100% first-order),
    where fan-out x cone-cost (``rank_candidates``) does not."""
    if dag is None:
        dag = build_dag()
    anc: Dict[str, Set[str]] = {}
    for n in dag.order:                          # file order == topological
        a: Set[str] = set()
        for d in dag.deps[n]:
            a.add(d); a |= anc[d]
        anc[n] = a
    outs = [n for n in dag.order if dag.is_output[n]]
    of: Dict[str, int] = {}
    for t in dag.order:
        if not dag.is_output[t]:
            of[t] = sum(1 for o in outs if t in anc[o])
    return of


def select_trunk_schedule(dag: DAG | None = None,
                          min_outfanout: int = 12) -> Schedule:
    """Materialize the **output-fanout trunk** (temps feeding >= ``min_outfanout`` of
    the 24 outputs), recompute the rest. This is the fp64 + SMEM-trunk strategy: the
    trunk is too wide for the fp64 register file (>=12 -> 128 temps -> ~123 peak live ->
    ~246 regs) so it is destined for SMEM, while the bulk is recomputed in registers.
    Unlike ``select_schedule`` this is NOT register-budget-capped — the trunk size is
    set by the dataflow (the shared tensor hierarchy), and SMEM holds it. See
    ``[[bssn-register-pressure-structural]]``."""
    if dag is None:
        dag = build_dag()
    of = output_fanout(dag)
    M = [t for t in dag.order if not dag.is_output[t] and of[t] >= min_outfanout]
    base = sum(dag.op_cost.values())
    ops = recompute_ops(dag, M)
    return Schedule(M, persistent_liveness(dag, M), ops, ops / base, min_outfanout)


def generate_staged(k: int = DEFAULT_K, cut_set: List[str] | None = None,
                    out: Path = STAGED_OUT) -> Path:
    """Emit ``_bssn_rhs_staged.py`` — verbatim algebra + optimization_barrier cuts.

    ``cut_set`` overrides the default top-``k`` ranking (so the 3.2 generator, or a
    spill-map-driven retune, can pin a hand-chosen set). The staged module is a
    *second* artifact; the verbatim ``_bssn_rhs_generated.py`` stays the oracle.
    """
    statements, grad1, grad2 = parse(DENDRO_CSE)
    if cut_set is None:
        cut_set = select_cut_set(k, build_dag(statements))
    src_hash = hashlib.sha256(DENDRO_CSE.read_bytes()).hexdigest()[:16]
    out.write_text(_emit_module(statements, grad1, grad2, src_hash,
                                src_name=DENDRO_CSE.name, cut_set=set(cut_set)))
    return out


def schedule_pylines(dag: DAG, M, statements=None) -> List[str]:
    """Emit the scheduled algebra as Python assignment lines: M-temps (SSA order) then
    the 24 outputs, with R-temps **inlined** (recomputed at each use) and M-temps left
    as variable references. The realization of a `Schedule` — consumed by the 3.2c
    Pallas backend and exec'd by the correctness gate. Pure substitution, so it equals
    the verbatim algebra to round-off (fp summation order aside)."""
    if statements is None:
        statements, _, _ = parse(DENDRO_CSE)
    Mset = set(M)
    py = {lhs: _translate_rhs(rhs) for lhs, rhs in statements}

    inl: Dict[str, str] = {}                     # memoised inlined string per R-temp

    def expand(expr: str) -> str:
        return _TEMP_RE.sub(
            lambda m: m.group(0) if m.group(0) in Mset else f"({inl[m.group(0)]})",
            expr)

    for n in dag.order:                          # SSA order → R-temp deps ready
        if n not in Mset and not dag.is_output[n]:
            inl[n] = expand(py[n])

    lines = []
    for n in dag.order:
        if n in Mset:
            lines.append(f"{n} = {expand(py[n])}")
    for tok in RHS_TO_FIELD:                      # the 24 outputs, file/return order
        lines.append(f"{tok} = {expand(py[tok])}")
    return lines


# ---------------------------------------------------------------------------
# 3.2 Phase 1.1 — straight-line liveness + the reassociation prediction
# ---------------------------------------------------------------------------
#
# The spill→0 lever for the *fp32* algebra kernel. The key compiler fact: `ptxas`
# freely reorders **independent** instructions (so the 3.2b materialize/recompute
# schedule washed out — it re-derived its own allocation), **but it cannot reassociate
# floating-point** (that changes rounding, which it must preserve). So a *reassociated*
# computation tree is BINDING where a mere reorder is not. The question this section
# answers — entirely on CPU, before any GPU time — is whether reassociating the wide
# associative contractions can push the peak live set below the 255-register file in
# fp32 (1 reg/value), or whether the width is structurally irreducible.
#
# Three models, increasing in what they let the scheduler do:
#   * straight_line_liveness(order)  — peak simultaneously-live values for a *fixed*
#     evaluation order (store-everything; file order or any topological permutation).
#   * min_liveness_order()           — greedy list schedule that REORDERS independent
#     statements to minimise the live set (the ptxas-achievable reorder; NOT binding).
#   * reassociation_floor()          — peak live **multi-use** (fan-out>=2) temps. A
#     reassociation can only collapse a *single-use* reduction chain (stream it into one
#     accumulator); a temp consumed by >=2 statements stays live across all of them no
#     matter how you associate. So this is the hard floor reassociation cannot beat.
# If the floor already exceeds the budget, reassociation is dead.


def node_consumers(dag: DAG) -> Dict[str, List[str]]:
    """``node -> list of nodes that directly reference it`` (leaves excluded)."""
    nodeset = set(dag.order)
    cons: Dict[str, List[str]] = {n: [] for n in dag.order}
    for n in dag.order:
        for d in dag.deps[n]:
            if d in nodeset:
                cons[d].append(n)
    return cons


def _regs_per_value(dtype: str) -> int:
    """fp64 occupies 2 32-bit registers per value; fp32 occupies 1. (This is the
    factor 3.2b's persistent-liveness model omitted — '133 live' was 266 fp64 regs.)"""
    if dtype not in ("fp32", "fp64"):
        raise ValueError(f"dtype must be 'fp32' or 'fp64', got {dtype!r}")
    return 1 if dtype == "fp32" else 2


def _peak_overlap(spans) -> int:
    """Peak count of overlapping ``[start, last]`` integer spans (defs before frees)."""
    events = []
    for start, last in spans:
        events.append((start, 0))        # 0 sorts before 1 → def first
        events.append((last + 1, 1))
    events.sort()
    live = peak = 0
    for _, kind in events:
        live += 1 if kind == 0 else -1
        peak = max(peak, live)
    return peak


def straight_line_liveness(dag: DAG, order: List[str] | None = None,
                           dtype: str = "fp32") -> int:
    """Peak register pressure of a straight-line (store-everything) evaluation.

    Each node is live from its position in ``order`` to its last use; the peak interval
    overlap, scaled by ``_regs_per_value(dtype)``, is the register pressure ptxas faces
    if it materialises every temp. ``order`` must be a topological permutation of
    ``dag.order`` (defaults to file order). Leaves (fields/derivatives/scalars) are not
    counted — they are reloadable inputs, modelled separately as the derivative-read
    reserve.
    """
    if order is None:
        order = dag.order
    pos = {n: i for i, n in enumerate(order)}
    last = {n: pos[n] for n in order}                # live at least at its own def
    for n in order:
        for d in dag.deps[n]:
            if d in pos:
                last[d] = max(last[d], pos[n])
    peak = _peak_overlap((pos[n], last[n]) for n in order)
    return peak * _regs_per_value(dtype)


def min_liveness_order(dag: DAG) -> List[str]:
    """Greedy min-register list schedule (the ptxas-achievable reorder, not binding).

    At each step emit the ready node that least increases the live set: it frees every
    operand whose last remaining use this is (stream-accumulating single-use reduction
    chains falls out of this — emitting the next addend frees the previous partial),
    and adds itself only if it still has downstream uses. Ties break toward the earlier
    file position (keeps the schedule close to the validated order). Because ptxas does
    exactly this reordering itself, the gain here is an *upper bound on a non-binding
    transform* — the honest baseline reassociation would have to beat.
    """
    nodeset = set(dag.order)
    cons = node_consumers(dag)
    order_idx = {n: i for i, n in enumerate(dag.order)}
    indeg = {n: sum(1 for d in dag.deps[n] if d in nodeset) for n in dag.order}
    remaining = {n: len(cons[n]) for n in dag.order}
    ready = [n for n in dag.order if indeg[n] == 0]
    sched: List[str] = []
    while ready:
        best, best_key = None, None
        for n in ready:
            freed = sum(1 for d in dag.deps[n] if d in nodeset and remaining[d] == 1)
            becomes_live = 1 if remaining[n] > 0 else 0
            key = (becomes_live - freed, order_idx[n])
            if best_key is None or key < best_key:
                best, best_key = n, key
        ready.remove(best)
        sched.append(best)
        for d in dag.deps[best]:
            if d in nodeset:
                remaining[d] -= 1
        for c in cons[best]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return sched


def reassociation_floor(dag: DAG, dtype: str = "fp32") -> int:
    """Hard lower bound on register pressure that NO reassociation/reorder can beat.

    Reassociation can stream a value into a single consumer (one accumulator), so it can
    shrink a **single-use** reduction chain to ~2 live values. It cannot help a temp with
    fan-out >= 2: that temp is live from its definition to its *last* consumer regardless
    of association, and the CSE temp-set + fan-outs are invariant under reassociation
    (only de-CSE / recompute changes them — a different, ptxas-re-derived lever). So the
    peak simultaneously-live multi-use temp count is the irreducible structural width.
    """
    cons = node_consumers(dag)
    multiuse = [n for n in dag.order if len(cons[n]) >= 2]
    pos = {n: i for i, n in enumerate(dag.order)}
    mset = set(multiuse)
    last = {n: pos[n] for n in multiuse}
    for n in dag.order:
        for d in dag.deps[n]:
            if d in mset:
                last[d] = max(last[d], pos[n])
    peak = _peak_overlap((pos[n], last[n]) for n in multiuse)
    return peak * _regs_per_value(dtype)


@dataclass
class ReassocPrediction:
    n_temps: int
    n_multiuse: int          # fan-out >= 2: the reassociation-invariant working set
    n_singleuse: int         # fan-out <= 1: the only reassociable reduction material
    file_live_fp32: int      # store-everything, file order
    reorder_live_fp32: int   # store-everything, greedy min-liveness reorder (ptxas-achievable)
    floor_fp32: int          # multi-use peak — reassociation cannot go below this
    floor_fp64: int
    budget: int
    viable: bool             # floor comfortably (<= 0.9*budget) under the register file


def predict_reassociation(dag: DAG | None = None, budget: int = 255,
                          margin: float = 0.9) -> ReassocPrediction:
    """The Phase-1.1 spill→0 verdict, computed entirely on CPU (no GPU time).

    Viable iff the reassociation floor sits *comfortably* (<= ``margin`` * ``budget``)
    under the register file in fp32 — i.e. there is real headroom for reassociation to
    realise. If the floor is at or above the budget, the width is structural and
    reassociation is dead; weight shifts to derivative fusion (1.2).
    """
    if dag is None:
        dag = build_dag()
    cons = node_consumers(dag)
    n_multi = sum(1 for n in dag.order if len(cons[n]) >= 2)
    floor32 = reassociation_floor(dag, "fp32")
    return ReassocPrediction(
        n_temps=len(dag.temps),
        n_multiuse=n_multi,
        n_singleuse=len(dag.temps) - n_multi,
        file_live_fp32=straight_line_liveness(dag, None, "fp32"),
        reorder_live_fp32=straight_line_liveness(dag, min_liveness_order(dag), "fp32"),
        floor_fp32=floor32,
        floor_fp64=reassociation_floor(dag, "fp64"),
        budget=budget,
        viable=floor32 <= margin * budget,
    )


def main() -> None:
    dag = build_dag()
    cone = cone_cost(dag)
    n_temps = len(dag.temps)
    n_out = sum(dag.is_output.values())
    total_ops = sum(dag.op_cost.values())

    print(f">> BSSN CSE dataflow | source = {DENDRO_CSE.name}")
    print(f">> {len(dag.order)} statements = {n_temps} temps + {n_out} outputs "
          f"| {total_ops} arithmetic ops total")

    fanouts = sorted(dag.fanout.values(), reverse=True)
    reused = [n for n in dag.temps if dag.fanout[n] >= 2]
    print(f">> fan-out: max={fanouts[0]}, "
          f"#temps reused (fanout>=2) = {len(reused)}, "
          f"single-use (fanout<=1) = {n_temps - len(reused)}")

    cands = rank_candidates(dag)
    print(f"\n>> Top 30 materialization candidates (score = fanout * cone_cost):")
    print(f"   {'rank':>4} {'temp':>10} {'fanout':>7} {'op':>4} {'cone':>6} {'score':>8}")
    for i, c in enumerate(cands[:30], 1):
        print(f"   {i:4d} {c.name:>10} {c.fanout:7d} {c.op_cost:4d} "
              f"{c.cone_cost:6d} {c.score:8d}")

    # A "store the top-K" sweep: how many temps to pin to cover most reuse.
    print(f"\n>> cumulative recompute coverage by storing top-K candidates:")
    total_reuse = sum(c.score for c in cands)
    cum = 0
    for k in (16, 32, 48, 64, 69, 96, 128):
        cum = sum(c.score for c in cands[:k])
        print(f"   K={k:4d}  covers {100*cum/total_reuse:5.1f}% of fanout*cone")

    # 3.2b — the materialize/recompute schedule: store-vs-recompute Pareto curve.
    print(f"\n>> store-vs-recompute curve (top-K materialized, rest recomputed):")
    print(f"   {'K':>4} {'peak_live':>9} {'recompute_x':>12}")
    curve = liveness_cost_curve(dag, (0, 16, 32, 48, 64, 96, 128, 192, 256, 384,
                                      len(dag.temps)),
                                ranked=[c.name for c in cands])
    for k, live, mult in curve:
        print(f"   {k:4d} {live:9d} {mult:11.2f}x")

    sched = select_schedule(dag, budget=200)
    print(f"\n>> selected schedule (budget {sched.budget} persistent regs): "
          f"|M|={len(sched.materialize)}, peak_live={sched.peak_live}, "
          f"recompute={sched.multiplier:.2f}x ({sched.recompute_ops} ops)")

    # 3.2 Phase 1.1 — the reassociation (spill→0) prediction.
    pred = predict_reassociation(dag, budget=255)
    print(f"\n>> reassociation prediction (Phase 1.1, fp32, budget {pred.budget} regs):")
    print(f"   temps = {pred.n_temps}  ({pred.n_multiuse} multi-use / "
          f"{pred.n_singleuse} single-use → only single-use is reassociable)")
    print(f"   store-everything peak live:  file order {pred.file_live_fp32}, "
          f"min-liveness reorder {pred.reorder_live_fp32}  (ptxas-achievable reorder)")
    print(f"   reassociation FLOOR (multi-use peak): {pred.floor_fp32} fp32 "
          f"/ {pred.floor_fp64} fp64")
    verdict = "VIABLE — emit + validate" if pred.viable else \
        "DEAD — width is structural; put all weight on 1.2 (derivative fusion)"
    print(f"   verdict: {verdict}")

    p = generate_staged(k=DEFAULT_K, cut_set=[c.name for c in cands[:DEFAULT_K]])
    print(f"\n>> wrote staged module ({DEFAULT_K} barrier cuts): {p}")


if __name__ == "__main__":
    main()
