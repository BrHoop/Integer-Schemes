"""Strategy-B cost model: partition the BSSN algebra DAG across warp lanes (CPU).

The Step-3.2f wall-B question is whether distributing the ~826-temp algebra across `g`
cooperating lanes keeps each lane register-resident (0 spill, the 1c sweep showed spill is
the cost) at an acceptable shuffle count. Strategy B decides the lane assignment by
**graph partition** (minimize cross-lane edges) on the flat CSE we already have — purely
mechanical, no tensor regen, each temp computed exactly once (no redundant recompute).

This module is the CPU gate that runs BEFORE any CUDA: it partitions the DAG and reports
  * **per-lane peak register load** -> does each lane fit 255 regs (= 0 spill)? *the go/no-go*
  * **shuffle count** = cross-lane (temp -> destination-lane) pairs per point
  * **recompute-able fraction** = cheap cut-temps we could recompute locally instead of
    shuffling (the "recompute > transfer" hybrid)
across g = 4 and g = 8, so we know what we'd be signing up for (and whether g=4 suffices or
g=8 is needed) before writing the kernel. It predicts STRUCTURE exactly; the wall-clock
shuffle-vs-spill tradeoff still needs the calibrated constant (1c sweep gives spill cost;
a `__shfl` microbench gives shuffle cost).

Run:  `python -m bssn3d.warp_partition`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .staging import build_dag, node_consumers, cone_cost, _peak_overlap, DAG

REG_PER_FP64 = 2
REG_FILE = 255


# ---------------------------------------------------------------------------
# Partitioner: greedy (deps-follow) init + Kernighan-Lin-style refinement,
# minimizing cross-lane edges subject to a balance cap. No external deps.
# ---------------------------------------------------------------------------
def partition(dag: DAG, g: int = 4, slack: float = 1.15, passes: int = 12) -> Dict[str, int]:
    order = dag.order
    nodeset = set(order)
    cons = node_consumers(dag)
    cap = int(slack * len(order) / g) + 1

    lane: Dict[str, int] = {}
    load = [0] * g
    for n in order:                                   # greedy in SSA order
        score = [0] * g
        for d in dag.deps[n]:
            if d in lane:
                score[lane[d]] += 1
        best, bkey = None, None
        for L in range(g):
            if load[L] >= cap:
                continue
            key = (-score[L], load[L])
            if bkey is None or key < bkey:
                best, bkey = L, key
        if best is None:
            best = min(range(g), key=lambda x: load[x])
        lane[n] = best
        load[best] += 1

    for _ in range(passes):                           # KL-style refinement
        moved = 0
        for n in order:
            cur = lane[n]
            nbrs = [d for d in dag.deps[n] if d in nodeset] + cons[n]
            if not nbrs:
                continue
            cnt = [0] * g
            for m in nbrs:
                cnt[lane[m]] += 1
            best = cur
            for L in range(g):
                if L == cur or load[L] >= cap:
                    continue
                if cnt[L] > cnt[best] or (cnt[L] == cnt[best] and load[L] < load[best]):
                    best = L
            if best != cur and cnt[best] > cnt[cur]:
                load[cur] -= 1
                load[best] += 1
                lane[n] = best
                moved += 1
        if moved == 0:
            break
    return lane


# ---------------------------------------------------------------------------
# Cost metrics for a partition
# ---------------------------------------------------------------------------
@dataclass
class PartCost:
    g: int
    sizes: List[int]                 # temps per lane
    peak_live: List[int]             # peak simultaneously-live values per lane
    peak_regs: List[int]             # = peak_live * 2 (fp64)
    fits: bool                       # max peak_regs (+ headroom) <= 255
    shuffles: int                    # cross-lane (temp -> dest-lane) pairs
    recompute_cheap: int             # cut-temps cheap enough to recompute instead
    recompute_total: int


def cost(dag: DAG, lane: Dict[str, int], g: int, headroom: int = 32,
         cheap_ops: int = 8) -> PartCost:
    order = dag.order
    pos = {n: i for i, n in enumerate(order)}
    # global last-use: a temp must stay live (on its lane) until its last consumer fires
    last = {n: pos[n] for n in order}
    for n in order:
        for d in dag.deps[n]:
            if d in pos:
                last[d] = max(last[d], pos[n])

    # values received on each lane (used there but computed elsewhere) = the shuffles
    received: List[Dict[str, List[int]]] = [dict() for _ in range(g)]
    for n in order:
        L = lane[n]
        for d in dag.deps[n]:
            if d in pos and lane[d] != L:
                span = received[L].get(d)
                if span is None:
                    received[L][d] = [pos[n], pos[n]]
                else:
                    span[1] = pos[n]
    shuffles = sum(len(r) for r in received)

    sizes, peak_live = [], []
    for L in range(g):
        spans = [(pos[n], last[n]) for n in order if lane[n] == L]
        sizes.append(len(spans))
        spans += [tuple(received[L][t]) for t in received[L]]   # received live windows
        peak_live.append(_peak_overlap(spans))
    peak_regs = [p * REG_PER_FP64 for p in peak_live]
    fits = max(peak_regs) + headroom <= REG_FILE

    cone = cone_cost(dag)
    rc_total = shuffles
    rc_cheap = sum(1 for L in range(g) for t in received[L] if cone[t] <= cheap_ops)
    return PartCost(g, sizes, peak_live, peak_regs, fits, shuffles, rc_cheap, rc_total)


def main() -> None:
    dag = build_dag()
    n_temps = len(dag.temps)
    n_out = sum(dag.is_output.values())
    # naive (single-thread) reference: store-everything peak live
    pos = {n: i for i, n in enumerate(dag.order)}
    last = {n: pos[n] for n in dag.order}
    for n in dag.order:
        for d in dag.deps[n]:
            if d in pos:
                last[d] = max(last[d], pos[n])
    naive_peak = _peak_overlap((pos[n], last[n]) for n in dag.order)

    print(f">> Strategy-B partition cost model | {len(dag.order)} nodes "
          f"({n_temps} temps + {n_out} outputs)")
    print(f">> naive 1-thread peak-live = {naive_peak} fp64 = {naive_peak*REG_PER_FP64} regs "
          f"(>> {REG_FILE} -> the measured ~4 KB spill)")
    print(f">> target: a partition whose max per-lane peak-regs (+headroom) <= {REG_FILE} "
          f"= 0 spill\n")

    for g in (4, 8):
        lane = partition(dag, g=g)
        c = cost(dag, lane, g)
        verdict = "FITS (0 spill)" if c.fits else "STILL SPILLS"
        print(f"== g = {g} ==  {verdict}")
        print(f"   temps/lane     : {c.sizes}")
        print(f"   peak-live/lane : {c.peak_live}  (regs {c.peak_regs}, max {max(c.peak_regs)})")
        print(f"   shuffles/point : {c.shuffles}  (cross-lane temp->dest-lane pairs)")
        print(f"   of which cheap : {c.recompute_cheap}/{c.recompute_total} "
              f"(cone<=8 ops -> recompute locally instead of shuffle)")
        print(f"   net shuffles   : {c.recompute_total - c.recompute_cheap} "
              f"(if cheap cuts are recomputed)\n")

    print(">> read: a lane FITS only if its peak-regs+headroom <= 255 (then 0 spill). The")
    print(">> shuffle count is the on-chip (not-HBM) traffic traded for the spill; the")
    print(">> recompute split is the 'recompute > transfer' hybrid. Wall-clock still needs")
    print(">> the shuffle-cost constant (a __shfl microbench) vs the 1c spill cost (~0.13 ms/KB).")


if __name__ == "__main__":
    main()
