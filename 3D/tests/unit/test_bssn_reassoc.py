"""Phase 3.2 / 1.1 gate: the straight-line liveness + reassociation prediction.

The spill→0 lever for the fp32 algebra kernel is *reassociation* (binding, where the
3.2b materialize/recompute reorder was not — ptxas re-derives a reorder but cannot
reassociate fp). These tests pin the CPU prediction model and its verdict:

  1. **Model sanity** — fp64 costs 2× fp32; the greedy min-liveness reorder is a valid
     topological order that does not increase the store-everything live set; the
     reassociation floor (multi-use peak) is a true lower bound (<= store-everything).

  2. **The verdict** — the floor sits well above the 255-register file even in fp32, so
     reassociation is DEAD: the width is the genuinely-shared tensor-hierarchy working
     set (757 of 826 temps are multi-use), not a reducible single-use reduction chain.
     Pinned so a refreshed CSE that changes the structure re-triggers the analysis.
"""

import pytest

from bssn3d import staging


@pytest.fixture(scope="module")
def dag():
    return staging.build_dag()


def test_regs_per_value(dag):
    f32 = staging.straight_line_liveness(dag, None, "fp32")
    f64 = staging.straight_line_liveness(dag, None, "fp64")
    assert f64 == 2 * f32                       # fp64 = 2 regs/value


def test_min_liveness_order_is_valid_topological(dag):
    order = staging.min_liveness_order(dag)
    assert sorted(order) == sorted(dag.order)   # a permutation of all nodes
    pos = {n: i for i, n in enumerate(order)}
    nodeset = set(dag.order)
    for n in order:                             # every node-dep precedes its use
        for d in dag.deps[n]:
            if d in nodeset:
                assert pos[d] < pos[n]


def test_reorder_does_not_worsen_liveness(dag):
    file_live = staging.straight_line_liveness(dag, None, "fp32")
    reorder_live = staging.straight_line_liveness(
        dag, staging.min_liveness_order(dag), "fp32")
    assert reorder_live <= file_live            # greedy never worse than file order


def test_floor_is_lower_bound(dag):
    """The multi-use floor cannot exceed store-everything pressure, and reassociation
    cannot go below it."""
    floor = staging.reassociation_floor(dag, "fp32")
    store_all = staging.straight_line_liveness(dag, None, "fp32")
    assert 0 < floor <= store_all
    assert staging.reassociation_floor(dag, "fp64") == 2 * floor


def test_verdict_reassociation_is_dead(dag):
    """The structural width exceeds the register file in fp32 → reassociation is dead.

    The floor is set by the ~757 multi-use temps; only ~69 single-use temps are
    reassociable. The greedy reorder barely moves store-everything (the wide output
    block forces the whole tensor hierarchy co-resident).
    """
    pred = staging.predict_reassociation(dag, budget=255)
    assert pred.n_multiuse > 700                # broad sharing, not a spike
    assert pred.n_singleuse < 100
    assert pred.floor_fp32 > pred.budget        # > 255 even in fp32 → cannot fit
    assert not pred.viable
    # greedy reorder removes < 15% of store-everything pressure (a reorder, not binding)
    assert pred.reorder_live_fp32 > 0.85 * pred.file_live_fp32
