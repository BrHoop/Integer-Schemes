"""Phase 3.2b gate: the materialize/recompute SCHEDULE generator.

Two things to prove on CPU before the schedule drives a Pallas kernel (3.2c):

  1. **Correctness** — the partition into a materialize set M (stored) + recompute set
     R (inlined per use) is pure substitution, so the scheduled algebra must equal the
     verbatim Dendro-GR RHS to round-off, for *any* M (small K, the budget-selected
     schedule, and store-everything). A substitution bug shows up as a value mismatch.

  2. **The liveness/cost model is sane** — recompute-everything (M=∅) costs the most
     ops at 0 persistent registers; store-everything (M=all) is 1.0x ops at the peak
     liveness; persistent liveness is monotone in K; the budget selector fits.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bssn3d import staging, oracle
from bssn3d._codegen import RHS_TO_FIELD
from bssn3d._bssn_rhs_generated import bssn_rhs_algebra as verbatim

TOL = 1e-10   # round-off bar — recompute reorders fp summation


def _verbatim(values):
    F = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["fields"].items()}
    D = {k: jnp.asarray([v], dtype=jnp.float64) for k, v in values["derivs"].items()}
    out = verbatim(F, D, values["eta"], values["lmbda"], values["lambda_f"],
                   values["BSSN_CAHD_C"], values["dt"], values["dx_i"],
                   values["h_ssl"], values["sig_ssl"], values["t"])
    return {k: float(np.asarray(v)[0]) for k, v in out.items()}


def _eval_schedule(M, values, dag):
    """Exec the scheduled assignment lines on single-point inputs."""
    lines = staging.schedule_pylines(dag, M)
    ns = {"jnp": jnp, "pow": pow}
    for k, v in values["fields"].items():
        ns[k] = jnp.asarray([v], dtype=jnp.float64)
    for k, v in values["derivs"].items():
        ns[k] = jnp.asarray([v], dtype=jnp.float64)
    ns.update(eta=values["eta"], lmbda=values["lmbda"], lambda_f=values["lambda_f"],
              BSSN_CAHD_C=values["BSSN_CAHD_C"], dt=values["dt"], dx_i=values["dx_i"],
              h_ssl=values["h_ssl"], sig_ssl=values["sig_ssl"], t=values["t"])
    for ln in lines:
        exec(ln, ns)
    return {field: float(np.asarray(ns[tok])[0]) for tok, field in RHS_TO_FIELD.items()}


@pytest.fixture(scope="module")
def dag():
    return staging.build_dag()


@pytest.mark.parametrize("k", [0, 64, 288, 826])   # recompute-all → store-all
@pytest.mark.parametrize("seed", [0, 3])
def test_schedule_equals_verbatim(dag, k, seed):
    ranked = [c.name for c in staging.rank_candidates(dag)]
    M = ranked[:k]
    values = oracle.random_inputs(seed)
    got = _eval_schedule(M, values, dag)
    ref = _verbatim(values)                       # keyed by field name (alpha, chi, ...)
    worst, where = 0.0, None
    for field in ref:
        rel = abs(got[field] - ref[field]) / max(abs(ref[field]), 1e-12)
        if rel > worst:
            worst, where = rel, field
    assert worst < TOL, f"K={k} seed={seed}: max rel diff {worst:.2e} on {where}"


def test_liveness_cost_endpoints(dag):
    n = len(dag.temps)
    base = sum(dag.op_cost.values())
    # recompute-everything: 0 persistent registers, ops blow up
    assert staging.persistent_liveness(dag, []) == 0
    assert staging.recompute_ops(dag, []) > 10 * base
    # store-everything: 1.0x ops, peak liveness is the full spill
    ranked = [c.name for c in staging.rank_candidates(dag)]
    assert staging.recompute_ops(dag, ranked) == base
    assert staging.persistent_liveness(dag, ranked) > 255   # the spill we must beat


def test_liveness_monotone_in_k(dag):
    ranked = [c.name for c in staging.rank_candidates(dag)]
    lives = [staging.persistent_liveness(dag, ranked[:k]) for k in (0, 64, 128, 256, 512)]
    assert lives == sorted(lives)            # more materialized → more persistent live


def test_budget_selector_fits(dag):
    sched = staging.select_schedule(dag, budget=200)
    assert sched.peak_live <= 200
    assert sched.multiplier < 2.0            # cheap recompute at this budget
    assert len(sched.materialize) > 0
