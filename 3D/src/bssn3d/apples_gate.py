"""Apples validation GATE (Step 3.4 item-1) — the safety net that replaces the bit-oracle.

A reassociating codegen change (e-graph, or any FMA-reordering kernel) changes rounding, so the
1.93e-16 Dendro bit-compare oracle is LOST. This gate is the replacement acceptance test: evolve a
set of apples-with-apples configurations with BOTH the verbatim RHS and a candidate `scheme`, and
require the candidate's constraint trajectories to TRACK verbatim within an accumulated-round-off
tolerance (NOT bit-identical — CUDA FMA-contracts differently than XLA). It also requires the two
to agree on the qualitative certificate (both damp H / both bounded / same blow-up verdict).

Configs (small N, few crossings -> a fast GATE, not a science campaign):
  * robust   — Minkowski + random noise; CAHD must DAMP the Hamiltonian constraint (bounded).
  * gauge    — forced gauge wave; the under-protected momentum channel drifts; must stay bounded.
  * hambump  — coherent Hamiltonian-constraint bump; CAHD decay certificate.

Usage:
  # the real gate (GPU; verbatim is a heavy XLA compile, so this is minutes on Marylou):
  python -m bssn3d.apples_gate --scheme cuda_fused
  # CPU smoke test of the harness itself (verbatim-vs-verbatim -> exact match):
  python -m bssn3d.apples_gate --scheme verbatim --quick

Exit code 0 = PASS (candidate is physics-equivalent to verbatim), 1 = FAIL.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List

import jax
jax.config.update("jax_enable_x64", True)

from .longrun_stability import run


@dataclass
class Config:
    id_name: str
    N: int
    crossings: float
    samples: int
    ko_sigma: float
    amp: float
    note: str


# Default gate suite — deliberately small so the whole gate is a handful of minutes on GPU.
GATE_SUITE: List[Config] = [
    Config("robust",  24, 2.0, 8, 0.1, 1e-8, "CAHD damps H on Minkowski+noise (bounded)"),
    Config("gauge",   24, 2.0, 8, 0.1, 1e-2, "gauge-wave momentum drift, bounded"),
    Config("hambump", 24, 2.0, 8, 0.1, 1e-3, "CAHD decay on a coherent H bump"),
]

_METRICS = ("H", "M", "max_alpha", "max_dev")


def _rel(a: float, b: float, floor: float) -> float:
    """Mixed abs/rel deviation, floored so near-zero constraints don't blow the ratio up."""
    return abs(a - b) / (abs(b) + floor)


def compare_one(cfg: Config, scheme: str, floor: float) -> dict:
    """Run verbatim + candidate for one config; return per-metric max deviation + verdict match."""
    print(f"\n=== apples gate :: {cfg.id_name} (N={cfg.N}, {cfg.crossings} cross) — {cfg.note} ===")
    print(f"--- verbatim ---")
    ref = run(cfg.id_name, N=cfg.N, crossings=cfg.crossings, samples=cfg.samples,
              ko_sigma=cfg.ko_sigma, scheme="verbatim", amp=cfg.amp)
    print(f"--- candidate: {scheme} ---")
    cand = run(cfg.id_name, N=cfg.N, crossings=cfg.crossings, samples=cfg.samples,
               ko_sigma=cfg.ko_sigma, scheme=scheme, amp=cfg.amp)

    n = min(len(ref), len(cand))
    dev = {m: 0.0 for m in _METRICS}
    for i in range(n):
        for m in _METRICS:
            dev[m] = max(dev[m], _rel(cand[i][m], ref[i][m], floor))
    # qualitative verdicts must agree: same number of samples (same blow-up point) + same finiteness
    verdict_match = (len(ref) == len(cand)) and all(
        ref[i]["finite"] == cand[i]["finite"] for i in range(n))
    return dict(id=cfg.id_name, dev=dev, n_ref=len(ref), n_cand=len(cand),
                verdict_match=verdict_match)


def gate(scheme: str, tol: float = 1e-6, floor: float = 1e-12,
         suite: List[Config] = GATE_SUITE, quick: bool = False) -> bool:
    if quick:
        suite = suite[:1]
    results = [compare_one(c, scheme, floor) for c in suite]

    print("\n" + "=" * 72)
    print(f"APPLES GATE SUMMARY :: candidate scheme = {scheme!r}   tol = {tol:.1e}")
    print(f"{'config':>10} | {'maxdev H':>10} {'maxdev M':>10} {'max|a|':>10} "
          f"{'max-dev':>10} | verdict")
    overall = True
    for r in results:
        d = r["dev"]
        worst = max(d.values())
        ok = (worst <= tol) and r["verdict_match"]
        overall &= ok
        vflag = "ok" if r["verdict_match"] else f"MISMATCH({r['n_ref']}!={r['n_cand']})"
        print(f"{r['id']:>10} | {d['H']:>10.2e} {d['M']:>10.2e} {d['max_alpha']:>10.2e} "
              f"{d['max_dev']:>10.2e} | {vflag} {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    print(f">> {'PASS' if overall else 'FAIL'} — candidate {scheme!r} is "
          f"{'physics-equivalent to' if overall else 'NOT equivalent to'} verbatim "
          f"(within {tol:.0e} relative over the apples suite).")
    if scheme == "verbatim":
        print("   (verbatim-vs-verbatim should be ~0 — this run only checks the harness.)")
    return overall


def main() -> int:
    ap = argparse.ArgumentParser(description="Apples validation gate (item-1).")
    ap.add_argument("--scheme", default="cuda_fused",
                    help="candidate scheme to validate against verbatim (default: cuda_fused)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max relative trajectory deviation to PASS (default 1e-6)")
    ap.add_argument("--quick", action="store_true",
                    help="run only the first config (harness smoke test)")
    a = ap.parse_args()
    return 0 if gate(a.scheme, tol=a.tol, quick=a.quick) else 1


if __name__ == "__main__":
    sys.exit(main())
