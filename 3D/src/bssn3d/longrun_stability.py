"""Long-run BSSN stability probe (Tier-A4 extended runs) — turnkey CLI.

The committed CPU test (``3D/tests/integration/test_bssn_longrun.py``) covers a SHORT
long-run (2 crossing times, small N). This script runs the EXTENDED configurations —
many light-crossing times at production resolution — that are too heavy for the laptop
and belong on Marylou (H200). It evolves the production CAHD+SSL RHS in chunks and dumps
the constraint history ||H||(t), ||M||(t), max|alpha|, max-deviation to CSV (the series
A5/A6 consume), printing a summary table.

The cubic domain is [-0.5, 0.5]^3 (length L = 1, c = 1) so one crossing time = 1/dt
steps. At N=64 that is 256 steps/crossing (N=96 -> 384), so 20-50 crossings is thousands
of RK4 steps on top of a heavy one-time XLA compile of the verbatim CSE RHS -> these runs
are HOURS at production resolution, not minutes. Two mitigations are built in: KO
dissipation defaults to ``--ko 0.4`` (specifiable per run) to suppress the high-frequency
growth, and a ``--max-alpha`` early-stop halts + reports WHEN a run blows up (the A5
datum) instead of integrating garbage to the full crossing count.

Examples (run on Marylou after ``./sync.sh push``):

    # robust stability, 50 crossing times at N=48 (should stay bounded — the real
    # long-term certificate for this gauge); KO 0.4
    python -m bssn3d.longrun_stability --id robust --N 48 --crossings 50 \
        --samples 50 --amp 1e-8 --ko 0.4 --out robust_N48_50cross.csv

    # resolution sweep, SHORT, to see whether the gauge-wave drift/blow-up is
    # truncation-driven (later blow-up as N rises) vs a real instability
    for N in 24 48 96; do \
      python -m bssn3d.longrun_stability --id gauge --N $N --crossings 5 \
        --samples 20 --ko 0.4 --out gauge_N${N}_5cross.csv ; done

    # long gauge run only once characterized; raise KO if it still blows up early
    python -m bssn3d.longrun_stability --id gauge --N 64 --crossings 20 \
        --ko 0.4 --out gauge_N64_20cross.csv
"""

import argparse
import csv
import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .grid import Grid
from .state import PhysicsParams
from .evolve import BSSNEvolution
from .constraints import ConstraintSolver
from . import initial_data as bid

CFL = 0.25

# Dedicated, sync-shielded home for run CSVs (see docs/bssn_validation_results/README.md
# and .rsyncignore). repo root is 3 parents above this file (3D/src/bssn3d/).
RESULTS_DIR = Path(__file__).resolve().parents[3] / "docs" / "bssn_validation_results"


def _resolve_out(out, id_name, N, crossings):
    """Resolve ``--out`` to an absolute path under the shielded results folder, with a
    unique timestamped name so reruns never overwrite (and ``./sync.sh push --delete``
    never erases them). A ``--out`` that is absolute or contains a path separator is
    honored literally (escape hatch); a bare name lands in the results folder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if out is None:
        return str(RESULTS_DIR / f"{id_name}_N{N}_{crossings:g}cross_{stamp}.csv")
    p = Path(out)
    if p.is_absolute() or len(p.parts) > 1:
        return out
    target = RESULTS_DIR / p.name
    if target.exists():                       # don't clobber a same-named earlier run
        target = RESULTS_DIR / f"{p.stem}_{stamp}{p.suffix}"
    return str(target)


def fit_rates(rows):
    """Fit secular growth/decay rates of ||H|| and ||M|| from a sample series (the
    A5 diagnostic). Returns, per crossing time:
      * ``*_lin``  — linear slope d||C||/dt (units: constraint-norm per crossing).
      * ``*_exp``  — exponential rate d ln||C|| /dt  (e-folds per crossing; <0 = decay,
                     a LARGE positive value flags exponential blow-up).
      * ``M_lin_vs_exp_resid`` — ratio (linear-fit residual)/(exp-fit residual) on ||M||;
                     < 1 means a LINEAR model fits better than exponential, i.e. the drift
                     is sub-exponential (bounded-ish), not a blow-up.
    Needs >= 3 samples. Operates on the dicts ``run``/``evolve_series`` produce.
    """
    import numpy as np
    t = np.array([r["ncross"] for r in rows], dtype=float)
    H = np.array([r["H"] for r in rows], dtype=float)
    M = np.array([r["M"] for r in rows], dtype=float)

    def _lin(y):
        c = np.polyfit(t, y, 1)
        return float(c[0]), float(np.sqrt(np.mean((np.polyval(c, t) - y) ** 2)))

    def _exp(y):
        ly = np.log(np.maximum(y, 1e-300))
        c = np.polyfit(t, ly, 1)
        resid = float(np.sqrt(np.mean((np.exp(np.polyval(c, t)) - y) ** 2)))
        return float(c[0]), resid

    H_lin, _ = _lin(H)
    M_lin, M_lin_resid = _lin(M)
    H_exp, _ = _exp(H)
    M_exp, M_exp_resid = _exp(M)
    return dict(
        H_lin=H_lin, M_lin=M_lin, H_exp=H_exp, M_exp=M_exp,
        M_lin_vs_exp_resid=float(M_lin_resid / max(M_exp_resid, 1e-300)),
    )


def _id_fn(name, amp):
    if name == "gauge":
        return lambda g: bid.gauge_wave(g, amplitude=amp, wavelength=1.0)
    if name == "robust":
        return lambda g: bid.robust_stability(g, amp=amp, seed=1)
    if name == "minkowski":
        return lambda g: bid.minkowski(g)
    if name == "hambump":
        # coherent Hamiltonian-constraint violation (A6 CAHD certificate); ``amp``
        # is the conformal-factor perturbation amplitude.
        return lambda g: bid.hamiltonian_bump(g, amplitude=amp)
    raise ValueError(f"unknown --id {name!r} (gauge|robust|minkowski|hambump)")


def run(id_name="gauge", N=64, crossings=20.0, samples=40, ko_sigma=0.4,
        scheme="verbatim", amp=0.01, out=None, max_alpha_stop=10.0, cahd_c=None):
    g = Grid.from_domain(N, order=6)
    # ``cahd_c=None`` uses the production default (0.06); A6 overrides it to vary the
    # Hamiltonian-constraint-damping strength and measure the resulting decay rate.
    params = PhysicsParams() if cahd_c is None else PhysicsParams(cahd_c=cahd_c)
    ev = BSSNEvolution(g, params, order=6, ko_sigma=ko_sigma,
                       bc="periodic", scheme=scheme)
    cs = ConstraintSolver(g, order=6)
    dt = CFL * g.dx
    total = round(crossings / dt)             # crossing time = L = 1
    chunk = max(1, round(total / samples))
    mink = bid.minkowski(g).data

    state = _id_fn(id_name, amp)(g)
    t = 0.0
    rows = []

    def sample(s, t):
        H, M = cs.l2(s)
        rows.append(dict(
            ncross=float(t), steps=round(t / dt),
            H=float(H), M=float(M),
            max_alpha=float(jnp.max(jnp.abs(s.alpha))),
            max_dev=float(jnp.max(jnp.abs(s.data - mink))),
            finite=bool(jnp.all(jnp.isfinite(s.data))),
        ))

    print(f"# {id_name}  N={N}  scheme={scheme}  ko={ko_sigma}  cahd_c={params.cahd_c}  "
          f"dt={dt:.5e}  steps/crossing={round(1/dt)}  total_steps={total}  "
          f"chunk={chunk}  max_alpha_stop={max_alpha_stop}")
    sample(state, t)
    blew_up = False
    for _ in range(samples):
        state = ev.evolve(state, dt, chunk, t0=t)
        t += chunk * dt
        sample(state, t)
        r = rows[-1]
        print(f"  t={r['ncross']:8.3f}cross  steps={r['steps']:7d}  "
              f"||H||={r['H']:.4e}  ||M||={r['M']:.4e}  "
              f"max|a|={r['max_alpha']:.5f}  max-dev={r['max_dev']:.3e}  "
              f"finite={r['finite']}")
        # early stop: a diverging run reports WHEN it blew up (the A5 datum) instead of
        # burning GPU hours integrating garbage to the full crossing count.
        if (not r["finite"]) or (r["max_alpha"] > max_alpha_stop):
            blew_up = True
            print(f"# BLEW UP at t={r['ncross']:.3f} crossing times "
                  f"(steps={r['steps']}): finite={r['finite']}, "
                  f"max|alpha|={r['max_alpha']:.3e} > {max_alpha_stop} — stopping early.")
            break
    if not blew_up:
        print(f"# completed {crossings} crossing times bounded "
              f"(max|alpha|={max(r['max_alpha'] for r in rows):.5f}).")

    if out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {len(rows)} samples -> {out}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Long-run BSSN stability probe (A4 extended).")
    p.add_argument("--id", default="gauge",
                   choices=["gauge", "robust", "minkowski", "hambump"])
    p.add_argument("--N", type=int, default=64, help="interior cells per axis")
    p.add_argument("--crossings", type=float, default=20.0, help="light-crossing times")
    p.add_argument("--samples", type=int, default=40, help="constraint samples along the run")
    p.add_argument("--ko", type=float, default=0.4, dest="ko_sigma",
                   help="KO dissipation sigma (default 0.4; specifiable per run)")
    p.add_argument("--scheme", default="verbatim",
                   choices=["verbatim", "staged", "pallas", "fused", "fused_fp64"])
    p.add_argument("--amp", type=float, default=0.01,
                   help="gauge-wave amplitude / robust-noise amplitude")
    p.add_argument("--max-alpha", type=float, default=10.0, dest="max_alpha_stop",
                   help="early-stop ceiling on max|alpha| (blow-up detector)")
    p.add_argument("--out", default=None,
                   help="CSV name (default: auto, timestamped, in "
                        "docs/bssn_validation_results/). A bare name lands there too; "
                        "pass a path with a separator to write elsewhere.")
    a = p.parse_args()
    out = _resolve_out(a.out, a.id, a.N, a.crossings)
    run(id_name=a.id, N=a.N, crossings=a.crossings, samples=a.samples,
        ko_sigma=a.ko_sigma, scheme=a.scheme, amp=a.amp, out=out,
        max_alpha_stop=a.max_alpha_stop)


if __name__ == "__main__":
    main()
