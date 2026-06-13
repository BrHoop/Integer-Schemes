"""Assessment of MSRK time integrators on the 2D MCS solver.

Answers the question posed before any 3D/BSSN commitment: *which* MSRK scheme
(if any) is worth carrying forward, and in which regime.  For each of RK4,
RK4-2(1), RK4-2(2), RK4-3 it measures, on the production MCS configuration:

  1. Temporal convergence order   — self-convergence in dt at fixed grid
                                     (cancels spatial error exactly → pure
                                     integrator order; must be ~4).
  2. Max stable CFL               — analytic von Neumann limit from the
                                     linearized semi-discrete symbol (full
                                     Brillouin zone, incl. FD dispersion, CS
                                     mass, KO, constraint damping).
  3. Effective CFL (ECF)          — CFL_max / n_stages  (the paper's efficiency
                                     metric: larger = cheaper per unit time).
  4. Error at a fixed CFL         — L2 error vs the 10-field oracle at a common
                                     CFL, to compare error constants.

Run:  python -m mcs2d.msrk_analysis            (prints table + writes figure)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

import sys
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from mcs_common.jax_config import setup as _jax_setup
_jax_setup()

import jax
import jax.numpy as jnp

from mcs2d.main import (MaxwellChernSimons2D, InitialData, load_parameters,
                        get_physical)
from mcs2d import msrk
from mcs2d import validate as V

_PARAMS_FILE = str(Path(__file__).resolve().parent.parent.parent / "params.toml")
_RESULTS_DIR = str(Path(__file__).resolve().parents[3]
                   / "docs/phases/phase_0_2d_foundation/step_0.1_results")

ALL_METHODS = ["rk4", "rk4_2_1", "rk4_2_2", "rk4_3"]
LABELS = {"rk4": "RK4", "rk4_2_1": "RK4-2(1)",
          "rk4_2_2": "RK4-2(2)", "rk4_3": "RK4-3"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_sim(nx: int, ny: int, **overrides):
    params = load_parameters(_PARAMS_FILE)
    params.update({"scheme": "floating_point", "Nx": nx, "Ny": ny,
                   "id_type": "birefringent", "bc_type": "periodic",
                   "sponge_strength": 0.0, **overrides})
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / ny
    sim = MaxwellChernSimons2D(dx, dy, params["Lambda"], params)
    state = InitialData(sim, params).generate()
    return sim, state, params


# ── 1. temporal convergence order (self-convergence in dt) ─────────────────────

def convergence_order(method: str, *, nx: int = 64, t_phys: float = 0.5,
                      base_steps: int = 200) -> float:
    """Richardson self-convergence order in time at a fixed grid.

    Integrate to the same physical time t_phys with N, 2N, 4N steps; the spatial
    operator is identical so spatial error cancels in the differences:
        order = log2( ||y_N - y_2N|| / ||y_2N - y_4N|| ).
    Expect ~4 (RK4 startup preserves order).
    """
    sim, state0, _ = _build_sim(nx, nx)
    sols = []
    for mult in (1, 2, 4):
        n = base_steps * mult
        dt = t_phys / n
        sf = msrk.evolve(sim, state0, dt, n, method)
        sols.append(np.asarray(get_physical(sf.data, sim.ng)))
    d1 = np.sqrt(np.mean((sols[0] - sols[1]) ** 2))
    d2 = np.sqrt(np.mean((sols[1] - sols[2]) ** 2))
    return float(np.log2(d1 / d2))


# ── 2/3. max stable CFL + ECF (analytic von Neumann) ───────────────────────────

def _semidiscrete_eigs(params, dx, dy, *, n_k: int = 96) -> np.ndarray:
    """All eigenvalues of the 10x10 MCS symbol over the Brillouin zone.

    Sweeps k*dx, k*dy in [-pi, pi] (the resolvable band) so the spectrum is the
    full semi-discrete operator the time integrator actually sees: 6th-order FD
    dispersion + CS mass + KO + constraint damping.
    """
    p = V.symbol_params(params, dx, dy)
    ts = np.linspace(-np.pi, np.pi, n_k)
    kxs, kys = ts / dx, ts / dy
    eigs: List[complex] = []
    for kx in kxs:
        for ky in kys:
            M = V._symbol(kx, ky, p, with_ko=True, with_cs=True)
            eigs.extend(np.linalg.eigvals(M).tolist())
    return np.asarray(eigs)


def max_stable_cfl(method: str, params, dx, dy, eigs=None) -> float:
    if eigs is None:
        eigs = _semidiscrete_eigs(params, dx, dy)
    dt_max = msrk.max_stable_dt(eigs, method)
    return dt_max / dx


# ── 4. error at fixed CFL vs the 10-field oracle ───────────────────────────────

def error_at_cfl(method: str, cfl: float, *, nx: int = 96,
                 t_phys: float = 0.5) -> float:
    sim, state0, params = _build_sim(nx, nx, cfl=cfl)
    dt = cfl * sim.dx
    n = int(round(t_phys / dt))
    t_actual = n * dt                       # exact integer-step physical time
    sf = msrk.evolve(sim, state0, dt, n, method)
    errs = V.field_l2_errors(sim, sf, params, t_actual)
    return float(np.sqrt(np.mean([e ** 2 for e in errs.values()])))


# ── driver ──────────────────────────────────────────────────────────────────

def run(outdir: str = _RESULTS_DIR) -> Dict[str, dict]:
    params = load_parameters(_PARAMS_FILE)
    # representative production grid for the stability spectrum
    nx_prod = params.get("Nx", 512)
    dx = (params["xmax"] - params["xmin"]) / nx_prod
    dy = (params["ymax"] - params["ymin"]) / nx_prod
    op_cfl = params.get("cfl", 0.05)

    print("Building semi-discrete spectrum (production grid, full Brillouin zone)...")
    eigs = _semidiscrete_eigs(params, dx, dy)
    print(f"  {len(eigs)} eigenvalues; max|Im|={np.max(np.abs(eigs.imag)):.3g}, "
          f"max|Re|={np.max(np.abs(eigs.real)):.3g}\n")

    rows = {}
    for k in ALL_METHODS:
        cfl_max = max_stable_cfl(k, params, dx, dy, eigs=eigs)
        ecf = cfl_max / msrk.STAGES[k]
        order = convergence_order(k)
        err_op = error_at_cfl(k, op_cfl)           # error at the MCS operating CFL
        rows[k] = dict(stages=msrk.STAGES[k], bufs=msrk.PREV_BUFFERS[k],
                       cfl_max=cfl_max, ecf=ecf, order=order, err_op=err_op,
                       intercept=msrk.imag_axis_intercept(k))

    # also: error at a near-limit CFL (the regime where stability bites)
    near_cfl = 0.6 * min(r["cfl_max"] for r in rows.values())
    for k in ALL_METHODS:
        rows[k]["err_near"] = error_at_cfl(k, near_cfl)
    rows["_meta"] = dict(op_cfl=op_cfl, near_cfl=near_cfl,
                         nx_prod=nx_prod, dx=dx)

    _print_table(rows)
    try:
        _plot(rows, eigs, dx, outdir)
    except Exception as e:                          # matplotlib optional
        print(f"[plot skipped: {e}]")
    return rows


def _print_table(rows: Dict[str, dict]) -> None:
    meta = rows["_meta"]
    print("=" * 78)
    print(f"MSRK assessment on 2D MCS  (production grid Nx={meta['nx_prod']}, "
          f"dx={meta['dx']:.4g}, KO=0.05, K1=K2=1, Lambda=0.4)")
    print("=" * 78)
    print(f"{'method':9s} {'stages':>6s} {'bufs':>4s} {'ASRimag':>8s} "
          f"{'CFLmax':>7s} {'ECF':>7s} {'order':>6s}")
    for k in ALL_METHODS:
        r = rows[k]
        print(f"{LABELS[k]:9s} {r['stages']:6d} {r['bufs']:4d} "
              f"{r['intercept']:8.3f} {r['cfl_max']:7.3f} {r['ecf']:7.4f} "
              f"{r['order']:6.2f}")
    print(f"\nError vs oracle (RMS over fields, t=0.5):")
    print(f"{'method':9s} {'@CFL=%.3g'%meta['op_cfl']:>14s} "
          f"{'@CFL=%.3g'%meta['near_cfl']:>14s}")
    for k in ALL_METHODS:
        r = rows[k]
        print(f"{LABELS[k]:9s} {r['err_op']:14.3e} {r['err_near']:14.3e}")
    print("=" * 78)
    # interpretation
    best_ecf = max(ALL_METHODS, key=lambda k: rows[k]["ecf"])
    print(f"Highest ECF (efficiency near stability limit): {LABELS[best_ecf]} "
          f"({rows[best_ecf]['ecf']:.4f})")
    print(f"At the MCS operating CFL={meta['op_cfl']:.3g}: all methods sit far "
          f"inside their ASRs (CFLmax >> {meta['op_cfl']:.3g}),")
    print(f"  so cost ∝ stages → RK4-3 (2 stages) is cheapest there; the "
          f"stability penalty only bites near CFLmax.")


def _plot(rows, eigs, dx, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) absolute-stability regions along the imaginary axis + ECF context
    ax = axes[0]
    B = np.linspace(0, 3.2, 400)
    for k in ALL_METHODS:
        rad = np.array([msrk.stability_radius(1j * b, k) for b in B])
        ax.plot(B, rad, label=f"{LABELS[k]} (ic={rows[k]['intercept']:.2f})")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("imag(z) = dt·|Im λ|");  ax.set_ylabel("stability radius")
    ax.set_title("Absolute stability along the imaginary axis")
    ax.set_ylim(0.9, 1.2);  ax.legend(fontsize=8)

    # (b) ECF bar chart
    ax = axes[1]
    xs = np.arange(len(ALL_METHODS))
    ecfs = [rows[k]["ecf"] for k in ALL_METHODS]
    cflm = [rows[k]["cfl_max"] for k in ALL_METHODS]
    ax.bar(xs - 0.2, cflm, 0.4, label="CFL_max")
    ax.bar(xs + 0.2, ecfs, 0.4, label="ECF = CFL_max/stages")
    ax.set_xticks(xs);  ax.set_xticklabels([LABELS[k] for k in ALL_METHODS])
    ax.set_title("Max stable CFL and effective CFL")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = f"{outdir}/msrk_assessment.png"
    fig.savefig(path, dpi=140)
    print(f"\nFigure → {path}")


if __name__ == "__main__":
    run()
