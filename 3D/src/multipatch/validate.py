"""Research-figure harness for the Llama multipatch prototype.

Mirrors ``mcs3d/validate.py``: a CPU-runnable script that regenerates the
characterization figures behind the milestones. Cheap figures (geometry,
derivative/interpolation convergence) run by default; the evolution
convergence figures (``--full``) compile the 7-patch step and are slow.

    python -m multipatch.validate            # geometry + derivative + interp
    python -m multipatch.validate --full     # + evolved wave/MCS convergence
    python -m multipatch.validate --outdir DIR
"""
import argparse
import os

import numpy as np
import jax.numpy as jnp

from mcs_common.jax_config import setup

from . import atlas as A, overlap as O, coord_maps as cm
from .derivative_curvilinear import CurvilinearDerivative

KA, KB, KC = 0.7, -0.4, 0.5


def _f(X, Y, Z):
    return jnp.sin(KA * X + KB * Y + KC * Z)


def _grid(N, order=6):
    return A.build_llama_grid(2.0, 1.8, 8.0, N, N, N, order=order)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_slice(outdir):
    """z~0 slice of the 7-patch tiling, points coloured by patch — shows the
    central cube wrapped by the cubed-sphere shells and their overlaps."""
    import matplotlib.pyplot as plt
    g = _grid(25)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for pi, p in enumerate(g.patches):
        X = np.asarray(p.X); Y = np.asarray(p.Y); Z = np.asarray(p.Z)
        sel = np.abs(Z) < 0.2          # thin z~0 slab
        ax.scatter(X[sel], Y[sel], s=2, color=colors[pi], label=p.name, alpha=0.5)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Llama 7-patch grid (z≈0 slice): cube + 6 cubed-sphere shells")
    ax.legend(markerscale=4, fontsize=8, loc="upper right")
    path = os.path.join(outdir, "grid_slice.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def fig_coverage(outdir):
    import matplotlib.pyplot as plt
    g = _grid(17)
    s = A.coverage_report(g)
    names = list(s["per_patch"].keys())
    cov = [s["per_patch"][n]["covered"] for n in names]
    bnd = [s["per_patch"][n]["boundary"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, cov, label="donor-covered ghosts")
    ax.bar(x, bnd, bottom=cov, label="outer-boundary ghosts (BC)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("ghost nodes"); ax.set_title(
        f"Ghost coverage (holes = {len(s['holes'])})")
    ax.legend()
    path = os.path.join(outdir, "coverage.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def fig_derivative_convergence(outdir):
    import matplotlib.pyplot as plt
    Ns = [13, 19, 27, 37]
    e1, elap, emix = [], [], []
    for N in Ns:
        sh = _grid(N).shells[0]
        d = CurvilinearDerivative(sh, order=6)
        F = _f(sh.X, sh.Y, sh.Z)
        intr = sh.interior
        s = jnp.sin(KA*sh.X+KB*sh.Y+KC*sh.Z)
        Dx, _, _ = d.grad(F)
        e1.append(float(jnp.max(jnp.abs((Dx - KA*jnp.cos(KA*sh.X+KB*sh.Y+KC*sh.Z))[intr]))))
        elap.append(float(jnp.max(jnp.abs((d.laplacian(F) + (KA*KA+KB*KB+KC*KC)*s)[intr]))))
        emix.append(float(jnp.max(jnp.abs((d.d2_world(F, 0, 1) + KA*KB*s)[intr]))))
    h = 1.0 / (np.array(Ns) - 1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lbl, e in [("1st deriv", e1), ("laplacian", elap), ("mixed d2", emix)]:
        ax.loglog(h, e, "o-", label=lbl)
    ax.loglog(h, e1[0] * (h / h[0]) ** 6, "k--", alpha=0.5, label="6th order")
    ax.set_xlabel("h (logical)"); ax.set_ylabel("max interior error")
    ax.set_title("Curvilinear derivative convergence (shell)"); ax.legend()
    path = os.path.join(outdir, "derivative_convergence.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def fig_interp_convergence(outdir):
    import matplotlib.pyplot as plt
    Ns = [13, 19, 27, 37]
    errs = []
    for N in Ns:
        g = _grid(N); tbl = O.build_overlap_table(g, order=6)
        fields = [_f(p.X, p.Y, p.Z)[None] for p in g.patches]
        filled = O.apply_overlap_fill(fields, tbl)
        me = 0.0
        for e in tbl.entries:
            P = g.patches[e.recv]
            Xf = jnp.asarray(P.X).ravel()[e.tgt_idx]
            Yf = jnp.asarray(P.Y).ravel()[e.tgt_idx]
            Zf = jnp.asarray(P.Z).ravel()[e.tgt_idx]
            got = filled[e.recv].reshape(1, -1)[0, e.tgt_idx]
            me = max(me, float(jnp.max(jnp.abs(got - _f(Xf, Yf, Zf)))))
        errs.append(me)
    h = 1.0 / (np.array(Ns) - 1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(h, errs, "o-", label="overlap interp error")
    ax.loglog(h, errs[0] * (h / h[0]) ** 7, "k--", alpha=0.5, label="7th order")
    ax.set_xlabel("h (logical)"); ax.set_ylabel("max ghost interp error")
    ax.set_title("Inter-patch interpolation convergence"); ax.legend()
    path = os.path.join(outdir, "interp_convergence.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def fig_evolution_convergence(outdir):
    """Slow: evolved wave + MCS cross-seam convergence vs the exact solutions."""
    import matplotlib.pyplot as plt
    from . import wave as W, mcs_multipatch as M
    from .evolve import MultipatchEvolution, make_exact_dirichlet_bc
    WK = (0.7, -0.4, 0.5); MK = (0.8, 0.6, 0.5); LAM = 0.1
    Ns = [15, 23, 31]

    def wave_err(N, dt=0.01, ns=40):
        g = _grid(N); tbl = O.build_overlap_table(g, 6)
        bc = make_exact_dirichlet_bc(g, tbl, lambda X, Y, Z, t: W.plane_wave_state(X, Y, Z, t, k=WK))
        ev = MultipatchEvolution(g, tbl, W.WaveSystem(), 6, 0.0, bc)
        fF, _ = ev.evolve(W.plane_wave_initial_data(g, 0.0, k=WK), dt, ns)
        T = dt * ns; m = 0.0
        for p, F in zip(g.patches, fF):
            ex = W.plane_wave_state(p.X, p.Y, p.Z, T, k=WK)
            m = max(m, float(jnp.max(jnp.abs((F - ex)[(slice(None),) + p.interior]))))
        return m

    def mcs_err(N, dt=0.01, ns=40):
        g = _grid(N); tbl = O.build_overlap_table(g, 6)
        bc = make_exact_dirichlet_bc(g, tbl, lambda X, Y, Z, t: M.mcs_exact_state(X, Y, Z, t, k=MK, Lambda=LAM))
        ev = MultipatchEvolution(g, tbl, M.MCSSystem(Lambda=LAM), 6, 0.0, bc)
        fF, _ = ev.evolve(M.mcs_initial_data(g, 0.0, k=MK, Lambda=LAM), dt, ns)
        T = dt * ns; m = 0.0
        for p, F in zip(g.patches, fF):
            ex = M.mcs_exact_state(p.X, p.Y, p.Z, T, k=MK, Lambda=LAM)
            m = max(m, float(jnp.max(jnp.abs((F - ex)[(slice(None),) + p.interior]))))
        return m

    we = [wave_err(N) for N in Ns]
    me = [mcs_err(N) for N in Ns]
    h = 1.0 / (np.array(Ns) - 1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(h, we, "o-", label="scalar wave")
    ax.loglog(h, me, "s-", label="MCS (birefringent)")
    ax.loglog(h, we[0] * (h / h[0]) ** 6, "k--", alpha=0.5, label="6th order")
    ax.set_xlabel("h (logical)"); ax.set_ylabel("max interior error vs exact")
    ax.set_title("Evolved cross-seam convergence (full 7-patch)"); ax.legend()
    path = os.path.join(outdir, "evolution_convergence.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="multipatch_figures")
    ap.add_argument("--full", action="store_true",
                    help="also run the slow evolved-convergence figure")
    args = ap.parse_args()
    setup(cache=True)
    os.makedirs(args.outdir, exist_ok=True)
    print("grid slice         ->", fig_slice(args.outdir))
    print("coverage           ->", fig_coverage(args.outdir))
    print("derivative conv    ->", fig_derivative_convergence(args.outdir))
    print("interp conv        ->", fig_interp_convergence(args.outdir))
    if args.full:
        print("evolution conv     ->", fig_evolution_convergence(args.outdir))


if __name__ == "__main__":
    main()
