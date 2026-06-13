"""
Visual sanity check for the MCS 2D solver.

Generates:
  output/wave_comparison_{scheme}.gif  — 3-panel animation: numerical Ez, exact Ez, |error|
  output/constraints_{scheme}.png      — divB and divE L2 norms over time

Usage (from repo root):
    python mcs2d/visualize.py [params.toml] [scheme] [output_dir]
    python mcs2d/visualize.py mcs2d/params.toml ozaki
"""

import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_dir  = str(Path(__file__).resolve().parent)
_root = str(Path(__file__).resolve().parent.parent)
for _p in [_dir, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcs_common.jax_config import setup as _jax_setup
_jax_setup()

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from mcs2d.main import (
    MaxwellChernSimons2D, InitialData,
    calc_constraints, get_physical, load_parameters, l2norm,
)


def _make_oracle(params):
    """Returns exact_Ez(X, Y, t) for the birefringent wave."""
    Lx    = params["xmax"] - params["xmin"]
    Ly    = params["ymax"] - params["ymin"]
    k_x   = 2.0 * jnp.pi / Lx
    k_y   = 2.0 * jnp.pi / Ly
    k     = jnp.sqrt(k_x**2 + k_y**2)
    m_cs  = params.get("id_m_cs", params.get("Lambda", 1.0) * 2.0)
    E0    = params.get("id_amp", 1.0)
    omega = jnp.sqrt(k**2 + m_cs * k)
    return lambda X, Y, t: E0 * jnp.sin(k_x * X + k_y * Y - omega * t)


def main():
    parfile = sys.argv[1] if len(sys.argv) > 1 else str(Path(_dir) / "params.toml")
    scheme  = sys.argv[2] if len(sys.argv) > 2 else "floating_point"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else str(Path(_dir) / "output")
    os.makedirs(out_dir, exist_ok=True)

    params = load_parameters(parfile)
    params.update({
        "scheme":          scheme,
        "id_type":         "birefringent",
        "bc_type":         "periodic",
        "sponge_strength": 0.0,
    })

    nx, ny  = params["Nx"], params["Ny"]
    nt      = params["Nt"]
    out_int = params["output_interval"]
    dx = (params["xmax"] - params["xmin"]) / nx
    dy = (params["ymax"] - params["ymin"]) / ny

    sim      = MaxwellChernSimons2D(dx, dy, params["Lambda"], params)
    state    = InitialData(sim, params).generate()
    exact_Ez = _make_oracle(params)

    x_phys = sim.x[sim.ng:-sim.ng]
    y_phys = sim.y[sim.ng:-sim.ng]
    X, Y   = jnp.meshgrid(x_phys, y_phys, indexing='ij')
    xn, yn = np.array(x_phys), np.array(y_phys)

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(top=0.85)

    exact0 = np.array(exact_Ez(X, Y, 0.0))
    num0   = np.array(get_physical(state.data[sim.EZ], sim.ng))
    vmax   = max(float(np.max(np.abs(exact0))), 1e-10)

    im0 = axes[0].pcolormesh(xn, yn, num0.T,                    cmap="RdBu_r", vmin=-vmax, vmax=vmax,  shading="auto")
    im1 = axes[1].pcolormesh(xn, yn, exact0.T,                  cmap="RdBu_r", vmin=-vmax, vmax=vmax,  shading="auto")
    im2 = axes[2].pcolormesh(xn, yn, np.abs(num0 - exact0).T,   cmap="magma",  vmin=0,     vmax=1e-2,  shading="auto")

    axes[0].set_title("Numerical $E_z$")
    axes[1].set_title("Exact $E_z$")
    axes[2].set_title(r"$|\mathrm{Error}|$")
    for ax, im in zip(axes, [im0, im1, im2]):
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

    # ── JIT-compiled stepper ─────────────────────────────────────────────────
    @jax.jit
    def advance(i, s):
        return sim.step_rk4(s, sim.dt)

    # ── Evolution loop ───────────────────────────────────────────────────────
    steps_log, divB_log, divE_log = [], [], []
    gif_path = os.path.join(out_dir, f"wave_comparison_{scheme}.gif")
    writer   = PillowWriter(fps=15)
    t_wall   = time.perf_counter()

    print(f"\n>> Visualizer | scheme={scheme} | {nx}×{ny} | {nt} steps")
    print(f"   GIF → {gif_path}")

    with writer.saving(fig, gif_path, dpi=120):
        for s in range(0, nt + 1, out_int):
            if s > 0:
                state = jax.lax.fori_loop(0, out_int, advance, state)
                state.data.block_until_ready()

            t_phys    = s * sim.dt
            num_arr   = np.array(get_physical(state.data[sim.EZ], sim.ng))
            exact_arr = np.array(exact_Ez(X, Y, t_phys))
            err_arr   = np.abs(num_arr - exact_arr)

            l2_err  = float(l2norm(err_arr))
            divE, divB = calc_constraints(sim, state)
            divB_l2 = float(l2norm(get_physical(divB, sim.ng)))
            divE_l2 = float(l2norm(get_physical(divE, sim.ng)))
            steps_log.append(s)
            divB_log.append(divB_l2)
            divE_log.append(divE_l2)

            im0.set_array(num_arr.T.ravel())
            im1.set_array(exact_arr.T.ravel())
            im2.set_array(err_arr.T.ravel())
            im2.set_clim(0, max(float(np.max(err_arr)), 1e-15))
            fig.suptitle(
                f"scheme={scheme}  |  step {s:05d}/{nt}  |  "
                f"t={t_phys:.3f}  |  L2 err={l2_err:.2e}",
                fontsize=12,
            )
            writer.grab_frame()

            print(f"  step {s:05d} | t={t_phys:.3f} | "
                  f"L2={l2_err:.2e} | divB={divB_l2:.2e} | divE={divE_l2:.2e}")

    plt.close(fig)
    print(f"\n>> GIF saved  →  {gif_path}  ({time.perf_counter()-t_wall:.1f}s total)")

    # ── Constraint plot ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.semilogy(steps_log, divB_log, label=r"$\|\nabla\!\cdot\!B\|_2$", lw=2)
    ax2.semilogy(steps_log, divE_log, label=r"$\|\nabla\!\cdot\!E\|_2$", lw=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("L2 norm")
    ax2.set_title(f"Constraint evolution  |  scheme={scheme}  |  {nx}×{ny} grid")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    con_path = os.path.join(out_dir, f"constraints_{scheme}.png")
    fig2.savefig(con_path, dpi=150)
    plt.close(fig2)
    print(f">> Constraints →  {con_path}")


if __name__ == "__main__":
    main()
