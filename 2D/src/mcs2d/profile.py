"""
GPU profiling helper for the MCS 2D solver.

Captures a JAX/XLA execution trace for a chosen scheme and grid size, saves
it to disk, and prints clear instructions for viewing in Perfetto.

What the trace shows you
------------------------
* Per-kernel GPU time — find the actual hotspots (GEMM? CRT? bias correction?)
* SM occupancy — are tiles running concurrently or serialised?
* Memory transfers — are CRT residues round-tripping to VRAM, or staying in
  shared memory / L2?
* Kernel launch overhead — is the GPU idle between kernels?

Usage (from repo root)
----------------------
    python mcs2d/profile.py                    # default: fused_ozaki @ 512
    python mcs2d/profile.py fused_ozaki 256
    python mcs2d/profile.py floating_point 512

It runs N_WARMUP steps (un-profiled, to trigger JIT compile and warm caches),
then N_PROFILE_STEPS steps inside a profiler-traced block.

Viewing the trace
-----------------
The script prints the exact trace-file path and a one-line `xdg-open` /
`open` command.  Open it at https://ui.perfetto.dev — drag the .json.gz file
into the page (or click "Open trace file").
"""

import os
import sys
import glob
import time
import shutil
import subprocess
from pathlib import Path


def _print_box(title, lines):
    """Pretty-print a boxed section."""
    width = max(len(title), *(len(l) for l in lines)) + 4
    print()
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│ {title:<{width - 4}} │")
    print("├" + "─" * (width - 2) + "┤")
    for l in lines:
        print(f"│ {l:<{width - 4}} │")
    print("└" + "─" * (width - 2) + "┘")


def _nvidia_smi_snapshot(tag):
    """Print a short nvidia-smi snapshot (GPU util / mem) if available."""
    if shutil.which("nvidia-smi") is None:
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        print(f"  [{tag}] nvidia-smi: {out}")
    except Exception:
        pass


def main():
    # ── CLI ────────────────────────────────────────────────────────────────
    scheme = sys.argv[1] if len(sys.argv) > 1 else "fused_ozaki"
    nx     = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    repo_root = Path(__file__).resolve().parent.parent
    mcs2d_dir = repo_root / "mcs2d"
    # Write under the repo (persistent home dir on HPC) rather than /tmp
    # which is per-node scratch and gets cleared unpredictably.
    trace_dir = repo_root / "traces"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(mcs2d_dir))
    sys.path.insert(0, str(repo_root))

    # JAX setup with persistent cache so we skip the slow compile if it ran before.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from mcs_common.jax_config import setup as jax_setup
    jax_setup(verbose=True)

    import jax
    from mcs2d.main import MaxwellChernSimons2D, InitialData, load_parameters

    N_WARMUP        = 3
    N_PROFILE_STEPS = 50    # number of full RK4 steps captured in the trace

    _print_box("MCS 2D Profile", [
        f"Scheme  : {scheme}",
        f"Grid    : {nx}×{nx}",
        f"Backend : {jax.default_backend()}",
        f"Device  : {jax.devices()[0]}",
        f"Warmup  : {N_WARMUP} steps (untraced)",
        f"Profile : {N_PROFILE_STEPS} steps (captured)",
        f"Trace   : {trace_dir}",
    ])

    # ── Build the sim and a scan-based stepping function ───────────────────
    params = load_parameters(str(mcs2d_dir / "params.toml"))
    params.update({
        "scheme":          scheme,
        "Nx": nx, "Ny": nx,
        "id_type":         "birefringent",
        "bc_type":         "periodic",
        "sponge_strength": 0.0,
    })
    dx = (params["xmax"] - params["xmin"]) / nx
    sim   = MaxwellChernSimons2D(dx, dx, params["Lambda"], params)
    state = InitialData(sim, params).generate()

    @jax.jit
    def warmup_step(s):
        return sim.step_rk4(s, sim.dt)

    @jax.jit
    def profile_run(s):
        return jax.lax.scan(lambda c, _: (sim.step_rk4(c, sim.dt), None),
                            s, None, length=N_PROFILE_STEPS)[0]

    # ── Warmup (compile + cache) ───────────────────────────────────────────
    print("\n>> Warming up (compile + cache priming)...")
    _nvidia_smi_snapshot("pre")
    t0 = time.perf_counter()
    state = warmup_step(state); state.data.block_until_ready()
    state = profile_run(state); state.data.block_until_ready()   # compile profile_run
    for _ in range(N_WARMUP):
        state = warmup_step(state)
    state.data.block_until_ready()
    print(f"   warmup wall: {time.perf_counter() - t0:.1f}s")
    _nvidia_smi_snapshot("warm")

    # ── Captured profile ───────────────────────────────────────────────────
    print(f"\n>> Profiling {N_PROFILE_STEPS} steps...")
    t0 = time.perf_counter()
    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        state = profile_run(state)
        state.data.block_until_ready()
    wall = time.perf_counter() - t0
    print(f"   profile wall: {wall:.2f}s  ({wall / N_PROFILE_STEPS * 1e3:.2f} ms/step)")
    _nvidia_smi_snapshot("post")

    # ── Find the generated trace files ─────────────────────────────────────
    candidates  = sorted(glob.glob(str(trace_dir / "**" / "*.trace.json.gz"),
                                   recursive=True))
    candidates += sorted(glob.glob(str(trace_dir / "**" / "*.pb"),
                                   recursive=True))

    _print_box("Trace saved", [
        f"Directory : {trace_dir}",
        f"Files     : {len(candidates)} found",
        *[f"  • {Path(c).relative_to(trace_dir)}  ({os.path.getsize(c)/1024:.0f} KB)"
          for c in candidates[:5]],
    ])

    if not candidates:
        print("\n!! No trace files generated.  Profiler may have silently failed.")
        sys.exit(1)

    best = candidates[0]
    _print_box("How to view", [
        "1. Copy the trace file to your local machine if running remotely:",
        f"   scp <user>@<host>:{best} ./",
        "",
        "2. Open Perfetto in your browser:",
        "   https://ui.perfetto.dev",
        "",
        "3. Either drag the .json.gz file into the page,",
        "   or click 'Open trace file' (top-left) and select it.",
        "",
        "4. Look for:",
        "   • 'GPU 0' track — per-kernel time, occupancy gaps",
        "   • 'HostToDeviceCopy' / 'DeviceToHostCopy' — should be near-zero",
        "   • Kernel names with 'gemm' (GEMM), 'reduce' (BFP scale),",
        "     'fusion' (XLA-fused ops — good!), 'copy' (bad if frequent)",
    ])

    print()


if __name__ == "__main__":
    main()
