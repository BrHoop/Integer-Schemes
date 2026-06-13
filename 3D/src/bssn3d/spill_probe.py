"""Phase-2.2 register-spill probe for the BSSN RHS (GPU-only; motivates Phase 3).

Turnkey: just run it on a GPU node, no XLA_FLAGS needed —

    python -m bssn3d.spill_probe

It (1) compiles the jitted BSSN RHS while dumping XLA's PTX to a temp dir, then
(2) runs ``ptxas -v`` on each dumped kernel for the device's arch and reports
per-kernel **registers** + **spill stores/loads**. The Phase-3 question: does XLA
spill (registers pegged at 255 + nonzero spill bytes) or split into many
HBM-backed kernels? Record the worst register count and total spill.

We dump+assemble PTX ourselves because the older `--xla_gpu_asm_extra_flags=-v`
route is rejected ("Unknown flag in XLA_FLAGS") on this XLA build. ptxas must be on
PATH (CUDA toolkit). Override arch with BSSN_PTXAS_ARCH (e.g. sm_90a).
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# Quiet the harmless SLURM/NUMA hwloc_set_cpubind errors (don't clobber if set).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# XLA flags must be set BEFORE jax imports. Dump PTX into a PERSISTENT dir (NOT
# /tmp — it is unreliable on this cluster and the dumped files vanish). Defaults
# to ./bssn_xla_dump in the cwd; override with BSSN_XLA_DUMP.
_DUMP_DIR = os.path.abspath(os.environ.get("BSSN_XLA_DUMP") or
                            str(Path.cwd() / "bssn_xla_dump"))
shutil.rmtree(_DUMP_DIR, ignore_errors=True)
os.makedirs(_DUMP_DIR, exist_ok=True)
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           f" --xla_dump_to={_DUMP_DIR}").strip()

import jax
jax.config.update("jax_enable_x64", True)
# Force a fresh compile every run (else a persistent-cache hit means XLA never
# compiles → nothing to dump). Empty dir disables the on-disk cache.
jax.config.update("jax_compilation_cache_dir", "")
import jax.numpy as jnp

from .grid import Grid
from .state import PhysicsParams
from .rhs import BSSNSolver
from . import initial_data as bid

_INFO_RE = re.compile(
    r"Function properties for (?P<fn>\S+)|"
    r"Used (?P<regs>\d+) registers|"
    r"(?P<sstores>\d+) bytes spill stores|"
    r"(?P<sloads>\d+) bytes spill loads"
)


def _ptxas_bin():
    """Resolve ptxas, preferring the CUDA-wheel copy that matches the installed
    jaxlib (so the PTX ISA version is supported); fall back to PATH."""
    import glob
    import sysconfig
    cands = []
    for key in ("purelib", "platlib"):
        base = sysconfig.get_paths().get(key)
        if base:
            cands += sorted(glob.glob(os.path.join(base, "nvidia", "*", "bin", "ptxas")))
    cands.append(shutil.which("ptxas"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def _arch_candidates():
    if os.environ.get("BSSN_PTXAS_ARCH"):
        return [os.environ["BSSN_PTXAS_ARCH"]]
    try:
        cc = jax.devices("gpu")[0].compute_capability  # e.g. "9.0"
        major, minor = cc.split(".")
        base = f"sm_{major}{minor}"
        return [base + "a", base] if (major, minor) == ("9", "0") else [base]
    except Exception:
        return ["sm_90a", "sm_90"]


def _ptxas_report(ptx_files, archs, ptxas):
    """Run ptxas -v over the PTX; return list of (file, fn, regs, spill_bytes)."""
    rows = []
    for ptx in ptx_files:
        out = None
        for arch in archs:
            proc = subprocess.run(
                [ptxas, "-v", f"-arch={arch}", ptx, "-o", os.devnull],
                capture_output=True, text=True)
            out = proc.stderr
            if proc.returncode == 0 or "registers" in out:
                break
        if not out:
            continue
        cur_fn, regs, spill = None, 0, 0
        for line in out.splitlines():
            m = re.search(r"Compiling entry function '([^']+)'", line)
            if m:
                if cur_fn is not None:
                    rows.append((os.path.basename(ptx), cur_fn, regs, spill))
                cur_fn, regs, spill = m.group(1), 0, 0
            mr = re.search(r"Used (\d+) registers", line)
            if mr:
                regs = int(mr.group(1))
            for ms in re.finditer(r"(\d+) bytes spill (?:stores|loads)", line):
                spill += int(ms.group(1))
        if cur_fn is not None:
            rows.append((os.path.basename(ptx), cur_fn, regs, spill))
    return rows


def main(N: int = 48, order: int = 6, scheme: str = None):
    # Pick the RHS algebra variant: "verbatim" (Phase-2 oracle) or "staged" (the
    # Step-3.1 optimization_barrier probe). Override with BSSN_SCHEME=staged.
    scheme = scheme or os.environ.get("BSSN_SCHEME", "verbatim")
    # FD order override (BSSN_ORDER) — the fp64 fused kernel is order-8-locked (ng=4),
    # so it must run at order 8; the algebra-only schemes are order-agnostic.
    order = int(os.environ.get("BSSN_ORDER", str(order)))
    print(f">> BSSN RHS spill probe | backend={jax.default_backend()} "
          f"| N={N}^3 | order={order} | scheme={scheme}")
    print(f">> XLA dump dir: {_DUMP_DIR}")
    print(f">> XLA_FLAGS seen: {os.environ.get('XLA_FLAGS')!r}")

    grid = Grid.from_domain(N, order=order)
    solver = BSSNSolver(grid, PhysicsParams(), order=order, scheme=scheme)
    state = bid.gauge_wave(grid, amplitude=0.01)
    rhs = jax.jit(solver.rhs)

    t0 = time.time()
    out = rhs(state)
    out.data.block_until_ready()
    print(f">> compiled + ran one RHS eval in {time.time() - t0:.2f}s "
          f"| finite={bool(jnp.all(jnp.isfinite(out.data)))}")

    if jax.default_backend() != "gpu":
        print(">> CPU backend: no PTX/ptxas. Run this on a GPU node for the spill "
              "numbers.")
        return

    all_files = sorted(os.listdir(_DUMP_DIR))
    ptx_files = [os.path.join(_DUMP_DIR, f) for f in all_files if f.endswith(".ptx")]
    if not ptx_files:
        # Show what XLA actually produced so we can spot a different extension.
        from collections import Counter
        exts = Counter("".join(Path(f).suffixes) or "(none)" for f in all_files)
        print(f">> No *.ptx in {_DUMP_DIR}. {len(all_files)} files dumped; "
              f"extensions: {dict(exts)}")
        print(">> sample names:")
        for f in all_files[:20]:
            print(f"     {f}")
        print(">> Paste this list back so I can point at the right artifact.")
        return

    archs = _arch_candidates()
    ptxas = _ptxas_bin()
    if ptxas is None:
        print(">> ptxas not found (PATH or CUDA wheels). Load the CUDA toolkit, then:")
        print(f"     for f in {_DUMP_DIR}/*.ptx; do ptxas -v -arch={archs[0]} "
              f'"$f" -o /dev/null; done 2>&1 | grep -E "registers|spill"')
        return
    print(f">> ptxas: {ptxas}")
    print(f">> ptxas arch candidates: {archs} | {len(ptx_files)} PTX module(s)")
    rows = _ptxas_report(ptx_files, archs, ptxas)

    if not rows:
        # ptxas ran but no kernel info parsed — show its raw output so we can see
        # the real error (ISA/arch mismatch, extern refs, unexpected format, ...).
        biggest = max(ptx_files, key=os.path.getsize)
        sizes = sorted((os.path.getsize(p) for p in ptx_files), reverse=True)
        print(f">> 0 kernels parsed. ptx sizes top10 (bytes): {sizes[:10]}")
        print(f">> diagnosing on largest module {os.path.basename(biggest)} "
              f"({os.path.getsize(biggest)} B):")
        for arch in archs:
            proc = subprocess.run([ptxas, "-v", f"-arch={arch}", biggest,
                                   "-o", os.devnull], capture_output=True, text=True)
            print(f"   --- ptxas -arch={arch}  returncode={proc.returncode} ---")
            for line in (proc.stderr or proc.stdout or "(no output)").splitlines()[:30]:
                print(f"     {line}")
        print(">> Paste this diagnostic block back.")
        return

    rows.sort(key=lambda r: (-r[3], -r[2]))   # worst spill, then most registers
    spillers = [r for r in rows if r[3] > 0]
    print(f"\n>> {len(rows)} kernel(s); {len(spillers)} spill. "
          f"Worst by spill, then registers:")
    print(f"   {'regs':>5} {'spill_B':>8}  kernel")
    for fname, fn, regs, spill in rows[:15]:
        print(f"   {regs:5d} {spill:8d}  {fn[:60]}")
    worst_regs = max((r[2] for r in rows), default=0)
    total_spill = sum(r[3] for r in rows)
    print(f"\n>> SUMMARY: max registers = {worst_regs} (budget 255) | "
          f"total spill bytes = {total_spill} | spilling kernels = {len(spillers)}")
    print(">> Paste this SUMMARY line + the table back.")


if __name__ == "__main__":
    main()
