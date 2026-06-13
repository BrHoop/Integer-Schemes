#!/usr/bin/env bash
# Serialize + memory-cap heavy local runs so two concurrent Claude sessions can't
# OOM-freeze the box. The local machine is only ~14 GiB RAM / 4 GiB swap, and a
# single BSSN/XLA compile can eat several GB; two at once thrash swap and hard-lock
# the machine. This wrapper makes heavy runs cooperative:
#
#   * flock on a SHARED lock file -> only one wrapped run executes at a time
#     (any second session blocks until the first finishes, instead of piling on).
#   * a memory-capped systemd --user scope -> a runaway gets its *job* killed at
#     the cap, with swap disabled for the job so it can't drag the OS into thrash.
#
# Usage (run BOTH sessions' heavy commands through it):
#   ./run_locked.sh python -m pytest 3D/tests/ -q
#   ./run_locked.sh python -m bssn3d._codegen
#   RUN_MEM_MAX=9G ./run_locked.sh python -m bssn3d.spill_probe   # raise the cap
#
# Env knobs:  RUN_MEM_MAX (default 7G), RUN_THREADS (default 4), RUN_LOCK_WAIT
# (seconds to wait for the lock before giving up; default: wait forever).
set -euo pipefail

LOCK="${RUN_LOCK:-/tmp/integer-schemes-run.lock}"
MEM="${RUN_MEM_MAX:-7G}"
THREADS="${RUN_THREADS:-4}"

# Cap BLAS/OMP threads so the held-back work (and JAX's own pools) don't thrash all
# 8 cores; XLA still uses the GPU normally on Marylou (this only bounds host work).
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

# flock waits for the shared lock (optionally with a timeout), then runs the command
# inside a memory-bounded transient scope. fd 9 holds the lock for the whole run.
exec 9>"$LOCK"
if [ -n "${RUN_LOCK_WAIT:-}" ]; then
  flock -w "$RUN_LOCK_WAIT" 9 || { echo "run_locked: lock busy after ${RUN_LOCK_WAIT}s" >&2; exit 75; }
else
  flock 9
fi

exec systemd-run --user --scope -q \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 \
  -- "$@"
