#!/usr/bin/env bash
# Sync helper for Integer-Schemes
#
#   ./sync.sh push         -- copy local code to supercomputer
#   ./sync.sh pull         -- copy sim output + logs + validation/profiling results
#   ./sync.sh pull logs    -- copy only test logs from supercomputer
#   ./sync.sh pull data    -- copy only simulation output (h5/xdmf) from supercomputer
#   ./sync.sh pull results -- copy validation + GPU-profiling deliverables (png/csv/json/txt)
#   ./sync.sh pull traces  -- copy GPU profiler traces
#
# Tip: always `pull results` BEFORE the next `push`. push uses --delete; the
# results folder is shielded from deletion via .rsyncignore, but pulling first
# keeps your local copy authoritative.

set -euo pipefail

REMOTE="bmh74@ssh.rc.byu.edu"
REMOTE_PATH="/home/bmh74/Integer-Schemes"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Where the Step 1.1 validation + profiling deliverables land (next to the spec).
RESULTS_REL="docs/phases/phase_0_2d_foundation/step_0.1_results"

# ── SSH connection sharing (ControlMaster) ────────────────────────────────────
# Every rsync below would otherwise open its own SSH connection and trigger Duo
# 2FA again.  We open ONE shared master connection up front (a single 2FA), route
# all rsync traffic over it via ControlPath, and tear it down on exit.  Result:
# exactly one 2FA prompt per `push`/`pull`, no matter how many rsync calls run.
_SSH_CTL="${TMPDIR:-/tmp}/ischemes-ssh-%r@%h:%p"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${_SSH_CTL} -o ControlPersist=300s"
RSYNC_E=(-e "ssh ${SSH_OPTS}")

ssh_master_open() {
    echo ">> Opening shared SSH connection (one 2FA for the whole run) ..."
    ssh ${SSH_OPTS} -fN "${REMOTE}"          # authenticate once; backgrounds the master
}
ssh_master_close() {
    ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true
}
trap ssh_master_close EXIT

push() {
    ssh_master_open
    echo ">> Pushing code to ${REMOTE}:${REMOTE_PATH} ..."
    rsync -avz --progress --delete "${RSYNC_E[@]}" \
        --exclude-from="${LOCAL_DIR}/.rsyncignore" \
        "${LOCAL_DIR}/" \
        "${REMOTE}:${REMOTE_PATH}/"
    echo ">> Push complete."
}

pull_data() {
    echo "   2D/output ..."
    mkdir -p "${LOCAL_DIR}/2D/output"
    rsync -avz --progress "${RSYNC_E[@]}" \
        --include="*.h5" --include="*.xdmf" --exclude="*" \
        "${REMOTE}:${REMOTE_PATH}/2D/output/" \
        "${LOCAL_DIR}/2D/output/"

    echo "   3D/output ..."
    mkdir -p "${LOCAL_DIR}/3D/output"
    rsync -avz --progress "${RSYNC_E[@]}" \
        --include="*.h5" --include="*.xdmf" --exclude="*" \
        "${REMOTE}:${REMOTE_PATH}/3D/output/" \
        "${LOCAL_DIR}/3D/output/"
}

pull_logs() {
    echo "   test logs ..."
    rsync -avz --progress "${RSYNC_E[@]}" \
        --include="*.log" --exclude="*" \
        "${REMOTE}:${REMOTE_PATH}/2D/tests/" \
        "${LOCAL_DIR}/2D/tests/"
}

pull_traces() {
    echo "   profiler traces ..."
    mkdir -p "${LOCAL_DIR}/traces"
    rsync -avz --progress "${RSYNC_E[@]}" \
        "${REMOTE}:${REMOTE_PATH}/traces/" \
        "${LOCAL_DIR}/traces/"
}

pull_results() {
    echo "   validation + profiling results (${RESULTS_REL}) ..."
    mkdir -p "${LOCAL_DIR}/${RESULTS_REL}"
    rsync -avz --progress "${RSYNC_E[@]}" \
        --include="*.png" --include="*.csv" --include="*.json" --include="*.txt" \
        --exclude="*" \
        "${REMOTE}:${REMOTE_PATH}/${RESULTS_REL}/" \
        "${LOCAL_DIR}/${RESULTS_REL}/"

    # benchmark.py written standalone (no out_dir arg) lands here instead.
    echo "   benchmark output dir (2D/src/mcs2d/output) ..."
    mkdir -p "${LOCAL_DIR}/2D/src/mcs2d/output"
    rsync -avz --progress "${RSYNC_E[@]}" \
        --include="*.png" --include="*.csv" --include="*.json" --include="*.txt" \
        --exclude="*" \
        "${REMOTE}:${REMOTE_PATH}/2D/src/mcs2d/output/" \
        "${LOCAL_DIR}/2D/src/mcs2d/output/"

    # 3D benchmark (Phase 1, Step 1.4) — standalone run lands here.
    echo "   3D benchmark output dir (3D/src/mcs3d/output) ..."
    mkdir -p "${LOCAL_DIR}/3D/src/mcs3d/output"
    rsync -avz --progress "${RSYNC_E[@]}" \
        --include="*.png" --include="*.csv" --include="*.json" --include="*.txt" \
        --exclude="*" \
        "${REMOTE}:${REMOTE_PATH}/3D/src/mcs3d/output/" \
        "${LOCAL_DIR}/3D/src/mcs3d/output/"
}

pull() {
    local what="${1:-all}"
    ssh_master_open
    echo ">> Pulling ${what} from ${REMOTE} ..."
    case "$what" in
        logs)    pull_logs ;;
        data)    pull_data ;;
        results) pull_results ;;
        traces)  pull_traces ;;
        all)     pull_data; pull_logs; pull_results ;;
        *)
            echo "Error: unknown pull target '${what}' (use: logs, data, results, traces, or omit for all)"
            exit 1
            ;;
    esac
    echo ">> Pull complete."
}

case "${1:-}" in
    push) push ;;
    pull) pull "${2:-all}" ;;
    *)
        echo "Usage: $0 {push|pull [logs|data|results|traces]}"
        echo "  push            -- sync local code to supercomputer"
        echo "  pull            -- sync sim output + logs + validation/profiling results"
        echo "  pull logs       -- sync only test logs from supercomputer"
        echo "  pull data       -- sync only simulation output (h5/xdmf) from supercomputer"
        echo "  pull results    -- sync validation + GPU-profiling deliverables (png/csv/json/txt)"
        echo "  pull traces     -- sync GPU profiler traces"
        exit 1
        ;;
esac
