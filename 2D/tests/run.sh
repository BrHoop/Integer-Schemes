#!/usr/bin/env bash
# Test orchestrator for the 2D MCS suite.
#
# Usage:
#   ./tests/run.sh                       # everything
#   ./tests/run.sh unit                  # only unit tests
#   ./tests/run.sh integration           # only integration tests
#   ./tests/run.sh regression            # only regression / no-recompile tests
#   ./tests/run.sh validation            # Step 1.1 convergence/spectrum/BC/constraint guards
#   ./tests/run.sh amr                   # AMR tests across all dirs
#   ./tests/run.sh fast                  # exclude @pytest.mark.slow
#
# Any extra args are forwarded to pytest (e.g. `./tests/run.sh unit -k topology`).

set -euo pipefail
cd "$(dirname "$0")/.."   # → 2D/

CATEGORY="${1:-all}"; shift || true

case "$CATEGORY" in
  all)         TARGET=(tests/) ;;
  unit)        TARGET=(tests/unit/) ;;
  integration) TARGET=(tests/integration/) ;;
  regression)  TARGET=(tests/regression/) ;;
  validation)  TARGET=(tests/validation/) ;;
  amr)         TARGET=(tests/ -k amr) ;;
  fast)        TARGET=(tests/ -m "not slow") ;;
  *)           echo "unknown category: $CATEGORY"; exit 2 ;;
esac

python -m pytest "${TARGET[@]}" -v "$@"
