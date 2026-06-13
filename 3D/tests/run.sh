#!/usr/bin/env bash
# Test orchestrator for the 3D MCS suite.  Same interface as 2D/tests/run.sh.

set -euo pipefail
cd "$(dirname "$0")/.."   # → 3D/

CATEGORY="${1:-all}"; shift || true

case "$CATEGORY" in
  all)         TARGET=(tests/) ;;
  unit)        TARGET=(tests/unit/) ;;
  integration) TARGET=(tests/integration/) ;;
  regression)  TARGET=(tests/regression/) ;;
  validation)  TARGET=(tests/validation/) ;;
  fast)        TARGET=(tests/ -m "not slow") ;;
  *)           echo "unknown category: $CATEGORY"; exit 2 ;;
esac

python -m pytest "${TARGET[@]}" -v "$@"
