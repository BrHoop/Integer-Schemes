#!/usr/bin/env bash
# Workspace-level test runner.  Runs both 2D and 3D suites.
# For finer control use ./2D/tests/run.sh or ./3D/tests/run.sh.

set -euo pipefail
cd "$(dirname "$0")"

python -m pytest 2D/tests/ 3D/tests/ -v "$@"
