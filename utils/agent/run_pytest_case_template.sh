#!/usr/bin/env bash

set -euo pipefail

# Template usage:
#   AGENT_DUMP_DIR=ptr_select_case \
#   PYTEST_TARGET=python/tests/test_ptr.py::test_pointer_select \
#     bash utils/agent/run_pytest_case_template.sh -q
#
# Additional arguments are forwarded to pytest:
#   AGENT_DUMP_DIR=structured_mask \
#   PYTEST_TARGET='python/tests/test_triton_to_structured.py::test_masked_1d[3]' \
#     bash utils/agent/run_pytest_case_template.sh -q

PYTEST_TARGET="${PYTEST_TARGET:-python/tests/test_triton_to_structured.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_test_env.sh"

LOG_PATH="$AGENT_DUMP_ROOT/pytest.log"

echo "[run-pytest-template] AGENT_DUMP_DIR=$AGENT_DUMP_DIR"
echo "[run-pytest-template] PYTEST_TARGET=$PYTEST_TARGET"
echo "[run-pytest-template] LOG=$LOG_PATH"
echo "[run-pytest-template] TRITON_DUMP_DIR=$TRITON_DUMP_DIR"
echo "[run-pytest-template] MLIR_ENABLE_DUMP_DIR=$MLIR_ENABLE_DUMP_DIR"

"$PYTHON_BIN" -m pytest "$PYTEST_TARGET" "$@" 2>&1 | tee "$LOG_PATH"
