#!/usr/bin/env bash

set -euo pipefail

# Template usage:
#   AGENT_DUMP_DIR=vec_add_case KERNEL_PY=python/tests/test_vec_add.py \
#     bash tools/agent/run_kernel_case_template.sh
#
# Optional arguments are forwarded to the Python script:
#   AGENT_DUMP_DIR=ptr_case KERNEL_PY=python/tests/test_ptr.py \
#     bash tools/agent/run_kernel_case_template.sh -k test_pointer_select

AGENT_DUMP_DIR="${AGENT_DUMP_DIR:-run_kernel_case}"
KERNEL_PY="${KERNEL_PY:-python/tests/test_vec_add.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_test_env.sh"

echo "[run-kernel-template] AGENT_DUMP_DIR=$AGENT_DUMP_DIR"
echo "[run-kernel-template] KERNEL_PY=$KERNEL_PY"
echo "[run-kernel-template] LOG=$AGENT_DUMP_ROOT/compile.log"
echo "[run-kernel-template] TRITON_DUMP_DIR=$TRITON_DUMP_DIR"
echo "[run-kernel-template] MLIR_ENABLE_DUMP_DIR=$MLIR_ENABLE_DUMP_DIR"

"$PYTHON_BIN" "$KERNEL_PY" "$@" 2>&1 | tee "$AGENT_DUMP_ROOT/compile.log"
