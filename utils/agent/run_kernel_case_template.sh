#!/usr/bin/env bash

set -euo pipefail

# Template usage:
#   AGENT_DUMP_DIR=vec_add_case KERNEL_PY=python/tests/test_vec_add.py \
#     bash utils/agent/run_kernel_case_template.sh
#
# Optional arguments are forwarded to the Python script:
#   AGENT_DUMP_DIR=ptr_case KERNEL_PY=python/tests/test_ptr.py \
#     bash utils/agent/run_kernel_case_template.sh -k test_pointer_select

AGENT_DUMP_DIR="${AGENT_DUMP_DIR:-run_kernel_case}"
KERNEL_PY="${KERNEL_PY:-python/tests/test_vec_add.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DIR="debug_agent/${AGENT_DUMP_DIR}"
mkdir -p "$DIR"

export TRITON_HOME="$DIR"
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export MLIR_ENABLE_DUMP="${MLIR_ENABLE_DUMP:-1}"
export MLIR_ENABLE_DUMP_DIR="$DIR/triton_xyz_mlir_dump"

# use TTA pipeline by default; set 0 to disable.
export TRITON_XYZ_USE_TTA="${TRITON_XYZ_USE_TTA:-1}"

echo "[run-kernel-template] AGENT_DUMP_DIR=$AGENT_DUMP_DIR"
echo "[run-kernel-template] KERNEL_PY=$KERNEL_PY"
echo "[run-kernel-template] LOG=$DIR/compile.log"
echo "[run-kernel-template] IR_DUMP_DIR=$MLIR_ENABLE_DUMP_DIR"

"$PYTHON_BIN" "$KERNEL_PY" "$@" 2>&1 | tee "$DIR/compile.log"
