#!/usr/bin/env bash
set -euo pipefail

DUMP_DIR="${DUMP_DIR:-$PWD/debug_agent/python_test}"
export TRITON_HOME="${TRITON_HOME:-$DUMP_DIR}"
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export MLIR_ENABLE_DUMP="${MLIR_ENABLE_DUMP:-1}"
export TRITON_KERNEL_DUMP="${TRITON_KERNEL_DUMP:-1}"
export TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-$DUMP_DIR/triton_dump}"
export MLIR_ENABLE_DUMP_DIR="${MLIR_ENABLE_DUMP_DIR:-$DUMP_DIR/triton_xyz_mlir_dump}"
mkdir -p "$DUMP_DIR" "$TRITON_HOME" "$TRITON_DUMP_DIR" "$MLIR_ENABLE_DUMP_DIR"

if [[ "${1:-}" != "--" ]]; then
  echo "usage:" >&2
  echo "tools/agent/run_python_test.sh -- python python/tests/test_vec_add.py" >&2
  echo "tools/agent/run_python_test.sh -- pytest python/tests/test_vec_add.py" >&2
  exit 2
fi
shift

if [[ "$#" -eq 0 ]]; then
  echo "missing command after --" >&2
  exit 2
fi

cmd=("$@")

if [[ "${cmd[0]}" == pytest* && "${cmd[0]}" != "pytest" ]]; then
  first="${cmd[0]#pytest}"
  cmd=(pytest "$first" "${cmd[@]:1}")
fi

LOG_PATH="$DUMP_DIR/python_test.log"

echo "[run-python-test] DUMP_DIR=$DUMP_DIR"
echo "[run-python-test] LOG=$LOG_PATH"
echo "[run-python-test] TRITON_DUMP_DIR=$TRITON_DUMP_DIR"
echo "[run-python-test] MLIR_ENABLE_DUMP_DIR=$MLIR_ENABLE_DUMP_DIR"
printf '[run-python-test] COMMAND='
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$LOG_PATH"
