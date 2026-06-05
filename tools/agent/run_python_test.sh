#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DUMP_NAME="${AGENT_DUMP_DIR:-python_test}"
if [[ -n "${AGENT_DUMP_ROOT:-}" ]]; then
  DUMP_DIR="$AGENT_DUMP_ROOT"
elif [[ "$DUMP_NAME" = /* ]]; then
  DUMP_DIR="$DUMP_NAME"
else
  DUMP_DIR="$REPO_ROOT/debug_agent/$DUMP_NAME"
fi

mkdir -p "$DUMP_DIR"

export TRITON_HOME="${TRITON_HOME:-$DUMP_DIR}"
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export MLIR_ENABLE_DUMP="${MLIR_ENABLE_DUMP:-1}"
export TRITON_KERNEL_DUMP="${TRITON_KERNEL_DUMP:-1}"
export TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-$DUMP_DIR/triton_dump}"
export MLIR_ENABLE_DUMP_DIR="${MLIR_ENABLE_DUMP_DIR:-$DUMP_DIR/triton_xyz_mlir_dump}"
export TRITON_XYZ_USE_TTA="${TRITON_XYZ_USE_TTA:-1}"

mkdir -p "$TRITON_HOME" "$TRITON_DUMP_DIR" "$MLIR_ENABLE_DUMP_DIR"

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
