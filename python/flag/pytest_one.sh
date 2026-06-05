#!/usr/bin/env bash
set -euo pipefail

DUMP_NAME=${AGENT_DUMP_DIR:-flaggems-pytest-one}
DUMP_DIR="$PWD/debug_agent/$DUMP_NAME"
mkdir -p "$DUMP_DIR"

export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export TRITON_XYZ_FIRST_CONFIG_ONLY="${TRITON_XYZ_FIRST_CONFIG_ONLY:-1}"
export TRITON_XYZ_PYTEST_TIMEOUT="${TRITON_XYZ_PYTEST_TIMEOUT:-120}"
export TRITON_HOME="${TRITON_HOME:-$DUMP_DIR/triton_home}"
export GEMS_VENDOR="${GEMS_VENDOR:-xyz}"

export MLIR_ENABLE_DUMP_DIR="$DUMP_DIR/mlir_dump"

mkdir -p "$TRITON_HOME"

if [ "$#" -eq 0 ]; then
  args=(
    -v
    --mode quick
    --ref cpu
    tests/test_abs.py
    # tests/test_unary_pointwise_ops.py
    # tests/test_tensor_constructor_ops.py
    #
    # tests/test_unary_pointwise_ops.py::test_accuracy_abs
    # tests/test_tensor_constructor_ops.py::test_accuracy_rand
  )
else
  args=(
    -v
    --mode quick
    --ref cpu
    "$@"
  )
fi

pushd third_party/FlagGems
pytest "${args[@]}" 2>&1 | tee "$DUMP_DIR/pytest_one.log"
popd
