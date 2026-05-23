#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DUMP_NAME=${AGENT_DUMP_DIR:-flaggems-pytest}
DUMP_DIR="$ROOT/debug_agent/$DUMP_NAME"

mkdir -p "$DUMP_DIR"

export PATH="$ROOT/build/bin:$PATH"
export PYTHONPATH="$ROOT/third_party/triton/python:$ROOT/third_party/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_BACKENDS_IN_TREE=1
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export TRITON_XYZ_FIRST_CONFIG_ONLY="${TRITON_XYZ_FIRST_CONFIG_ONLY:-1}"
export TRITON_XYZ_PYTEST_TIMEOUT="${TRITON_XYZ_PYTEST_TIMEOUT:-120}"
export TRITON_HOME="${TRITON_HOME:-$DUMP_DIR/triton_home}"
export GEMS_VENDOR="${GEMS_VENDOR:-xyz}"

mkdir -p "$TRITON_HOME"

args=(
  -v
  --mode quick
  --ref cpu
  tests/test_unary_pointwise_ops.py
  tests/test_binary_pointwise_ops.py
  tests/test_tensor_constructor_ops.py
  tests/test_distribution_ops.py
  tests/ks_tests.py
  tests/test_general_reduction_ops.py
  tests/test_reduction_ops.py
  tests/test_norm_ops.py
  tests/test_blas_ops.py
)

if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("xdist")
PY
then
  args=(-n "${TRITON_XYZ_PYTEST_WORKERS:-4}" "${args[@]}")
fi

pushd "$ROOT/third_party/FlagGems" >/dev/null
pytest "${args[@]}" 2>&1 | tee "$DUMP_DIR/pytest.log"
popd >/dev/null
