#!/usr/bin/env bash
set -euo pipefail

DUMP_NAME=${AGENT_DUMP_DIR:-flaggems-pytest}
DUMP_DIR="$PWD/debug_agent/$DUMP_NAME"
mkdir -p "$DUMP_DIR"

export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export TRITON_XYZ_FIRST_CONFIG_ONLY="${TRITON_XYZ_FIRST_CONFIG_ONLY:-1}"
export TRITON_XYZ_PYTEST_TIMEOUT="${TRITON_XYZ_PYTEST_TIMEOUT:-120}"
export TRITON_HOME="${TRITON_HOME:-$DUMP_DIR/triton_home}"
export GEMS_VENDOR="${GEMS_VENDOR:-xyz}"

mkdir -p "$TRITON_HOME"

args=(
  -v
  #
  --mode quick
  --ref cpu
  #
  -n "${TRITON_XYZ_PYTEST_WORKERS:-4}"
  #
  tests
)

pushd third_party/FlagGems
pytest "${args[@]}" 2>&1 | tee "$DUMP_DIR/pytest.log"
popd
