#!/usr/bin/env bash
set -euo pipefail

DUMP_DIR="${DUMP_DIR:-$PWD/debug_agent/flaggems-pytest-one}"
mkdir -p "$DUMP_DIR"

export PATH="$PWD/build/bin:$PWD/llvm-triton/llvm-project/build/bin:$PATH"

export TRITON_HOME="${TRITON_HOME:-$DUMP_DIR}"
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export MLIR_ENABLE_DUMP="${MLIR_ENABLE_DUMP:-0}"
export TRITON_KERNEL_DUMP="${TRITON_KERNEL_DUMP:-0}"
export TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-$DUMP_DIR/triton_dump}"

export TT_XYZ_ENABLE_DUMP="${TT_XYZ_ENABLE_DUMP:-1}"
export TT_XYZ_ENABLE_DUMP_DIR="${TT_XYZ_ENABLE_DUMP_DIR:-$DUMP_DIR/triton_xyz_mlir_dump}"

export TRITON_XYZ_FIRST_CONFIG_ONLY="${TRITON_XYZ_FIRST_CONFIG_ONLY:-1}"
export TRITON_XYZ_PYTEST_TIMEOUT="${TRITON_XYZ_PYTEST_TIMEOUT:-120}"

export TT_XYZ_ENABLE_DUMP=0

export GEMS_VENDOR="${GEMS_VENDOR:-xyz}"

if [ "$#" -eq 0 ]; then
  args=(
    #
    -v
    #
    --mode quick
    --ref cpu
    #
    tests/test_abs.py
  )
else
  args=(
    #
    -v
    #
    --mode quick
    --ref cpu
    #
    -n 32
    #
    "$@"
  )
fi

pushd third_party/FlagGems
pytest "${args[@]}" 2>&1 | tee "$DUMP_DIR/pytest_one.log"
popd
