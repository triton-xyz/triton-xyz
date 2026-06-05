#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source tools/agent/python_test_env.sh instead of executing it" >&2
  exit 1
fi

AGENT_DUMP_DIR="${AGENT_DUMP_DIR:-run_kernel_case}"

_agent_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$AGENT_DUMP_DIR" = /* ]]; then
  AGENT_DUMP_ROOT="$AGENT_DUMP_DIR"
else
  AGENT_DUMP_ROOT="$_agent_repo_root/debug_agent/$AGENT_DUMP_DIR"
fi

mkdir -p "$AGENT_DUMP_ROOT"

export TRITON_HOME="$AGENT_DUMP_ROOT"
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export MLIR_ENABLE_DUMP="${MLIR_ENABLE_DUMP:-1}"
export TRITON_KERNEL_DUMP="${TRITON_KERNEL_DUMP:-1}"
export TRITON_DUMP_DIR="$AGENT_DUMP_ROOT/triton_dump"
export MLIR_ENABLE_DUMP_DIR="$AGENT_DUMP_ROOT/triton_xyz_mlir_dump"
export TRITON_XYZ_USE_TTA="${TRITON_XYZ_USE_TTA:-1}"

mkdir -p "$TRITON_HOME" "$TRITON_DUMP_DIR" "$MLIR_ENABLE_DUMP_DIR"
