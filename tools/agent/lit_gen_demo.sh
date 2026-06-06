#!/usr/bin/env bash
set -euo pipefail

MLIR=test/Conversion/triton-to-xyz-lowering.mlir
# get `ARGS` from `MLIR`
ARGS=(--split-input-file --triton-to-xyz)
# defalut `CHECK`
PREFIX="CHECK"
SOURCE_DELIM_REGEX='^(?!\s*//)\s*(tt\.func|func\.func|llvm\.func)\b'
if triton-xyz-opt "${ARGS[@]}" $MLIR | tools/generate-test-checks.py --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR >/dev/null; then
  triton-xyz-opt "${ARGS[@]}" $MLIR | tools/generate-test-checks.py -i --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR
else
  echo "error in $(tools/generate-test-checks.py), needs recheck"
fi
