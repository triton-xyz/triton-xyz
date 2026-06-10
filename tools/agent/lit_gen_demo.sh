#!/usr/bin/env bash

# for tests only one check
MLIR=test/Conversion/triton-to-xyz-lowering.mlir
# get `ARGS` from `MLIR`
ARGS=(--split-input-file --triton-to-xyz)
# defalut `CHECK`
PREFIX="CHECK"
SOURCE_DELIM_REGEX='^(?!\s*//)\s*(func\.func|llvm\.func)\b'
if triton-xyz-opt "${ARGS[@]}" "$MLIR" | tools/agent/generate-test-checks.py --source_delim_regex "$SOURCE_DELIM_REGEX" --strict_name_re 1 --check-prefix "$PREFIX" --source "$MLIR" >/dev/null; then
  triton-xyz-opt "${ARGS[@]}" "$MLIR" | tools/agent/generate-test-checks.py -i --source_delim_regex "$SOURCE_DELIM_REGEX" --strict_name_re 1 --check-prefix "$PREFIX" --source "$MLIR"
else
  echo "error in tools/agent/generate-test-checks.py, needs recheck"
fi

# for mlir-translate tests
MLIR=xten-llvm/test/tensilica-call-intrinsic.mlir
# get `ARGS` from `MLIR`
ARGS=(--split-input-file --mlir-to-llvmir)
# defalut `CHECK`
PREFIX="CHECK"
SOURCE_DELIM_REGEX='^module attributes'
if mlir-translate "${ARGS[@]}" "$MLIR" | tools/agent/generate-test-checks.py -i --source_delim_regex "$SOURCE_DELIM_REGEX" --strict_name_re 1 --check-prefix "$PREFIX" --source "$MLIR" >/dev/null; then
  mlir-translate "${ARGS[@]}" "$MLIR" | tools/agent/generate-test-checks.py -i --source_delim_regex "$SOURCE_DELIM_REGEX" --strict_name_re 1 --check-prefix "$PREFIX" --source "$MLIR"
else
  echo "error in tools/agent/generate-test-checks.py, needs recheck"
fi
