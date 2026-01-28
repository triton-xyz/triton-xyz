# for tests only one check
MLIR=test/Conversion/triton-to-ptr.mlir
# get `ARGS` from `MLIR`
ARGS=(--split-input-file --triton-tensor-ptr-to-linalg --triton-to-ptr)
# defalut `CHECK`
PREFIX="CHECK"
SOURCE_DELIM_REGEX='^(?!\s*//)\s*(tt\.func|func\.func)\b'
if triton-xyz-opt "${ARGS[@]}" $MLIR | utils/generate-test-checks.py --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR >/dev/null; then
  triton-xyz-opt "${ARGS[@]}" $MLIR | utils/generate-test-checks.py -i --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR
else
  echo "error in $(utils/generate-test-checks.py), needs recheck"
fi

# for tests only more than one checks
MLIR=test/Conversion/triton-to-structured-prepass.mlir
# get `ARGS` from `MLIR`
ARGS=(--split-input-file --triton-to-structured --remove-dead-values --canonicalize)
# defalut `CHECK`
PREFIX="CHECK"
SOURCE_DELIM_REGEX='^(?!\s*//)\s*(tt\.func|func\.func)\b'
if triton-xyz-opt "${ARGS[@]}" $MLIR | utils/generate-test-checks.py --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR >/dev/null; then
  triton-xyz-opt "${ARGS[@]}" $MLIR | utils/generate-test-checks.py -i --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR
else
  echo "error in $(utils/generate-test-checks.py), needs recheck"
fi
ARGS=(--split-input-file --triton-to-structured="run-prepass-only=true")
# another check `PREPASS`
PREFIX="PREPASS"
SOURCE_DELIM_REGEX='^(?!\s*//)\s*(tt\.func|func\.func)\b'
if triton-xyz-opt "${ARGS[@]}" $MLIR | utils/generate-test-checks.py --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR >/dev/null; then
  triton-xyz-opt "${ARGS[@]}" $MLIR | utils/generate-test-checks.py -i --source_delim_regex $SOURCE_DELIM_REGEX --strict_name_re 1 --check-prefix $PREFIX --source $MLIR
else
  echo "error in $(utils/generate-test-checks.py), needs recheck"
fi
