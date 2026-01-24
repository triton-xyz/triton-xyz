MLIR=test/Conversion/TritonToPtr/cast_with_int_ptr.mlir
ARGS='--split-input-file --triton-arith-to-linalg --triton-tensor-ptr-to-linalg --triton-to-ptr'

triton-shared-opt $ARGS $MLIR | utils/generate-test-checks.py -i --source_delim_regex "tt.func|func.func" --strict_name_re 1 --source $MLIR
