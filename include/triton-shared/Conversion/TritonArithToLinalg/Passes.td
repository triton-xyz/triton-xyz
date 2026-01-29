#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonArithToLinalg : Pass<"triton-arith-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Triton arithmetic operations into linalg";
  let options = [
      Option<"ttToFuncFunc", "tt-to-func-func", "bool", "true",
             "Convert tt.func to func.func">,
      Option<"assertToCf", "assert-to-cf", "bool", "true",
             "Convert tt.assert to cf.assert">,
      Option<"transposeReduceToRank0", "transpose-reduce-to-rank0", "bool", "true",
             "Transpose reductions to rank 0 which collapses remaining dimensions. Otherwise "
             "transpose any rank - 1 reduction to rank - 2 as the old behavior. These transposes "
             "are planned for removal as Triton Shared should not be responsible for this.">,
  ];
}

def TritonPidsToFuncArgs : Pass<"triton-pids-to-func-args", "mlir::ModuleOp"> {
  let summary = "Convert tt.get_program_id and tt.get_num_programs to function arguments";
}

#endif
