#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonArithToLinalg : Pass<"triton-arith-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Triton arithmetic operations into linalg";
  let options = [
      Option<"assertToCf", "assert-to-cf", "bool", "true",
             "Convert tt.assert to cf.assert">,
  ];
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect",
    "math::MathDialect",
    "linalg::LinalgDialect",
    "affine::AffineDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
    "tensor::TensorDialect",
    "bufferization::BufferizationDialect"
  ];
}

def TritonPidsToFuncArgs : Pass<"triton-pids-to-func-args", "mlir::ModuleOp"> {
  let summary = "Convert tt.get_program_id and tt.get_num_programs to function arguments";
}

#endif
