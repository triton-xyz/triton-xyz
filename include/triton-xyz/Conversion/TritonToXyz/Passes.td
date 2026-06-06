#ifndef TRITON_TO_XYZ_CONVERSION_PASSES
#define TRITON_TO_XYZ_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToTTAStructured : Pass<"triton-to-tta-structured", "mlir::ModuleOp"> {
  let summary = "Convert Triton block pointer ops to tta address ops";
  let dependentDialects = [
    "tta::TritonAddressDialect",
    "triton::TritonDialect"
  ];
}

def TritonToTTAUnstructured : Pass<"triton-to-tta-unstructured", "mlir::ModuleOp"> {
  let summary = "Lower Triton unstructured pointer chains directly to tta";
  let options = [
      Option<"offsetBitWidth", "offset-bit-width", "size_t", "32",
             "Bitwidth used for the starting offset of each pointer">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "triton::TritonDialect",
    "tta::TritonAddressDialect"
  ];
}

def TritonUnstructuredFallback : Pass<"triton-unstructured-fallback", "mlir::ModuleOp"> {
  let summary = "Scalarize tensor-of-ptr memory ops as a fallback";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "triton::TritonDialect"
  ];
}

def VerifyTTABridgeEliminated : Pass<"verify-tta-bridge-eliminated", "mlir::ModuleOp"> {
  let summary = "Verify that tta bridge ops are removed after import stage";
  let dependentDialects = [
    "tta::TritonAddressDialect"
  ];
}

def TTANormalize : Pass<"tta-normalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize and compose tta address chains";
  let dependentDialects = [
    "arith::ArithDialect",
    "tta::TritonAddressDialect"
  ];
}

def TritonPidsToFuncArgs : Pass<"triton-pids-to-func-args", "mlir::ModuleOp"> {
  let summary = "Convert tt.get_program_id and tt.get_num_programs to function arguments";
}

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

def TTAToMemref : Pass<"tta-to-memref", "mlir::ModuleOp"> {
  let summary = "Lower tta load/store directly to memref/linalg/scf";
  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "bufferization::BufferizationDialect",
    "memref::MemRefDialect",
    "tta::TritonAddressDialect"
  ];
}

def TritonScanToSCF : Pass<"triton-scan-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower triton scan operations to SCF loops";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "triton::TritonDialect"
  ];
}

def TritonPtrToMemref : Pass<"triton-ptr-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton pointer to unranked memref";
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def VerifyTTALowered : Pass<"verify-tta-lowered", "mlir::ModuleOp"> {
  let summary = "Verify that tta address IR is fully lowered away";
  let dependentDialects = [
    "tta::TritonAddressDialect"
  ];
}

#endif
