#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonPtrToMemref : Pass<"triton-ptr-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton pointer to unranked memref";
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
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

def ConvertXyzToLLVM : Pass<"convert-xyz-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert xyz ptr/memref IR to LLVM with ptr-aware memory space mapping";
  let dependentDialects = [
    "ptr::PtrDialect",
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "triton::TritonDialect"
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

#endif
