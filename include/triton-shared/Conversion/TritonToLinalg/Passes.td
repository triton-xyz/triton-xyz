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

def TritonToPtr : Pass<"triton-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the Ptr dialect";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "ptr::PtrDialect"
  ];
}

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between tt.ptr or ptr.ptr to memref to to_memref or from_memref";
  let dependentDialects = [
    "ptr::PtrDialect",
    "memref::MemRefDialect"
  ];
}

def ConvertXyzToLLVM : Pass<"convert-xyz-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert xyz ptr/memref IR to LLVM with ptr-aware memory space mapping";
  let dependentDialects = [
    "ptr::PtrDialect",
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect"
  ];
}

def CollapseShape : Pass<"collapse-shape", "mlir::ModuleOp"> {
  let summary = "Compress tensor dimensions to improve linalg{broadcast,transpose,fill,reduce} efficiency";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect"
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

def TritonTensorPtrToLinalg : Pass<"triton-tensor-ptr-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert triton ops on tensor of pointers to linalg.generic";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "triton::TritonDialect"
  ];
}

def TritonTtPtrToPtr : Pass<"triton-tt-ptr-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton tt.ptr function signatures to ptr.ptr";
  let dependentDialects = [
    "func::FuncDialect",
    "ptr::PtrDialect",
    "triton::TritonDialect"
  ];
}

// TODO: mv out
def LowerProtonCpuInstrumentation : Pass<"lower-proton-cpu-instrumentation", "mlir::ModuleOp"> {
  let summary = "Lower proton.record ops to CPU runtime calls";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}

#endif
