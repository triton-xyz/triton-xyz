#ifndef XYZ_TO_LLVM_CONVERSION_PASSES
#define XYZ_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

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

#endif
