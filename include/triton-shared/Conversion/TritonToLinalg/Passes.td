#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToStructured : Pass<"triton-to-structured", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", "true",
             "Enable make_gather_scatter_tptr support">,
      Option<"runPrepassOnly", "run-prepass-only", "bool", "false",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">,
      Option<"skipPrepass", "skip-prepass", "bool", "false",
             "Skip the prepass">,
      Option<"useUnsafeMask", "use-unsafe-mask", "bool", "false",
             "Assume that the mask bounds are never less than starting offsets. May produce incorrect results.">
  ];
  let dependentDialects = [
    "scf::SCFDialect",
    "tts::TritonStructuredDialect"
  ];
}

def TritonToTTAStructured : Pass<"triton-to-tta-structured", "mlir::ModuleOp"> {
  let summary = "Convert Triton block pointer ops to tta address ops";
  let dependentDialects = [
    "tta::TritonAddressDialect",
    "triton::TritonDialect"
  ];
}

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "bufferization::BufferizationDialect",
    "memref::MemRefDialect"
  ];
}

def TritonToUnstructured : Pass<"triton-to-unstructured", "mlir::ModuleOp"> {
  let summary = "Transforms tt.addptr ops into offset accumulation ops";
  let options = [
      Option<"offsetBitWidth", "offset-bit-width", "size_t", "32",
             "Bitwidth used for the starting offset of each pointer">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "triton::TritonDialect",
    "tts::TritonStructuredDialect"
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

def UnstructuredToMemref : Pass<"unstructured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert unstructured triton ptr (gather / scatter) to memref";
  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "bufferization::BufferizationDialect",
    "memref::MemRefDialect"
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

def NormalizeTensorPtrOrder : Pass<"normalize-tensor-ptr-order", "mlir::ModuleOp"> {
  let summary = "Normalize block pointer order for tts.make_tptr";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "triton::TritonDialect",
    "tts::TritonStructuredDialect"
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

#endif
