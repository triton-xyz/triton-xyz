#ifndef TRITON_TO_LINALG_TTA_CONVERSION_PASSES
#define TRITON_TO_LINALG_TTA_CONVERSION_PASSES

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

def TTAToXyz : Pass<"tta-to-xyz", "mlir::ModuleOp"> {
  let summary = "Convert indirect tta load/store to xyz gather/scatter ops";
  let dependentDialects = [
    "tta::TritonAddressDialect",
    "xyz::XyzDialect"
  ];
}

def TTAAddressNormalize : Pass<"tta-address-normalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize and compose tta address chains";
  let dependentDialects = [
    "arith::ArithDialect",
    "tta::TritonAddressDialect"
  ];
}

def VerifyTTABridgeEliminated : Pass<"verify-tta-bridge-eliminated", "mlir::ModuleOp"> {
  let summary = "Verify that tta bridge ops are removed after import stage";
  let dependentDialects = [
    "tta::TritonAddressDialect"
  ];
}

def VerifyTTALowered : Pass<"verify-tta-lowered", "mlir::ModuleOp"> {
  let summary = "Verify that tta address IR is fully lowered away";
  let dependentDialects = [
    "tta::TritonAddressDialect"
  ];
}

#endif
