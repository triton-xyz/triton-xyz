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
}

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
}

def TritonToUnstructured : Pass<"triton-to-unstructured", "mlir::ModuleOp"> {
  let summary = "Transforms tt.addptr ops into offset accumulation ops";
  let options = [
      Option<"offsetBitWidth", "offset-bit-width", "size_t", "32",
             "Bitwidth used for the starting offset of each pointer">
  ];
}

def UnstructuredToMemref : Pass<"unstructured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert unstructured triton ptr (gather / scatter) to memref";
}

def TritonPtrToMemref : Pass<"triton-ptr-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton pointer to unranked memref";
}

def TritonToPtr : Pass<"triton-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the Ptr dialect";
}

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between tt.ptr or ptr.ptr to memref to to_memref or from_memref";
}

def CollapseShape : Pass<"collapse-shape", "mlir::ModuleOp"> {
  let summary = "Compress tensor dimensions to improve linalg{broadcast,transpose,fill,reduce} efficiency";
}

def TritonUnstructuredFallback : Pass<"triton-unstructured-fallback", "mlir::ModuleOp"> {
  let summary = "Scalarize tensor-of-ptr memory ops as a fallback";
}

def TritonTensorPtrToLinalg : Pass<"triton-tensor-ptr-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert triton ops on tensor of pointers to linalg.generic";
}

def NormalizeTensorPtrOrder : Pass<"normalize-tensor-ptr-order", "mlir::ModuleOp"> {
  let summary = "Normalize block pointer order for tts.make_tptr";
}

def TritonTtPtrToPtr : Pass<"triton-tt-ptr-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton tt.ptr function signatures to ptr.ptr";
}

#endif
