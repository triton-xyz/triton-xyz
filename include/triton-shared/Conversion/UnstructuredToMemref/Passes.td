#ifndef UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES
#define UNSTRUCTURED_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def UnstructuredToMemref : Pass<"unstructured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert unstructured triton ptr (gather / scatter) to memref";
}

#endif
