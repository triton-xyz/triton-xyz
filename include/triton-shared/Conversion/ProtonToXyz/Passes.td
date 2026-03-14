#ifndef PROTON_TO_XYZ_CONVERSION_PASSES
#define PROTON_TO_XYZ_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ProtonToXyz : Pass<"proton-to-xyz", "mlir::ModuleOp"> {
  let summary = "Lower proton.record ops to CPU runtime calls";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}

#endif
