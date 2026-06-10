#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::xyz_conversion {

inline constexpr StringLiteral kScalarFallbackWarnAttrName = "xyz.warn";
inline constexpr StringLiteral kScalarFallbackWarnMessage = "scalar_fallback";

inline void markScalarFallbackWarn(Operation *op) {
  if (!op) {
    return;
  }

  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func) {
    func = op->getParentOfType<FunctionOpInterface>();
  }
  if (!func) {
    return;
  }

  MLIRContext *ctx = func->getContext();
  func->setAttr(kScalarFallbackWarnAttrName,
                StringAttr::get(ctx, kScalarFallbackWarnMessage));
}

} // namespace mlir::triton::xyz_conversion
