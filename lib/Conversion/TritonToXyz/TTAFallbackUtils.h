#pragma once

#include "XyzWarnUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::tta_conversion {

inline constexpr StringLiteral kFallbackAttrName = "tta.fallback";
inline constexpr StringLiteral kFallbackReasonAttrName = "tta.fallback_reason";

inline void markFallback(Operation *op, StringRef reason) {
  if (!op) {
    return;
  }

  MLIRContext *ctx = op->getContext();
  op->setAttr(kFallbackAttrName, UnitAttr::get(ctx));
  op->setAttr(kFallbackReasonAttrName, StringAttr::get(ctx, reason));
  xyz_conversion::markScalarFallbackWarn(op);
}

inline void markFallback(Operation *op, StringRef reason,
                         PatternRewriter &rewriter) {
  if (!op) {
    return;
  }

  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(kFallbackAttrName, rewriter.getUnitAttr());
    op->setAttr(kFallbackReasonAttrName, rewriter.getStringAttr(reason));
  });
  xyz_conversion::markScalarFallbackWarn(op);
}

} // namespace mlir::triton::tta_conversion
