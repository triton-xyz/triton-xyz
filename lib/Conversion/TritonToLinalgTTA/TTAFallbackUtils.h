#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::tta_conversion {

inline constexpr StringLiteral kFallbackAttrName = "tta.fallback";
inline constexpr StringLiteral kFallbackReasonAttrName = "tta.fallback_reason";

inline bool hasLoweredTTAAddressRoot(Value value) {
  if (!value) {
    return false;
  }

  return value.getDefiningOp<tta::MakeAddrOp>() ||
         value.getDefiningOp<tta::ReindexOp>() ||
         value.getDefiningOp<tta::AdvanceOp>();
}

inline void markFallback(Operation *op, StringRef reason) {
  if (!op) {
    return;
  }

  MLIRContext *ctx = op->getContext();
  op->setAttr(kFallbackAttrName, UnitAttr::get(ctx));
  op->setAttr(kFallbackReasonAttrName, StringAttr::get(ctx, reason));
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
}

} // namespace mlir::triton::tta_conversion
