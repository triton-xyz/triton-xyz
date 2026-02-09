#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/AnalysisAddress/AnalysisAddress.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <optional>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOTTASTRUCTURED
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

using AnalysisAddress = mlir::triton::address::AnalysisAddress;
using TTAEmitter = mlir::triton::address::TTAEmitter;

static constexpr StringLiteral kFallbackAttrName = "tta.fallback";
static constexpr StringLiteral kFallbackReasonAttrName = "tta.fallback_reason";

static void markFallback(Operation *op, StringRef reason,
                         PatternRewriter &rewriter) {
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(kFallbackAttrName, rewriter.getUnitAttr());
    op->setAttr(kFallbackReasonAttrName, rewriter.getStringAttr(reason));
  });
}

template <typename T>
static bool hasNonEmptyBoundaryCheck(ArrayRef<T> boundaryCheck) {
  return !boundaryCheck.empty();
}

static FailureOr<Value> getOrCreateAddress(Value ptrLike, Location loc,
                                           PatternRewriter &rewriter) {
  if (ptrLike.getDefiningOp<tta::MakeAddrOp>()) {
    return ptrLike;
  }

  AnalysisAddress analysis(/*enableMakeGatherScatterTensorPtr=*/false);
  auto maybeAddr = analysis.analyze(ptrLike, loc, rewriter);
  if (failed(maybeAddr)) {
    return failure();
  }

  return TTAEmitter::emitMakeAddr(*maybeAddr, loc, rewriter);
}

static std::optional<StringRef> getEarlyFallbackReason(triton::LoadOp op) {
  if (hasNonEmptyBoundaryCheck(op.getBoundaryCheck())) {
    return "boundary_check_not_supported";
  }

  if (Value mask = op.getMask()) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskType || maskType.getRank() != 1) {
      return "mask_rank_not_1d";
    }
  }

  return std::nullopt;
}

static std::optional<StringRef> getEarlyFallbackReason(triton::StoreOp op) {
  if (hasNonEmptyBoundaryCheck(op.getBoundaryCheck())) {
    return "boundary_check_not_supported";
  }

  if (Value mask = op.getMask()) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskType || maskType.getRank() != 1) {
      return "mask_rank_not_1d";
    }
  }

  return std::nullopt;
}

struct ConvertTTLoadPattern : OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(kFallbackAttrName)) {
      return failure();
    }

    if (auto reason = getEarlyFallbackReason(op)) {
      markFallback(op, *reason, rewriter);
      return success();
    }

    auto maybeAddr = getOrCreateAddress(op.getPtr(), op.getLoc(), rewriter);
    if (failed(maybeAddr)) {
      markFallback(op, "address_analysis_failed", rewriter);
      return success();
    }

    auto maybeMaskDims =
        TTAEmitter::analyzeMaskDims(op.getMask(), op.getLoc(), rewriter);
    if (failed(maybeMaskDims)) {
      markFallback(op, "mask_analysis_failed", rewriter);
      return success();
    }

    auto maybeOther =
        TTAEmitter::getScalarOther(op.getOther(), op.getLoc(), rewriter);
    if (failed(maybeOther)) {
      markFallback(op, "other_not_scalar_splat", rewriter);
      return success();
    }

    auto load = tta::LoadOp::create(rewriter, op.getLoc(), *maybeAddr,
                                    *maybeMaskDims, *maybeOther);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct ConvertTTStorePattern : OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(kFallbackAttrName)) {
      return failure();
    }

    if (auto reason = getEarlyFallbackReason(op)) {
      markFallback(op, *reason, rewriter);
      return success();
    }

    auto maybeAddr = getOrCreateAddress(op.getPtr(), op.getLoc(), rewriter);
    if (failed(maybeAddr)) {
      markFallback(op, "address_analysis_failed", rewriter);
      return success();
    }

    auto maybeMaskDims =
        TTAEmitter::analyzeMaskDims(op.getMask(), op.getLoc(), rewriter);
    if (failed(maybeMaskDims)) {
      markFallback(op, "mask_analysis_failed", rewriter);
      return success();
    }

    rewriter.replaceOpWithNewOp<tta::StoreOp>(op, *maybeAddr, op.getValue(),
                                              *maybeMaskDims);
    return success();
  }
};

struct ConvertTTAdvancePattern : OpRewritePattern<triton::AdvanceOp> {
  using OpRewritePattern<triton::AdvanceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AdvanceOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(kFallbackAttrName)) {
      return failure();
    }

    AnalysisAddress analysis(/*enableMakeGatherScatterTensorPtr=*/false);
    auto maybeAddr = analysis.analyze(op.getResult(), op.getLoc(), rewriter);
    if (failed(maybeAddr)) {
      markFallback(op, "address_analysis_failed", rewriter);
      return success();
    }

    auto maybeMakeAddr =
        TTAEmitter::emitMakeAddr(*maybeAddr, op.getLoc(), rewriter);
    if (failed(maybeMakeAddr)) {
      markFallback(op, "emit_make_addr_failed", rewriter);
      return success();
    }

    rewriter.replaceOp(op, *maybeMakeAddr);
    return success();
  }
};

struct MarkUnhandledTensorPtrPattern
    : OpRewritePattern<triton::MakeTensorPtrOp> {
  using OpRewritePattern<triton::MakeTensorPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(kFallbackAttrName) &&
        op->hasAttr(kFallbackReasonAttrName)) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&]() {
      op->setAttr(kFallbackAttrName, rewriter.getUnitAttr());
      if (!op->hasAttr(kFallbackReasonAttrName)) {
        op->setAttr(kFallbackReasonAttrName,
                    rewriter.getStringAttr("tensor_ptr_unhandled"));
      }
    });
    return success();
  }
};

class TritonToTTAStructuredPass
    : public mlir::triton::impl::TritonToTTAStructuredBase<
          TritonToTTAStructuredPass> {
public:
  using Base =
      mlir::triton::impl::TritonToTTAStructuredBase<TritonToTTAStructuredPass>;
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertTTLoadPattern, ConvertTTStorePattern,
                 ConvertTTAdvancePattern, MarkUnhandledTensorPtrPattern>(
        context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
