#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/AnalysisAddress/AnalysisAddress.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOTTASTRUCTURED
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

using AnalysisAddress = mlir::triton::address::AnalysisAddress;
using TTAEmitter = mlir::triton::address::TTAEmitter;

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

static bool shouldStayForFallback(triton::LoadOp op) {
  if (hasNonEmptyBoundaryCheck(op.getBoundaryCheck())) {
    return true;
  }

  if (Value mask = op.getMask()) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskType || maskType.getRank() != 1) {
      return true;
    }
  }

  return false;
}

static bool shouldStayForFallback(triton::StoreOp op) {
  if (hasNonEmptyBoundaryCheck(op.getBoundaryCheck())) {
    return true;
  }

  if (Value mask = op.getMask()) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskType || maskType.getRank() != 1) {
      return true;
    }
  }

  return false;
}

struct ConvertTTLoadPattern : OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (shouldStayForFallback(op)) {
      return failure();
    }

    auto maybeAddr = getOrCreateAddress(op.getPtr(), op.getLoc(), rewriter);
    if (failed(maybeAddr)) {
      return failure();
    }

    auto maybeMaskDims =
        TTAEmitter::analyzeMaskDims(op.getMask(), op.getLoc(), rewriter);
    if (failed(maybeMaskDims)) {
      return failure();
    }

    auto maybeOther =
        TTAEmitter::getScalarOther(op.getOther(), op.getLoc(), rewriter);
    if (failed(maybeOther)) {
      return failure();
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
    if (shouldStayForFallback(op)) {
      return failure();
    }

    auto maybeAddr = getOrCreateAddress(op.getPtr(), op.getLoc(), rewriter);
    if (failed(maybeAddr)) {
      return failure();
    }

    auto maybeMaskDims =
        TTAEmitter::analyzeMaskDims(op.getMask(), op.getLoc(), rewriter);
    if (failed(maybeMaskDims)) {
      return failure();
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
    AnalysisAddress analysis(/*enableMakeGatherScatterTensorPtr=*/false);
    auto maybeAddr = analysis.analyze(op.getResult(), op.getLoc(), rewriter);
    if (failed(maybeAddr)) {
      return failure();
    }

    auto maybeMakeAddr =
        TTAEmitter::emitMakeAddr(*maybeAddr, op.getLoc(), rewriter);
    if (failed(maybeMakeAddr)) {
      return failure();
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
    if (op->hasAttr("tta.fallback")) {
      return failure();
    }
    rewriter.modifyOpInPlace(
        op, [&]() { op->setAttr("tta.fallback", rewriter.getUnitAttr()); });
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
