#include "TTAFallbackUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Analysis/AnalysisAddress.h"
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <optional>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOTTASTRUCTURED
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h.inc"
} // namespace mlir::triton

#define DEBUG_TYPE "triton-to-tta-structured"

using namespace mlir;

namespace {

using mlir::triton::tta_conversion::kFallbackAttrName;
using mlir::triton::tta_conversion::kFallbackReasonAttrName;
using mlir::triton::tta_conversion::markFallback;

using AnalysisAddress = mlir::triton::address::AnalysisAddress;
using TTAEmitter = mlir::triton::address::TTAEmitter;

struct AddressAndMaskDims {
  Value addr;
  SmallVector<OpFoldResult> maskDims;
};

static FailureOr<Value>
analyzeAndEmitAddress(Value ptrLike, Location loc, PatternRewriter &rewriter,
                      StringRef &reason,
                      bool enableMakeGatherScatterTensorPtr = true) {
  if (isa<tta::AddrType>(ptrLike.getType())) {
    return ptrLike;
  }

  if (ptrLike.getDefiningOp<tta::FromTTPtrOp>()) {
    return ptrLike;
  }

  AnalysisAddress analysis(enableMakeGatherScatterTensorPtr);
  std::optional<StringRef> analyzeFailureReason;
  auto maybeDescriptor =
      analysis.analyzeDescriptor(ptrLike, loc, rewriter, &analyzeFailureReason);
  if (failed(maybeDescriptor)) {
    if (analyzeFailureReason && !analyzeFailureReason->empty()) {
      reason = *analyzeFailureReason;
    } else {
      reason = "address_analysis_failed";
    }
    return failure();
  }

  std::optional<StringRef> emitFailureReason;
  auto maybeAddress = TTAEmitter::emitAddress(*maybeDescriptor, loc, rewriter,
                                              &emitFailureReason);
  if (failed(maybeAddress)) {
    if (emitFailureReason && !emitFailureReason->empty()) {
      reason = *emitFailureReason;
    } else {
      reason = "emit_address_failed";
    }
    return failure();
  }

  return *maybeAddress;
}

template <typename OpTy>
static std::optional<StringRef> getEarlyFallbackReason(OpTy op) {
  if (!op.getBoundaryCheck().empty()) {
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

template <typename OpTy>
static FailureOr<AddressAndMaskDims>
getAddressAndMaskDims(OpTy op, PatternRewriter &rewriter, StringRef &reason,
                      bool enableMakeGatherScatterTensorPtr = true) {
  auto maybeAddr =
      analyzeAndEmitAddress(op.getPtr(), op.getLoc(), rewriter, reason,
                            enableMakeGatherScatterTensorPtr);
  if (failed(maybeAddr)) {
    return failure();
  }

  auto maybeMaskDims =
      TTAEmitter::analyzeMaskDims(op.getMask(), op.getLoc(), rewriter);
  if (failed(maybeMaskDims)) {
    reason = "mask_analysis_failed";
    return failure();
  }

  return AddressAndMaskDims{*maybeAddr, std::move(*maybeMaskDims)};
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

    StringRef failureReason;
    auto maybeAddressAndMask =
        getAddressAndMaskDims(op, rewriter, failureReason);
    if (failed(maybeAddressAndMask)) {
      markFallback(op, failureReason, rewriter);
      return success();
    }

    auto maybeOther =
        TTAEmitter::getScalarOther(op.getOther(), op.getLoc(), rewriter);
    if (failed(maybeOther)) {
      markFallback(op, "other_not_scalar_splat", rewriter);
      return success();
    }

    SmallVector<Value> dynamicMaskDims;
    SmallVector<int64_t> staticMaskDims;
    dispatchIndexOpFoldResults(maybeAddressAndMask->maskDims, dynamicMaskDims,
                               staticMaskDims);

    auto load = tta::LoadOp::create(
        rewriter, op.getLoc(), op.getType(), maybeAddressAndMask->addr,
        dynamicMaskDims, rewriter.getDenseI64ArrayAttr(staticMaskDims),
        *maybeOther);
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

    StringRef failureReason;
    auto maybeAddressAndMask =
        getAddressAndMaskDims(op, rewriter, failureReason);
    if (failed(maybeAddressAndMask)) {
      markFallback(op, failureReason, rewriter);
      return success();
    }

    rewriter.replaceOpWithNewOp<tta::StoreOp>(op, maybeAddressAndMask->addr,
                                              op.getValue(),
                                              maybeAddressAndMask->maskDims);
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

    StringRef failureReason;
    auto maybeAddress = analyzeAndEmitAddress(op.getResult(), op.getLoc(),
                                              rewriter, failureReason);
    if (failed(maybeAddress)) {
      markFallback(op, failureReason, rewriter);
      return success();
    }

    rewriter.replaceOp(op, *maybeAddress);
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
    markFallback(op, "tensor_ptr_unhandled", rewriter);
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
