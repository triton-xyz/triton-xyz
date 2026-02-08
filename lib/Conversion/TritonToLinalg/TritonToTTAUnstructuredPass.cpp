#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <optional>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOTTAUNSTRUCTURED
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

static std::optional<int64_t> getStaticOffsetSize(Type offsetType) {
  auto tensorType = dyn_cast<RankedTensorType>(offsetType);
  if (!tensorType || tensorType.getRank() != 1 || tensorType.isDynamicDim(0)) {
    return std::nullopt;
  }
  if (!tensorType.getElementType().isInteger(32) &&
      !tensorType.getElementType().isInteger(64)) {
    return std::nullopt;
  }
  return tensorType.getShape()[0];
}

static FailureOr<tta::MakeAddrOp> buildLinearMakeAddr(PatternRewriter &rewriter,
                                                      Location loc, Value ptr,
                                                      Value offset) {
  if (!isa<triton::PointerType>(ptr.getType())) {
    return failure();
  }

  auto offsetSize = getStaticOffsetSize(offset.getType());
  if (!offsetSize) {
    return failure();
  }

  SmallVector<int64_t> sizes{*offsetSize};
  SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
  SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0)};
  SmallVector<OpFoldResult> shape{rewriter.getIndexAttr(0)};
  return tta::MakeAddrOp::create(rewriter, loc, ptr, sizes, strides, offsets,
                                 shape, ArrayRef<int32_t>{});
}

static FailureOr<Value> getScalarOther(PatternRewriter &rewriter,
                                       tts::GatherOp op) {
  if (Value other = op.getOther()) {
    auto scalar = tts::utils::getScalarValue(other, op.getLoc(), rewriter);
    if (!scalar) {
      return failure();
    }
    return scalar;
  }

  Type elemType = mlir::getElementTypeOrSelf(op.getType());
  auto zeroAttr = rewriter.getZeroAttr(elemType);
  if (!zeroAttr) {
    return failure();
  }
  return arith::ConstantOp::create(rewriter, op.getLoc(), zeroAttr).getResult();
}

struct ConvertMakeGatherScatterPattern
    : OpRewritePattern<tts::MakeGatherScatterTensorPtrOp> {
  using OpRewritePattern<tts::MakeGatherScatterTensorPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::MakeGatherScatterTensorPtrOp op,
                                PatternRewriter &rewriter) const override {
    auto rank = op.getSizes().size();
    SmallVector<OpFoldResult> shape(rank, rewriter.getIndexAttr(0));

    auto makeAddr = tta::MakeAddrOp::create(
        rewriter, op.getLoc(), op.getBase(), op.getSizes(),
        op.getMixedStrides(), op.getMixedOffsets(), shape, ArrayRef<int32_t>{});

    auto reindex = tta::ReindexOp::create(
        rewriter, op.getLoc(), makeAddr.getResult(),
        op.getGatherScatterOffset(), op.getGatherScatterMask(),
        op.getGatherScatterDim(), op.getMixedOffsets());

    rewriter.replaceOp(op, reindex.getResult());
    return success();
  }
};

struct ConvertGatherPattern : OpRewritePattern<tts::GatherOp> {
  using OpRewritePattern<tts::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto makeAddr =
        buildLinearMakeAddr(rewriter, op.getLoc(), op.getPtr(), op.getOffset());
    if (failed(makeAddr)) {
      return failure();
    }

    SmallVector<OpFoldResult> zeroOffsets{rewriter.getIndexAttr(0)};

    tta::ReindexOp reindex;
    if (Value mask = op.getMask()) {
      reindex =
          tta::ReindexOp::create(rewriter, op.getLoc(), makeAddr->getResult(),
                                 op.getOffset(), mask, 0, zeroOffsets);
    } else {
      reindex =
          tta::ReindexOp::create(rewriter, op.getLoc(), makeAddr->getResult(),
                                 op.getOffset(), 0, zeroOffsets);
    }

    auto other = getScalarOther(rewriter, op);
    if (failed(other)) {
      return failure();
    }

    auto load = tta::LoadOp::create(rewriter, op.getLoc(), reindex.getResult(),
                                    ArrayRef<OpFoldResult>{}, *other);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct ConvertScatterPattern : OpRewritePattern<tts::ScatterOp> {
  using OpRewritePattern<tts::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto makeAddr =
        buildLinearMakeAddr(rewriter, op.getLoc(), op.getPtr(), op.getOffset());
    if (failed(makeAddr)) {
      return failure();
    }

    SmallVector<OpFoldResult> zeroOffsets{rewriter.getIndexAttr(0)};

    tta::ReindexOp reindex;
    if (Value mask = op.getMask()) {
      reindex =
          tta::ReindexOp::create(rewriter, op.getLoc(), makeAddr->getResult(),
                                 op.getOffset(), mask, 0, zeroOffsets);
    } else {
      reindex =
          tta::ReindexOp::create(rewriter, op.getLoc(), makeAddr->getResult(),
                                 op.getOffset(), 0, zeroOffsets);
    }

    Value value = op.getValue();
    if (!isa<RankedTensorType>(value.getType())) {
      value = tensor::FromElementsOp::create(rewriter, op.getLoc(),
                                             ValueRange{value})
                  .getResult();
    }

    rewriter.replaceOpWithNewOp<tta::StoreOp>(op, reindex.getResult(), value,
                                              ArrayRef<OpFoldResult>{});
    return success();
  }
};

class TritonToTTAUnstructuredPass
    : public mlir::triton::impl::TritonToTTAUnstructuredBase<
          TritonToTTAUnstructuredPass> {
public:
  using Base = mlir::triton::impl::TritonToTTAUnstructuredBase<
      TritonToTTAUnstructuredPass>;
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertMakeGatherScatterPattern, ConvertGatherPattern,
                 ConvertScatterPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
