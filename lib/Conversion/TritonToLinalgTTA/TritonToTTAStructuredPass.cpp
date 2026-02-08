#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOTTASTRUCTURED
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

static bool isBlockPointerLike(Type type) {
  auto ptrType = dyn_cast<triton::PointerType>(type);
  return ptrType && isa<RankedTensorType>(ptrType.getPointeeType());
}

static FailureOr<SmallVector<OpFoldResult>>
castValuesToIndexOfr(ValueRange values, Location loc,
                     PatternRewriter &rewriter) {
  SmallVector<OpFoldResult> result;
  result.reserve(values.size());

  for (Value value : values) {
    if (value.getType().isIndex()) {
      result.push_back(value);
      continue;
    }
    if (value.getType().isIntOrIndex()) {
      auto cast = arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getIndexType(), value);
      result.push_back(cast.getResult());
      continue;
    }
    return failure();
  }

  return result;
}

struct ConvertTTMakeTensorPtrPattern
    : OpRewritePattern<triton::MakeTensorPtrOp> {
  using OpRewritePattern<triton::MakeTensorPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBlockPointerLike(op.getType())) {
      return failure();
    }

    auto ptrType = cast<triton::PointerType>(op.getType());
    auto blockType = cast<RankedTensorType>(ptrType.getPointeeType());

    auto maybeStrides =
        castValuesToIndexOfr(op.getStrides(), op.getLoc(), rewriter);
    if (failed(maybeStrides)) {
      return failure();
    }
    auto maybeOffsets =
        castValuesToIndexOfr(op.getOffsets(), op.getLoc(), rewriter);
    if (failed(maybeOffsets)) {
      return failure();
    }
    auto maybeShape =
        castValuesToIndexOfr(op.getShape(), op.getLoc(), rewriter);
    if (failed(maybeShape)) {
      return failure();
    }

    auto makeAddr = tta::MakeAddrOp::create(
        rewriter, op.getLoc(), op.getBase(), blockType.getShape(),
        *maybeStrides, *maybeOffsets, *maybeShape, op.getOrder());
    rewriter.replaceOp(op, makeAddr.getResult());
    return success();
  }
};

struct ConvertTTAdvancePattern : OpRewritePattern<triton::AdvanceOp> {
  using OpRewritePattern<triton::AdvanceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AdvanceOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBlockPointerLike(op.getType())) {
      return failure();
    }

    auto maybeDeltas =
        castValuesToIndexOfr(op.getOffsets(), op.getLoc(), rewriter);
    if (failed(maybeDeltas)) {
      return failure();
    }

    auto advance = tta::AdvanceOp::create(rewriter, op.getLoc(), op.getPtr(),
                                          *maybeDeltas);
    rewriter.replaceOp(op, advance.getResult());
    return success();
  }
};

struct ConvertTTBlockLoadPattern : OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBlockPointerLike(op.getPtr().getType())) {
      return failure();
    }
    if (op.getMask() || op.getOther() || !op.getBoundaryCheck().empty()) {
      return failure();
    }

    auto load = tta::LoadOp::create(rewriter, op.getLoc(), op.getPtr(),
                                    ArrayRef<OpFoldResult>{}, Value{});
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct ConvertTTBlockStorePattern : OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBlockPointerLike(op.getPtr().getType())) {
      return failure();
    }
    if (op.getMask() || !op.getBoundaryCheck().empty()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tta::StoreOp>(op, op.getPtr(), op.getValue(),
                                              ArrayRef<OpFoldResult>{});
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
    patterns.add<ConvertTTMakeTensorPtrPattern, ConvertTTAdvancePattern,
                 ConvertTTBlockLoadPattern, ConvertTTBlockStorePattern>(
        context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
