#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/Xyz/IR/XyzDialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TTATOXYZ
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

struct ConvertIndirectLoadPattern : public OpRewritePattern<tta::LoadOp> {
  using OpRewritePattern<tta::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tta::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getMixedMaskDims().empty()) {
      return failure();
    }

    auto indirect = op.getPtr().getDefiningOp<tta::IndirectReindexOp>();
    if (!indirect) {
      return failure();
    }

    auto gather = xyz::GatherOp::create(
        rewriter, op.getLoc(), op.getType(), indirect.getAddress(),
        indirect.getIndirectIndex(), indirect.getIndirectDimAttr(),
        indirect.getMask(), op.getOther());
    rewriter.replaceOp(op, gather.getResult());
    return success();
  }
};

struct ConvertIndirectStorePattern : public OpRewritePattern<tta::StoreOp> {
  using OpRewritePattern<tta::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tta::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getMixedMaskDims().empty()) {
      return failure();
    }

    auto indirect = op.getPtr().getDefiningOp<tta::IndirectReindexOp>();
    if (!indirect) {
      return failure();
    }

    xyz::ScatterOp::create(rewriter, op.getLoc(), indirect.getAddress(),
                           indirect.getIndirectIndex(), op.getValue(),
                           indirect.getIndirectDimAttr(), indirect.getMask());
    rewriter.eraseOp(op);
    return success();
  }
};

class TTAToXyzPass : public mlir::triton::impl::TTAToXyzBase<TTAToXyzPass> {
  using Base = mlir::triton::impl::TTAToXyzBase<TTAToXyzPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertIndirectLoadPattern, ConvertIndirectStorePattern>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
