#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOTTASTRUCTURED
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

struct ConvertMakeTptrPattern : OpRewritePattern<tts::MakeTensorPtrOp> {
  using OpRewritePattern<tts::MakeTensorPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override {
    auto newOp = tta::MakeAddrOp::create(rewriter, op.getLoc(), op.getBase(),
                                         op.getSizes(), op.getMixedStrides(),
                                         op.getMixedOffsets(),
                                         op.getMixedShape(), op.getOrder());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertLoadPattern : OpRewritePattern<tts::LoadOp> {
  using OpRewritePattern<tts::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto newOp = tta::LoadOp::create(rewriter, op.getLoc(), op.getPtr(),
                                     op.getMixedMaskDims(), op.getOther());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertStorePattern : OpRewritePattern<tts::StoreOp> {
  using OpRewritePattern<tts::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::StoreOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tta::StoreOp>(op, op.getPtr(), op.getValue(),
                                              op.getMixedMaskDims());
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
    patterns
        .add<ConvertMakeTptrPattern, ConvertLoadPattern, ConvertStorePattern>(
            context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
