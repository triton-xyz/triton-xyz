#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TTANORMALIZE
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

static void populateTTANormalizePatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  tta::FromTTPtrOp::getCanonicalizationPatterns(patterns, context);
  tta::ReindexOp::getCanonicalizationPatterns(patterns, context);
  tta::AdvanceOp::getCanonicalizationPatterns(patterns, context);
}

} // namespace

namespace {

class TTANormalizePass
    : public mlir::triton::impl::TTANormalizeBase<TTANormalizePass> {
  using Base = mlir::triton::impl::TTANormalizeBase<TTANormalizePass>;
  using Base::Base;

public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTTANormalizePatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
