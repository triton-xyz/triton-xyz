#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TTAADDRESSNORMALIZE
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class TTAAddressNormalizePass
    : public mlir::triton::impl::TTAAddressNormalizeBase<
          TTAAddressNormalizePass> {
  using Base =
      mlir::triton::impl::TTAAddressNormalizeBase<TTAAddressNormalizePass>;
  using Base::Base;

public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    tta::FromTTPtrOp::getCanonicalizationPatterns(patterns, &getContext());
    tta::ReindexOp::getCanonicalizationPatterns(patterns, &getContext());
    tta::AdvanceOp::getCanonicalizationPatterns(patterns, &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
