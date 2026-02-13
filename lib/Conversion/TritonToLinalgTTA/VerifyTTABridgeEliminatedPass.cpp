#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_VERIFYTTABRIDGEELIMINATED
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class VerifyTTABridgeEliminatedPass
    : public mlir::triton::impl::VerifyTTABridgeEliminatedBase<
          VerifyTTABridgeEliminatedPass> {
  using Base = mlir::triton::impl::VerifyTTABridgeEliminatedBase<
      VerifyTTABridgeEliminatedPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool hasError = false;

    moduleOp.walk([&](tta::FromTTPtrOp op) {
      op.emitOpError("must be eliminated before tta mid-lowering stage");
      hasError = true;
    });

    if (hasError) {
      signalPassFailure();
    }
  }
};

} // namespace
