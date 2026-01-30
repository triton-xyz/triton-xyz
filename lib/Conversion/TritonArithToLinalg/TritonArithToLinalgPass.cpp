#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "triton-arith-to-linalg"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONARITHTOLINALG
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class TritonArithToLinalgPass
    : public triton::impl::TritonArithToLinalgBase<TritonArithToLinalgPass> {
  using Base = triton::impl::TritonArithToLinalgBase<TritonArithToLinalgPass>;
  using Base::Base;

  LogicalResult applyTensorConcatDecomposition() {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    tensor::populateDecomposeTensorConcatPatterns(patterns);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return failure();
    }
    return success();
  }

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      mlir::triton::populateTritonArithToLinalgCanonicalizationPatterns(
          patterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        tts::TritonStructuredDialect>();

    target.addLegalOp<ModuleOp>();

    target.addLegalOp<triton::FuncOp, triton::ReturnOp>();
    target.addLegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          // Lower dense constant to linalg.fill
          if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            if (!isa<RankedTensorType>(constOp.getResult().getType())) {
              return true;
            }

            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constOp.getValue())) {
              if (denseAttr.isSplat() &&
                  isa<FloatType, IntegerType>(denseAttr.getElementType())) {
                return false;
              }
            }
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return isa<RankedTensorType>(type);
              });

          return !operateOnTensors;
        });

    target.addDynamicallyLegalOp<triton::BitcastOp>([](triton::BitcastOp op) {
      return triton::isPtrTypeLike(op.getType());
    });

    if (!assertToCf) {
      target.addLegalOp<triton::AssertOp>();
    }

    triton::populateTritonArithToLinalgConversionPatterns(
        assertToCf, transposeReduceToRank0, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    if (failed(applyTensorConcatDecomposition())) {
      signalPassFailure();
    }

    // Convert tt.func and tt.return into func's counterparts
    if (ttToFuncFunc) {
      moduleOp.walk([&](triton::FuncOp func) {
        OpBuilder builder(func);

        auto name = func.getName();
        auto type = func.getFunctionType();

        SmallVector<DictionaryAttr> argAttrs, resAttrs;
        func.getAllArgAttrs(argAttrs);
        func.getAllResultAttrs(resAttrs);

        auto funcFunc =
            func::FuncOp::create(builder, func.getLoc(), name, type);
        // Preserve the visibility attribute
        funcFunc.setVisibility(func.getVisibility());
        funcFunc.setAllArgAttrs(argAttrs);
        funcFunc.setAllResultAttrs(resAttrs);

        auto &funcFuncBody = funcFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&funcFuncBody, map);

        for (Block &block : funcFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          // Only convert to func.return if the terminator is a tt.return.
          // Otherwise, we will accidentally convert cf.br ops which are also
          // considered terminators.
          if (isa<triton::ReturnOp>(term)) {
            builder.setInsertionPoint(term);
            func::ReturnOp::create(builder, func.getLoc(), term->getOperands());
            term->erase();
          }
        }
        func.erase();
      });
    }
  }
};

} // namespace
