#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOLINALGEXPERIMENTAL
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace mlir::triton

namespace {

class TritonToLinalgExperimentalPass
    : public triton::impl::TritonToLinalgExperimentalBase<
          TritonToLinalgExperimentalPass> {
  using Base = triton::impl::TritonToLinalgExperimentalBase<
      TritonToLinalgExperimentalPass>;
  using Base::Base;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    ttx::TritonTilingExtDialect, tts::TritonStructuredDialect,
                    tptr::TPtrDialect, ptr::PtrDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());

    TritonToStructuredOptions triton_to_structured_options;
    triton_to_structured_options.enableMakeGatherScatterTensorPtr =
        enableMakeGatherScatterTensorPtr;
    pm.addPass(createTritonToStructured(triton_to_structured_options));

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createTritonToUnstructured());
    TritonArithToLinalgOptions triton_arith_to_linalg_options;
    triton_arith_to_linalg_options.tensorPtrToLinalg = true;
    pm.addPass(createTritonArithToLinalg(triton_arith_to_linalg_options));

    pm.addPass(createStructuredToMemref());
    pm.addPass(createUnstructuredToMemref());
    pm.addPass(createTritonPtrToMemref());
    pm.addPass(createTritonToPtr());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCasts());

    // Now that remove-dead-values fully works with linalg ops, clean up the IR
    // again, particularly unused loop iter-args that were created
    // during triton-to-structured.
    pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (enableCollapseShape) {
      // Canonicalizer pass will rewrite tensor.expand_shape(linalg.fill) to
      // linalg.fill(tensor.expand_shape) so we need to run it before
      // collapseShape pass
      pm.addPass(createCollapseShape());
    }

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace
