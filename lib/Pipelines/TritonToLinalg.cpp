#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Pipelines/Pipelines.h"

void mlir::triton::buildTritonToLinalgPipeline(
    OpPassManager &pm,
    const TritonToLinalgPipelineOptions &options) {
  TritonToStructuredOptions tritonToStructuredOptions;
  tritonToStructuredOptions.enableMakeGatherScatterTensorPtr =
      options.enableMakeGatherScatterTensorPtr;
  pm.addPass(createTritonToStructured(tritonToStructuredOptions));

  // Erase dead code and fold constants created during lowering.
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(createTritonToUnstructured());
  if (options.enableUnstructuredFallback) {
    pm.addPass(createTritonUnstructuredFallback());
  }

  TritonArithToLinalgOptions tritonArithToLinalgOptions;
  tritonArithToLinalgOptions.pidsToFuncArgs = options.pidsToFuncArgs;
  tritonArithToLinalgOptions.ttToFuncFunc = options.ttToFuncFunc;
  tritonArithToLinalgOptions.assertToCf = options.assertToCf;
  pm.addPass(createTritonArithToLinalg(tritonArithToLinalgOptions));

  pm.addPass(createTritonTensorPtrToLinalg());

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
  if (options.enableCollapseShape) {
    // Canonicalizer pass will rewrite tensor.expand_shape(linalg.fill) to
    // linalg.fill(tensor.expand_shape) so we need to run it before
    // collapseShape pass.
    pm.addPass(createCollapseShape());
  }
}

void mlir::triton::registerTritonToLinalgPipelines() {
  PassPipelineRegistration<TritonToLinalgPipelineOptions>(
      "triton-to-linalg", "Convert Triton to Linalg dialect.",
      buildTritonToLinalgPipeline);
}
