#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Pipelines/Pipelines.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

void mlir::triton::buildTritonToLinalgTTAPipeline(
    OpPassManager &pm, const TritonToLinalgPipelineOptions &options) {
  // TODO: Support block tensor pointers with boundary semantics directly in
  // the TTA route so this pre-normalization pass can become optional.
  pm.addPass(createTritonRewriteTensorPointer());

  pm.addPass(createTritonToTTAStructured());
  pm.addPass(createTritonToTTAUnstructured());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createTritonUnstructuredFallback());

  if (options.pidsToFuncArgs) {
    pm.addPass(createTritonPidsToFuncArgs());
  }
  TritonArithToLinalgOptions tritonArithToLinalgOptions;
  tritonArithToLinalgOptions.ttToFuncFunc = options.ttToFuncFunc;
  tritonArithToLinalgOptions.assertToCf = options.assertToCf;
  pm.addPass(createTritonArithToLinalg(tritonArithToLinalgOptions));
  pm.addPass(createTritonTensorPtrToLinalg());

  pm.addPass(createTTAToMemref());
  pm.addPass(createTritonPtrToMemref());

  pm.addPass(createTritonToPtr());
  pm.addPass(createTritonTtPtrToPtr());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createReconcilePtrCasts());

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

void mlir::triton::registerTritonToLinalgTTAPipelines() {
  PassPipelineRegistration<TritonToLinalgPipelineOptions>(
      "triton-to-linalg-tta", "Convert Triton to Linalg dialect via TTA.",
      buildTritonToLinalgTTAPipeline);
}
