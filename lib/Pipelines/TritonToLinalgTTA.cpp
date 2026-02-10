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
  if (options.ttaPreRewriteTensorPointer) {
    pm.addPass(createTritonRewriteTensorPointer());
  }

  pm.addPass(createTritonToTTAStructured());
  pm.addPass(createTritonToTTAUnstructured());
  pm.addPass(createTritonUnstructuredFallback());
  pm.addPass(createVerifyTTABridgeEliminated());

  if (options.pidsToFuncArgs) {
    pm.addPass(createTritonPidsToFuncArgs());
  }
  TritonArithToLinalgOptions tritonArithToLinalgOptions;
  tritonArithToLinalgOptions.ttToFuncFunc = options.ttToFuncFunc;
  tritonArithToLinalgOptions.assertToCf = options.assertToCf;
  pm.addPass(createTritonArithToLinalg(tritonArithToLinalgOptions));

  // disable for TTA pipeline
  // pm.addPass(createTritonTensorPtrToLinalg());

  pm.addPass(createTTAToMemref());
  pm.addPass(createTritonPtrToMemref());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // disable for TTA pipeline
  // pm.addPass(createTritonToPtr());
  // pm.addPass(createTritonTtPtrToPtr());
  // pm.addPass(createReconcileUnrealizedCastsPass());
  // pm.addPass(createReconcilePtrCasts());

  pm.addPass(createVerifyTTALowered());
  pm.addPass(createCanonicalizerPass());
}

void mlir::triton::registerTritonToLinalgTTAPipelines() {
  PassPipelineRegistration<TritonToLinalgPipelineOptions>(
      "triton-to-linalg-tta", "Convert Triton to Linalg dialect via TTA.",
      buildTritonToLinalgTTAPipeline);
}
