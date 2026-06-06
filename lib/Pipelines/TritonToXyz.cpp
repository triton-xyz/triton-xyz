#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/TritonToXyz/Passes.h"
#include "triton-shared/Pipelines/Pipelines.h"

void mlir::triton::buildTritonToXyzPipeline(
    OpPassManager &pm, const TritonToXyzPipelineOptions &options) {
  pm.addPass(createTritonToTTAStructured());
  pm.addPass(createTritonToTTAUnstructured());
  pm.addPass(createTritonUnstructuredFallback());
  pm.addPass(createVerifyTTABridgeEliminated());
  pm.addPass(createTTANormalize());

  if (options.pidsToFuncArgs) {
    pm.addPass(createTritonPidsToFuncArgs());
  }
  TritonArithToLinalgOptions tritonArithToLinalgOptions;
  tritonArithToLinalgOptions.assertToCf = options.assertToCf;
  pm.addPass(createTritonArithToLinalg(tritonArithToLinalgOptions));

  pm.addPass(createTTAToMemref());
  pm.addPass(createTritonScanToSCF());
  pm.addPass(createTritonPtrToMemref());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createVerifyTTALowered());
}

void mlir::triton::registerTritonToXyzPipelines() {
  PassPipelineRegistration<TritonToXyzPipelineOptions>(
      "triton-to-xyz", "Convert Triton to XYZ IR.", buildTritonToXyzPipeline);
}
