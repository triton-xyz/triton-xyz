add_triton_library(TritonToLinalgPipelines
  TritonToLinalgTTA.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  TritonArithToLinalg
  TritonToLinalg
  TritonToLinalgTTA
)
