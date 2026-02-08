add_triton_library(TritonToLinalgPipelines
  TritonToLinalg.cpp
  TritonToLinalgTTA.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  TritonArithToLinalg
  TritonToLinalg
  TritonToLinalgTTA
)
