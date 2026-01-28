add_triton_library(TritonToLinalgPipelines
  TritonToLinalg.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  TritonArithToLinalg
  TritonToLinalg
)
