add_triton_library(TritonToXyzPipelines
  TritonToXyz.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  TritonToXyz
)
