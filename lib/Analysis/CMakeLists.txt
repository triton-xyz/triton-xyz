add_triton_library(TritonSharedAnalysis
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrExprAnalysis.cpp
  UseAnalysis.cpp

  DEPENDS
  TritonTableGen
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
