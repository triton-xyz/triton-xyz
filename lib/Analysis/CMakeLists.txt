add_triton_library(TritonSharedAnalysis
  AnalysisAddress.cpp
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrExprAnalysis.cpp

  DEPENDS
  TritonAddressTableGen
  TritonTableGen

  LINK_LIBS PUBLIC
  TritonAddressIR
  TritonSharedUtils
  MLIRAnalysis
)
