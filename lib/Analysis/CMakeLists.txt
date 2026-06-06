add_triton_library(TritonXyzAnalysis
  AnalysisAddress.cpp
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrExprAnalysis.cpp

  DEPENDS
  TritonAddressTableGen
  TritonTableGen

  LINK_LIBS PUBLIC
  TritonAddressIR
  TritonXyzUtils
  MLIRAnalysis
)
