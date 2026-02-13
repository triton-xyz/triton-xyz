add_triton_library(TritonSharedAnalysis
  AnalysisAddress.cpp
  AnalysisStructured.cpp
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrExprAnalysis.cpp

  DEPENDS
  TritonAddressTableGen
  TritonTableGen
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  TritonAddressIR
  TritonStructuredIR
  MLIRAnalysis
)
