add_triton_library(TritonSharedAnalysisAddress
  AnalysisAddress.cpp

  DEPENDS
  TritonAddressTableGen
  TritonStructuredTableGen

  LINK_LIBS PUBLIC
  TritonAddressIR
  TritonStructuredIR
  TritonSharedAnalysis
)
