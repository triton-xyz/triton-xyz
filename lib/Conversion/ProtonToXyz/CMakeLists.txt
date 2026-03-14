add_triton_library(ProtonToXyz
  ProtonToXyzPass.cpp

  DEPENDS
  ProtonToXyzPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  ProtonAnalysis
  ProtonIR
)
