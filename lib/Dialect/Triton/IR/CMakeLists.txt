add_triton_library(TritonDialectXyz
  TritonDialectXyz.cpp

  DEPENDS
  TritonXyzOpsGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
)
