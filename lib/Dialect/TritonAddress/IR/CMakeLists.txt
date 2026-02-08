add_triton_library(TritonAddressIR
  TritonAddressOps.cpp
  TritonAddressDialect.cpp

  DEPENDS
  TritonAddressTableGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
)
