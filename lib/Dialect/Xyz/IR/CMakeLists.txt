add_triton_library(XyzIR
  XyzDialect.cpp
  XyzOps.cpp

  DEPENDS
  XyzTableGen

  LINK_LIBS PUBLIC
  TritonAddressIR
  TritonIR
  MLIRIR
)
