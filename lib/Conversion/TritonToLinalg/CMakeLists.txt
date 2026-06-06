add_triton_library(TritonToLinalg
  TritonPtrToMemrefPass.cpp
  ConvertXyzToLLVMPass.cpp
  TritonScanToSCFPass.cpp

  TritonUnstructuredFallbackPass.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRPtrDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRPtrToLLVM
  TritonIR
  TritonTransforms
  TritonAddressIR
  TritonSharedAnalysis
)
