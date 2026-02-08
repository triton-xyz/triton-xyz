add_triton_library(TritonToLinalgTTA
  TTAToMemrefPass.cpp
  TritonToTTAStructuredPass.cpp
  TritonToTTAUnstructuredPass.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRPtrDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  MLIRPtrToLLVM
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonAddressIR
  TritonSharedAnalysis
  TritonSharedUtils
  TritonArithToLinalg
)
