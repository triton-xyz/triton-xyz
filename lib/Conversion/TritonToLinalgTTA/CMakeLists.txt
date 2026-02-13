add_triton_library(TritonToLinalgTTA
  TTAAddressNormalizePass.cpp
  TTAToMemrefPass.cpp
  TritonToTTAStructuredPass.cpp
  VerifyTTABridgeEliminatedPass.cpp
  VerifyTTALoweredPass.cpp
  TritonToTTAUnstructuredPass.cpp

  DEPENDS
  TritonToLinalgTTAConversionPassIncGen

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
