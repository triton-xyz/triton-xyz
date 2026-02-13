add_triton_library(TritonToLinalgTTA
  TTAAddressNormalizePass.cpp
  TTAToMemrefPass.cpp
  TTAToXyzPass.cpp
  TritonToTTAStructuredPass.cpp
  VerifyTTABridgeEliminatedPass.cpp
  VerifyTTALoweredPass.cpp
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
  XyzIR
  TritonSharedAnalysisAddress
  TritonSharedAnalysis
  TritonSharedUtils
  TritonArithToLinalg
)
