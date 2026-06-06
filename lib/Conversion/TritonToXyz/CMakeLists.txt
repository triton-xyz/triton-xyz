add_triton_library(TritonToXyz
  TTANormalizePass.cpp
  TTAToMemrefPass.cpp
  TritonArithToLinalg.cpp
  TritonArithToLinalgPass.cpp
  TritonPidsToFuncArgsPass.cpp
  TritonPtrToMemrefPass.cpp
  TritonScanToSCFPass.cpp
  TritonToTTAStructuredPass.cpp
  TritonToTTAUnstructuredPass.cpp
  TritonUnstructuredFallbackPass.cpp
  VerifyTTABridgeEliminatedPass.cpp
  VerifyTTALoweredPass.cpp

  DEPENDS
  TritonToXyzConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRLinalgTransforms
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
  TritonIR
  TritonTransforms
  TritonAddressIR
  TritonSharedAnalysis
  TritonSharedUtils
)
