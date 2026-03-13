add_triton_library(TritonToLinalg
  LowerProtonCpuInstrumentationPass.cpp
  TritonPtrToMemrefPass.cpp
  ReconcilePtrCastsPass.cpp
  ConvertXyzToLLVMPass.cpp
  TritonToPtrPass.cpp
  CollapseShape.cpp

  TritonUnstructuredFallbackPass.cpp
  TritonTensorPtrToLinalgPass.cpp
  TritonPtrToPtrPass.cpp

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
  TritonAddressIR
  TritonSharedAnalysis
  TritonSharedUtils
  TritonArithToLinalg
  ProtonAnalysis
  ProtonIR
)
