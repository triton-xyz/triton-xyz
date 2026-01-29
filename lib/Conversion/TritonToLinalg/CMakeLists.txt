add_triton_library(TritonToLinalg
  StructuredToMemref.cpp
  StructuredToMemrefPass.cpp
  TritonPtrToMemrefPass.cpp
  ReconcilePtrCastsPass.cpp
  TritonToPtrPass.cpp
  CollapseShape.cpp
  TritonToStructuredPass.cpp
  TritonToUnstructuredPass.cpp
  UnstructuredToMemrefPass.cpp

  TritonUnstructuredFallbackPass.cpp
  TritonTensorPtrToLinalgPass.cpp
  NormalizeTensorPtrOrderPass.cpp
  TritonPtrToPtrPass.cpp

  DEPENDS
  TritonStructuredTableGen
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
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
  TritonSharedAnalysis
  TritonSharedAnalysisStructured
  TritonSharedUtils
  TritonArithToLinalg
)
