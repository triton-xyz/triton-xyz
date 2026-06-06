add_triton_library(XyzToLLVM
  ConvertXyzToLLVMPass.cpp

  DEPENDS
  XyzToLLVMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAnalysis
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrDialect
  MLIRTransforms
  MLIRSupport
  MLIRConvertToLLVMInterface
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPtrToLLVM
  TritonIR
)
