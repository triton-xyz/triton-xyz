get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

set(triton_shared_tool_libs
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIRRegisterAllDialects
)

add_llvm_executable(triton-xyz-opt triton-xyz-opt.cpp PARTIAL_SOURCES_INTENDED)

target_link_libraries(triton-xyz-opt PRIVATE
  ${triton_shared_tool_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-xyz-opt)

add_llvm_executable(triton-xyz-lsp triton-xyz-lsp.cpp PARTIAL_SOURCES_INTENDED)

target_link_libraries(triton-xyz-lsp PRIVATE
  ${triton_shared_tool_libs}
  MLIRLspServerLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
)

mlir_check_all_link_libraries(triton-xyz-lsp)
