set(LLVM_TARGET_DEFINITIONS TritonXyzOps.td)
mlir_tablegen(TritonXyzOps.h.inc -gen-op-decls)
mlir_tablegen(TritonXyzOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TritonXyzOpsGen)
