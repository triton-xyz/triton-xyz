set(LLVM_TARGET_DEFINITIONS XyzDialect.td)
mlir_tablegen(XyzDialect.h.inc -gen-dialect-decls -dialect=xyz)
mlir_tablegen(XyzDialect.cpp.inc -gen-dialect-defs -dialect=xyz)
mlir_tablegen(XyzOps.h.inc -gen-op-decls)
mlir_tablegen(XyzOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(XyzTableGen)
