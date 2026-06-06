set(LLVM_TARGET_DEFINITIONS TritonAddressDialect.td)
mlir_tablegen(TritonAddressDialect.h.inc -gen-dialect-decls -dialect=tta)
mlir_tablegen(TritonAddressDialect.cpp.inc -gen-dialect-defs -dialect=tta)
mlir_tablegen(TritonAddressOps.h.inc -gen-op-decls)
mlir_tablegen(TritonAddressOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TritonAddressTypes.td)
mlir_tablegen(TritonAddressTypes.h.inc -gen-typedef-decls -typedefs-dialect=tta)
mlir_tablegen(TritonAddressTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tta)

add_public_tablegen_target(TritonAddressTableGen)
