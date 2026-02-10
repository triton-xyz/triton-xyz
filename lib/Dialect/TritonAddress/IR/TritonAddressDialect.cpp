#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"

#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep

#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressTypes.cpp.inc"

using namespace mlir;
using namespace mlir::tta;

LogicalResult AddrType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType, int64_t rank,
                               int addressSpace) {
  if (isa<triton::PointerType>(elementType) || isa<TensorType>(elementType)) {
    return emitError()
           << "element type must be scalar (non-pointer, non-tensor)";
  }
  if (rank <= 0) {
    return emitError() << "rank must be greater than 0";
  }
  if (addressSpace < 0) {
    return emitError() << "address space must be non-negative";
  }
  return success();
}

void mlir::tta::TritonAddressDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.cpp.inc"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressOps.cpp.inc"
