#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"

void mlir::tta::TritonAddressDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.cpp.inc"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressOps.cpp.inc"
