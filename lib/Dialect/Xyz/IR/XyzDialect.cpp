#include "triton-shared/Dialect/Xyz/IR/XyzDialect.h"

using namespace mlir;

void mlir::xyz::XyzDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/Xyz/IR/XyzOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "triton-shared/Dialect/Xyz/IR/XyzDialect.cpp.inc"
#include "triton-shared/Dialect/Xyz/IR/XyzOps.cpp.inc"
