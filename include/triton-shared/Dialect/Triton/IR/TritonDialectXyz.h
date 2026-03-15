#pragma once

#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep

namespace mlir {
class DialectRegistry;
}

namespace mlir::triton::xyz {

void registerTritonDialectXyzExtension(DialectRegistry &registry);

} // namespace mlir::triton::xyz

#define GET_OP_CLASSES
#include "triton-shared/Dialect/Triton/IR/TritonXyzOps.h.inc"
