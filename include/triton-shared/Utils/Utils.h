#pragma once

#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep

namespace mlir {
namespace triton {
// Return true if the input type is a triton pointer or a tensor of triton
// pointers
bool isPtrTypeLike(Type t);

// Extract a scalar from a scalar or splat-shaped value, rebuilding simple
// scalar casts when needed.
Value getScalarValue(Value operand, Location loc, OpBuilder &builder);
} // namespace triton

} // namespace mlir
