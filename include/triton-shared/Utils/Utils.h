#pragma once

#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep

namespace mlir {
namespace triton {
// Return true if the input type is a triton pointer or a tensor of triton
// pointers
bool isPtrTypeLike(Type t);
} // namespace triton

} // namespace mlir
