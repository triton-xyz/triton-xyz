#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h" // IWYU pragma: keep
#include "mlir/Pass/Pass.h"                // IWYU pragma: keep
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h.inc"

} // namespace triton
} // namespace mlir
