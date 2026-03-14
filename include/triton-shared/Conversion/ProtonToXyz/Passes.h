#pragma once

#include "mlir/IR/BuiltinOps.h" // IWYU pragma: keep
#include "mlir/Pass/Pass.h"     // IWYU pragma: keep

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/ProtonToXyz/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/ProtonToXyz/Passes.h.inc"

} // namespace triton
} // namespace mlir
