#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h.inc"

} // namespace triton
} // namespace mlir
