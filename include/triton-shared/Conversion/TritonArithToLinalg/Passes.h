#pragma once
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir
