#pragma once
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h.inc"

} // namespace triton
} // namespace mlir
