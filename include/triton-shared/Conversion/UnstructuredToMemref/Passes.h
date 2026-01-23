#pragma once

#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h.inc"

} // namespace triton
} // namespace mlir
