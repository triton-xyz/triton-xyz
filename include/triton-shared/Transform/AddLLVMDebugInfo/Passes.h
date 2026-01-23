#pragma once

#include "triton-shared/Transform/AddLLVMDebugInfo/AddLLVMDebugInfo.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Transform/AddLLVMDebugInfo/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Transform/AddLLVMDebugInfo/Passes.h.inc"

} // namespace triton
} // namespace mlir
