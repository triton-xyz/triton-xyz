#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"    // IWYU pragma: keep
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"       // IWYU pragma: keep
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"         // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h" // IWYU pragma: keep

//===----------------------------------------------------------------------===//
// Temporary Pointer Dialect Operations
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h.inc"

// Include the auto-generated header files containing the declarations of the
// Temporary Pointer Dialect operations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrAttributes.h.inc"
