#pragma once

#include "mlir/IR/Dialect.h"                      // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"                  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"                 // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h" // IWYU pragma: keep
#include "triton-shared/Dialect/Xyz/IR/XyzDialect.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep

#define GET_OP_CLASSES
#include "triton-shared/Dialect/Xyz/IR/XyzOps.h.inc"
