#pragma once

#include "mlir/IR/Dialect.h"      // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h" // IWYU pragma: keep
#include "triton-xyz/Dialect/TritonAddress/IR/TritonAddressDialect.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "triton-xyz/Dialect/TritonAddress/IR/TritonAddressTypes.h.inc"

#define GET_OP_CLASSES
#include "triton-xyz/Dialect/TritonAddress/IR/TritonAddressOps.h.inc"
