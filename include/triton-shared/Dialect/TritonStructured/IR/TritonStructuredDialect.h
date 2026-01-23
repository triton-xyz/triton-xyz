#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace tts {
namespace utils {
mlir::Value getScalarValue(mlir::Value operand, mlir::Location loc,
                           mlir::OpBuilder &builder);
}
} // namespace tts
} // namespace mlir

//===----------------------------------------------------------------------===//
// TritonStructured Operations
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"
