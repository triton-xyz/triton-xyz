#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonToStructuredPass(bool enableMakeGatherScatterTensorPtr = true);

} // namespace triton
} // namespace mlir
