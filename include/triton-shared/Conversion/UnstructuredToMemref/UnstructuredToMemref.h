#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnstructuredToMemrefPass();

} // namespace triton
} // namespace mlir
