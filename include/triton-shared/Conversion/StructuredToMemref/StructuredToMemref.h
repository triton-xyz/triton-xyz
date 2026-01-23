#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

void populateStructuredToMemrefConversionPatterns(RewritePatternSet &patterns,
                                                  TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createStructuredToMemrefPass();

} // namespace triton
} // namespace mlir
