#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

void populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTritonArithToLinalgConversionPatterns(bool pidsToFuncArgs,
                                                   bool addptrToLinalg,
                                                   bool assertToCf,
                                                   bool transposeReduceToRank0,
                                                   RewritePatternSet &patterns);

// Expand the triton pointer ops operating on pointers to linalg
void populateTritonTensorPtrConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>>
createTritonArithToLinalgPass(bool tensorPtrToLinalg = false,
                              bool transposeReduceToRank0 = true);

} // namespace triton
} // namespace mlir
