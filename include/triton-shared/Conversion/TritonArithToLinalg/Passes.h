#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

void populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTritonArithToLinalgConversionPatterns(bool pidsToFuncArgs,
                                                   bool addptrToLinalg,
                                                   bool assertToCf,
                                                   bool transposeReduceToRank0,
                                                   RewritePatternSet &patterns);

// Expand the triton pointer ops operating on pointers to linalg
void populateTritonTensorPtrConversionPatterns(RewritePatternSet &patterns);

} // namespace triton
} // namespace mlir
