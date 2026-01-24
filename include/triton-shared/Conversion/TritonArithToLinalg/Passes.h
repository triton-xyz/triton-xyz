#pragma once

#include "mlir/Pass/Pass.h" // IWYU pragma: keep

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

void populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTritonArithToLinalgConversionPatterns(bool pidsToFuncArgs,
                                                   bool assertToCf,
                                                   bool transposeReduceToRank0,
                                                   RewritePatternSet &patterns);

} // namespace triton
} // namespace mlir
