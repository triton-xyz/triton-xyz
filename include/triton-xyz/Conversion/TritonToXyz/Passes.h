#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h" // IWYU pragma: keep
#include "mlir/Pass/Pass.h"                // IWYU pragma: keep

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-xyz/Conversion/TritonToXyz/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-xyz/Conversion/TritonToXyz/Passes.h.inc"

void populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTritonArithToLinalgConversionPatterns(bool assertToCf,
                                                   RewritePatternSet &patterns);

} // namespace triton
} // namespace mlir
