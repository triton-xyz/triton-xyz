#pragma once

#include "mlir/Pass/Pass.h" // IWYU pragma: keep
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"

void populateStructuredToMemrefConversionPatterns(RewritePatternSet &patterns,
                                                  TypeConverter &typeConverter);

} // namespace triton
} // namespace mlir
