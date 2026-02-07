#pragma once

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Pipelines/Pipelines.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

inline void registerTritonSharedDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::registerAllExtensions(registry);
  mlir::registerLinalgPasses();

  mlir::triton::registerTritonPasses();

  mlir::triton::registerTritonToLinalgPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerTritonToLinalgPipelines();

  registry.insert<mlir::ttx::TritonTilingExtDialect,
                  mlir::tts::TritonStructuredDialect,
                  mlir::triton::TritonDialect>();
}
