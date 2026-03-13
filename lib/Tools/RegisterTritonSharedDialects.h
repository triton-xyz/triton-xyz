#pragma once

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Pipelines/Pipelines.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

// TODO: add a macro
#if __has_include("proton/Dialect/include/Dialect/Proton/IR/Dialect.h")
#define TRITON_XYZ_HAS_PROTON 1
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#endif

inline void registerTritonSharedDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::registerAllExtensions(registry);
  mlir::registerLinalgPasses();

  mlir::triton::registerTritonPasses();

  mlir::triton::registerTritonToLinalgPasses();
  mlir::triton::registerTritonToLinalgTTAPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerTritonToLinalgPipelines();
  mlir::triton::registerTritonToLinalgTTAPipelines();

  registry.insert<
      mlir::ttx::TritonTilingExtDialect, mlir::tta::TritonAddressDialect,
      mlir::tts::TritonStructuredDialect, mlir::triton::TritonDialect>();
#ifdef TRITON_XYZ_HAS_PROTON
  registry.insert<mlir::triton::proton::ProtonDialect>();
#endif
}
