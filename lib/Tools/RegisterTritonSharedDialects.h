#pragma once

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h"
#include "triton-shared/Dialect/Triton/IR/TritonDialectXyz.h"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Pipelines/Pipelines.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#if defined(TRITON_XYZ_BUILD_PROTON)
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton-shared/Conversion/ProtonToXyz/Passes.h"
#endif

inline void registerTritonSharedDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::registerAllExtensions(registry);
  mlir::registerLinalgPasses();

  mlir::triton::xyz::registerTritonDialectXyzExtension(registry);

  mlir::triton::registerTritonPasses();

  mlir::triton::registerTritonToLinalgPasses();
  mlir::triton::registerTritonToLinalgTTAPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerTritonToLinalgTTAPipelines();
#if defined(TRITON_XYZ_BUILD_PROTON)
  mlir::triton::registerProtonToXyzPasses();
#endif

  registry
      .insert<mlir::tta::TritonAddressDialect, mlir::triton::TritonDialect>();
#if defined(TRITON_XYZ_BUILD_PROTON)
  registry.insert<mlir::triton::proton::ProtonDialect>();
#endif
}
