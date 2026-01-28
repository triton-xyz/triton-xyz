#pragma once

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  mlir::registerAllPasses();
  mlir::registerAllExtensions(registry);
  mlir::registerLinalgPasses();

  mlir::triton::registerTritonPasses();

  mlir::triton::registerTritonToLinalgPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerTritonToLinalgPipelines();

  registry
      .insert<mlir::ptr::PtrDialect, mlir::ttx::TritonTilingExtDialect,
              mlir::tts::TritonStructuredDialect, mlir::triton::TritonDialect,
              mlir::cf::ControlFlowDialect, mlir::math::MathDialect,
              mlir::arith::ArithDialect, mlir::scf::SCFDialect,
              mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
              mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
              mlir::bufferization::BufferizationDialect>();
}
