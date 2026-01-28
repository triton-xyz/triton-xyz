#pragma once

#include "mlir/Pass/PassOptions.h"

namespace mlir::triton {

struct TritonToLinalgPipelineOptions
    : public PassPipelineOptions<TritonToLinalgPipelineOptions> {
  PassOptions::Option<bool> enableMakeGatherScatterTensorPtr{
      *this, "enable-make-gather-scatter",
      llvm::cl::desc("Enable make_gather_scatter_tptr support"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> enableUnstructuredFallback{
      *this, "enable-unstructured-fallback",
      llvm::cl::desc("Enable scalar fallback for tensor-of-ptr memory ops"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> enableCollapseShape{
      *this, "enable-collapse-shape",
      llvm::cl::desc("Enable collapse shape pass"), llvm::cl::init(false)};
  PassOptions::Option<bool> pidsToFuncArgs{
      *this, "pids-to-func-args",
      llvm::cl::desc("Convert tt.get_program_id and tt.get_num_programs to "
                     "reference to function arguments"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> ttToFuncFunc{
      *this, "tt-to-func-func", llvm::cl::desc("Convert tt.func to func.func"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> assertToCf{
      *this, "assert-to-cf", llvm::cl::desc("Convert tt.assert to cf.assert"),
      llvm::cl::init(true)};
};

void buildTritonToLinalgPipeline(OpPassManager &pm,
                                 const TritonToLinalgPipelineOptions &options);

void registerTritonToLinalgPipelines();

} // namespace mlir::triton
