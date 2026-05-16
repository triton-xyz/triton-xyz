#pragma once

#include "mlir/Pass/PassOptions.h"

namespace mlir::triton {

struct TritonToLinalgPipelineOptions
    : public PassPipelineOptions<TritonToLinalgPipelineOptions> {
  PassOptions::Option<bool> pidsToFuncArgs{
      *this, "pids-to-func-args",
      llvm::cl::desc("Convert tt.get_program_id and tt.get_num_programs to "
                     "reference to function arguments"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> ttToFuncFunc{
      *this, "tt-to-func-func", llvm::cl::desc("Convert tt.func to func.func"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> assertToCf{
      *this, "assert-to-cf", llvm::cl::desc("Convert tt.assert to cf.assert"),
      llvm::cl::init(true)};
};

void buildTritonToLinalgTTAPipeline(
    OpPassManager &pm, const TritonToLinalgPipelineOptions &options);

void registerTritonToLinalgTTAPipelines();

} // namespace mlir::triton
