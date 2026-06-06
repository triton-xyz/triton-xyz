#pragma once

#include "mlir/Pass/PassOptions.h"

namespace mlir::triton {

struct TritonToXyzPipelineOptions
    : public PassPipelineOptions<TritonToXyzPipelineOptions> {
  PassOptions::Option<bool> pidsToFuncArgs{
      *this, "pids-to-func-args",
      llvm::cl::desc("Convert tt.get_program_id and tt.get_num_programs to "
                     "reference to function arguments"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> assertToCf{
      *this, "assert-to-cf", llvm::cl::desc("Convert tt.assert to cf.assert"),
      llvm::cl::init(true)};
};

void buildTritonToXyzPipeline(OpPassManager &pm,
                              const TritonToXyzPipelineOptions &options);

void registerTritonToXyzPipelines();

} // namespace mlir::triton
