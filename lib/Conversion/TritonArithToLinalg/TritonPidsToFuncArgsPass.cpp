#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h" // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "triton-pids-to-func-args"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONPIDSTOFUNCARGS
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// get_program_id and get_num_programs:
// When launching triton kernels, we pass 6 additional arguments to indicate
// num_programs and program_id. Amongst those six, we have 3 arguments
// correspond to each axis for num_programs followed by 3 additional arguments
// for program_id.
//
// For instance, with triton kernel example_kernel(a, b, c), we have:
//  example_kernel(
//    a, b, c,
//    num_programs_axis_0,
//    num_programs_axis_1,
//    num_programs_axis_2,
//    program_id_axis_0,
//    program_id_axis_1,
//    program_id_axis_2,
//   )
//
struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      triton::getMaxEnumValForProgramIDDim() + 1;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

struct GetNumProgramsConverter
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  using OpConversionPattern<triton::GetNumProgramsOp>::OpConversionPattern;

  static uint32_t constexpr LAUNCH_GRID_RANK =
      triton::getMaxEnumValForProgramIDDim() + 1;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK * 2 + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

class TritonPidsToFuncArgsPass
    : public triton::impl::TritonPidsToFuncArgsBase<TritonPidsToFuncArgsPass> {
  using Base = triton::impl::TritonPidsToFuncArgsBase<TritonPidsToFuncArgsPass>;
  using Base::Base;

  static auto constexpr LAUNCH_GRID_RANK =
      triton::getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

  // Add additional I32 arguments to represent:
  // - num_programs, 3 in total, one for each axis of the launch grid
  // - program_id, 3 in total, one for each axis of the launch grid
  static void addProgramInfo(triton::FuncOp func) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add empty attributes for each new argument if needed
    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    // Add the corresponding arguments to function body
    for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    for (auto func : moduleOp.getOps<triton::FuncOp>()) {
      addProgramInfo(func);
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<GetProgramIDConverter, GetNumProgramsConverter>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, triton::FuncOp, triton::ReturnOp>();
    target.addIllegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
