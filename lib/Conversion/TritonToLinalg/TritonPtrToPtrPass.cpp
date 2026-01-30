#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h" // IWYU pragma: keep
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTTPTRTOPTR
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values) {
    llvm::append_range(result, vals);
  }
  return result;
}

class TritonPtrSignatureConverter : public TypeConverter {
public:
  TritonPtrSignatureConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) {
      return ptr::PtrType::get(context, ptr::GenericSpaceAttr::get(context));
    });
    addConversion([context](RankedTensorType tensorType) {
      if (isa<triton::PointerType>(tensorType.getElementType())) {
        return RankedTensorType::get(
            tensorType.getShape(),
            ptr::PtrType::get(context, ptr::GenericSpaceAttr::get(context)));
      }
      return tensorType;
    });

    auto createCast = [&](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    addTargetMaterialization(createCast);
    addSourceMaterialization(createCast);
  }
};

struct TritonCallOpSignatureConversion
    : public OpConversionPattern<triton::CallOp> {
  using OpConversionPattern<triton::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp callOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<size_t> numResultsReplacements;
    SmallVector<Type, 1> convertedResults;
    size_t numFlattenedResults = 0;
    for (auto type : callOp.getResultTypes()) {
      if (failed(typeConverter->convertTypes(type, convertedResults))) {
        return failure();
      }
      numResultsReplacements.push_back(convertedResults.size() -
                                       numFlattenedResults);
      numFlattenedResults = convertedResults.size();
    }

    auto newCallOp = triton::CallOp::create(
        rewriter, callOp.getLoc(), callOp.getCallee(), convertedResults,
        flattenValues(adaptor.getOperands()));
    SmallVector<ValueRange> replacements;
    size_t offset = 0;
    for (int i = 0, e = callOp->getNumResults(); i < e; ++i) {
      replacements.push_back(
          newCallOp->getResults().slice(offset, numResultsReplacements[i]));
      offset += numResultsReplacements[i];
    }
    assert(offset == convertedResults.size() &&
           "expected all converted results to be used");
    rewriter.replaceOpWithMultiple(callOp, replacements);
    return success();
  }
};

struct TritonReturnOpTypeConversion
    : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<triton::ReturnOp>(
        op, flattenValues(adaptor.getOperands()));
    return success();
  }
};

class TritonTtPtrToPtrPass
    : public triton::impl::TritonTtPtrToPtrBase<TritonTtPtrToPtrPass> {
  using Base = triton::impl::TritonTtPtrToPtrBase<TritonTtPtrToPtrPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonPtrSignatureConverter typeConverter(&getContext());

    target.addDynamicallyLegalOp<func::FuncOp, triton::FuncOp>([&](auto op) {
      return typeConverter.isSignatureLegal(
          cast<FunctionType>(cast<FunctionOpInterface>(op).getFunctionType()));
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getResultTypes()) &&
             typeConverter.isLegal(op.getOperandTypes());
    });

    target.addDynamicallyLegalOp<triton::CallOp>([&](triton::CallOp op) {
      return typeConverter.isLegal(op.getResultTypes()) &&
             typeConverter.isLegal(op.getOperandTypes());
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });

    target.addDynamicallyLegalOp<triton::ReturnOp>([&](triton::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<TritonCallOpSignatureConversion, TritonReturnOpTypeConversion>(
        typeConverter, patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
