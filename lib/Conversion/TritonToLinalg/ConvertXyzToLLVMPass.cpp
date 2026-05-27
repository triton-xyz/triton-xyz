#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrEnums.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTXYZTOLLVM
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

static LLVM::AtomicOrdering
convertAtomicOrdering(ptr::AtomicOrdering ordering) {
  switch (ordering) {
  case ptr::AtomicOrdering::not_atomic:
    return LLVM::AtomicOrdering::not_atomic;
  case ptr::AtomicOrdering::unordered:
    return LLVM::AtomicOrdering::unordered;
  case ptr::AtomicOrdering::monotonic:
    return LLVM::AtomicOrdering::monotonic;
  case ptr::AtomicOrdering::acquire:
    return LLVM::AtomicOrdering::acquire;
  case ptr::AtomicOrdering::release:
    return LLVM::AtomicOrdering::release;
  case ptr::AtomicOrdering::acq_rel:
    return LLVM::AtomicOrdering::acq_rel;
  case ptr::AtomicOrdering::seq_cst:
    return LLVM::AtomicOrdering::seq_cst;
  }
  return LLVM::AtomicOrdering::not_atomic;
}

static void
addPtrAwareMemRefAddressSpaceConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addTypeAttributeConversion(
      [](BaseMemRefType type, ptr::GenericSpaceAttr memorySpace)
          -> TypeConverter::AttributeConversionResult {
        if (type.getMemorySpace() != memorySpace)
          return TypeConverter::AttributeConversionResult::na();
        return IntegerAttr::get(IntegerType::get(type.getContext(), 32), 0);
      });

  typeConverter.addTypeAttributeConversion(
      [](BaseMemRefType type, LLVM::AddressSpaceAttr memorySpace)
          -> TypeConverter::AttributeConversionResult {
        if (type.getMemorySpace() != memorySpace)
          return TypeConverter::AttributeConversionResult::na();
        return IntegerAttr::get(IntegerType::get(type.getContext(), 32),
                                memorySpace.getAddressSpace());
      });
}

struct TritonBitcastOpConversion
    : public ConvertOpToLLVMPattern<triton::BitcastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), resultType, src);
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

struct ErfOpConversion : public ConvertOpToLLVMPattern<math::ErfOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::ErfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operand = adaptor.getOperand();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    auto floatType = dyn_cast<FloatType>(op.getOperand().getType());
    if (!floatType)
      return failure();
    StringRef funcName = (floatType.getWidth() == 64) ? "erf" : "erff";
    auto module = op->getParentOfType<ModuleOp>();
    if (!module.lookupSymbol(funcName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(
          op.getLoc(), funcName,
          LLVM::LLVMFunctionType::get(resultType, {operand.getType()}));
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, resultType, funcName, ValueRange{operand});
    return success();
  }
};

struct PtrLoadOpConversion : public ConvertOpToLLVMPattern<ptr::LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ptr::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<LLVM::LLVMPointerType>(adaptor.getPtr().getType())) {
      return rewriter.notifyMatchFailure(op, "expected llvm pointer operand");
    }

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert load type");

    auto syncScope = op.getSyncscope().value_or(StringRef());
    auto llvmLoad = LLVM::LoadOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getPtr(),
        op.getAlignment().value_or(0), op.getVolatile_(), op.getNontemporal(),
        op.getInvariant(), op.getInvariantGroup(),
        convertAtomicOrdering(op.getOrdering()), syncScope);

    rewriter.replaceOp(op, llvmLoad.getResult());
    return success();
  }
};

struct PtrStoreOpConversion : public ConvertOpToLLVMPattern<ptr::StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ptr::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<LLVM::LLVMPointerType>(adaptor.getPtr().getType())) {
      return rewriter.notifyMatchFailure(op, "expected llvm pointer operand");
    }

    auto syncScope = op.getSyncscope().value_or(StringRef());
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                          adaptor.getPtr(), op.getAlignment().value_or(0),
                          op.getVolatile_(), op.getNontemporal(),
                          op.getInvariantGroup(),
                          convertAtomicOrdering(op.getOrdering()), syncScope);
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertXyzToLLVMPass
    : public triton::impl::ConvertXyzToLLVMBase<ConvertXyzToLLVMPass> {
  using Base = triton::impl::ConvertXyzToLLVMBase<ConvertXyzToLLVMPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    const auto &dlAnalysis = getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dl = dlAnalysis.getAtOrAbove(moduleOp);
    LowerToLLVMOptions options(&getContext(), dl);
    LLVMTypeConverter typeConverter(&getContext(), options, &dlAnalysis);
    addPtrAwareMemRefAddressSpaceConversions(typeConverter);

    typeConverter.addConversion([&](triton::PointerType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext(),
                                        type.getAddressSpace());
    });

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect, triton::TritonDialect>();

    populateOpConvertToLLVMConversionPatterns(moduleOp, target, typeConverter,
                                              patterns);

    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<ErfOpConversion>(typeConverter);

    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    ptr::populatePtrToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<PtrLoadOpConversion, PtrStoreOpConversion>(typeConverter);

    patterns.add<TritonBitcastOpConversion>(typeConverter);
    target.addIllegalOp<triton::BitcastOp>();
    target.addIllegalOp<ptr::LoadOp, ptr::StoreOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
