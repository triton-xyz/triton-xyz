#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
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

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();

    populateConversionTargetFromOperation(moduleOp, target, typeConverter,
                                          patterns);
    populateOpConvertToLLVMConversionPatterns(moduleOp, target, typeConverter,
                                              patterns);

    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    ptr::populatePtrToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<PtrLoadOpConversion, PtrStoreOpConversion>(typeConverter);

    target.addIllegalOp<ptr::LoadOp, ptr::StoreOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
