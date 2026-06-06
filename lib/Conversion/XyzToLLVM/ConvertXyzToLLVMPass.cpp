#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrEnums.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/XyzToLLVM/Passes.h" // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTXYZTOLLVM
#include "triton-shared/Conversion/XyzToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

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

} // namespace

namespace {

struct TritonBitcastOpConversion
    : public ConvertOpToLLVMPattern<triton::BitcastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto cast = UnrealizedConversionCastOp::create(rewriter, op.getLoc(),
                                                   resultType, src);
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

static void populateXyzTritonToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<TritonBitcastOpConversion>(typeConverter);
}

} // namespace

namespace {

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
      LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), funcName,
          LLVM::LLVMFunctionType::get(resultType, {operand.getType()}));
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, resultType, funcName,
                                              ValueRange{operand});
    return success();
  }
};

static void
populateXyzMathToLLVMConversionPatterns(const LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  patterns.add<ErfOpConversion>(typeConverter);
}

} // namespace

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

static void
populateXyzPtrToLLVMConversionPatterns(const LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  patterns.add<PtrLoadOpConversion, PtrStoreOpConversion>(typeConverter);
}

} // namespace

namespace {

struct FloatGenericAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<memref::GenericAtomicRMWOp> {
  explicit FloatGenericAtomicRMWOpConversion(
      const LLVMTypeConverter &typeConverter, PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(memref::GenericAtomicRMWOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto floatType = dyn_cast<FloatType>(atomicOp.getResult().getType());
    if (!floatType)
      return failure();

    Location loc = atomicOp.getLoc();
    Type valueType =
        getTypeConverter()->convertType(atomicOp.getResult().getType());
    if (!valueType)
      return failure();
    Type intType =
        IntegerType::get(atomicOp.getContext(), floatType.getWidth());

    auto *initBlock = rewriter.getInsertionBlock();
    auto *loopBlock = rewriter.splitBlock(initBlock, Block::iterator(atomicOp));
    loopBlock->addArgument(intType, loc);

    auto *endBlock =
        rewriter.splitBlock(loopBlock, Block::iterator(atomicOp)++);

    rewriter.setInsertionPointToEnd(initBlock);
    auto memRefType = cast<MemRefType>(atomicOp.getMemref().getType());
    Value dataPtr = getStridedElementPtr(
        rewriter, loc, memRefType, adaptor.getMemref(), adaptor.getIndices());
    Value init = LLVM::LoadOp::create(rewriter, loc, intType, dataPtr);
    LLVM::BrOp::create(rewriter, loc, init, loopBlock);

    rewriter.setInsertionPointToStart(loopBlock);
    Value loopArgument = loopBlock->getArgument(0);
    Value current =
        LLVM::BitcastOp::create(rewriter, loc, valueType, loopArgument);

    IRMapping mapping;
    mapping.map(atomicOp.getCurrentValue(), current);
    Block &entryBlock = atomicOp.body().front();
    for (auto &nestedOp : entryBlock.without_terminator()) {
      Operation *clone = rewriter.clone(nestedOp, mapping);
      mapping.map(nestedOp.getResults(), clone->getResults());
    }

    Value result =
        mapping.lookupOrNull(entryBlock.getTerminator()->getOperand(0));
    if (!result)
      return atomicOp.emitError("result not defined in region");
    Value resultBits = LLVM::BitcastOp::create(rewriter, loc, intType, result);

    auto successOrdering = LLVM::AtomicOrdering::acq_rel;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, dataPtr, loopArgument, resultBits, successOrdering,
        failureOrdering);
    Value newLoaded = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg, 0);
    Value ok = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg, 1);

    LLVM::CondBrOp::create(rewriter, loc, ok, endBlock, ArrayRef<Value>(),
                           loopBlock, newLoaded);

    rewriter.setInsertionPoint(atomicOp);
    Value newLoadedValue =
        LLVM::BitcastOp::create(rewriter, loc, valueType, newLoaded);
    rewriter.replaceOp(atomicOp, {newLoadedValue});
    return success();
  }
};

static void populateXyzMemRefToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<FloatGenericAtomicRMWOpConversion>(typeConverter);
}

} // namespace

namespace {

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

    populateConversionTargetFromOperation(moduleOp, target, typeConverter,
                                          patterns);
    populateOpConvertToLLVMConversionPatterns(moduleOp, target, typeConverter,
                                              patterns);

    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateXyzMathToLLVMConversionPatterns(typeConverter, patterns);

    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateXyzMemRefToLLVMConversionPatterns(typeConverter, patterns);
    ptr::populatePtrToLLVMConversionPatterns(typeConverter, patterns);
    populateXyzPtrToLLVMConversionPatterns(typeConverter, patterns);

    populateXyzTritonToLLVMConversionPatterns(typeConverter, patterns);
    target.addIllegalOp<triton::BitcastOp>();
    target.addIllegalOp<ptr::LoadOp, ptr::StoreOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
