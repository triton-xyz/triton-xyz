//===----------------------------------------------------------------------===//
// Throughout the conversion process, we convert !tt.ptr -> {!ptr.ptr or
// memref<*>}. This process leaves around unrealized_conversion_cast ops between
// these types. We want to remove these unrealized casts and use the proper
// conversion ops in the PtrDialect: to_memref or from_memref. To do this, we
// use a pattern that simplifies the chain of conversions by removing
// intermediate conversion cast ops. At the end, we are left with just pointer
// to memref or vice versa. We then convert the unrealized cast to to_memref or
// from_memref accordingly.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_RECONCILEPTRCASTS
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isOneToOneCast(UnrealizedConversionCastOp op) {
  return (op.getInputs().size() == 1 && op->getNumResults() == 1);
}

struct SimplifyUnrealizedCast
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  SimplifyUnrealizedCast(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto in = op.getInputs().front();

    if (auto unrealizedCast = in.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!isOneToOneCast(unrealizedCast)) {
        return failure();
      }

      auto prevInput = unrealizedCast.getInputs().front();
      auto newCast = UnrealizedConversionCastOp::create(
          rewriter, op->getLoc(), op->getResultTypes(), ValueRange{prevInput});

      rewriter.replaceOp(op, newCast);
      return success();
    }
    return failure();
  }
};

struct FromMemrefConverter
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  FromMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }

    auto input = op.getInputs().front();
    auto unrankedInput = dyn_cast<UnrankedMemRefType>(input.getType());
    auto output = op.getResult(0);
    auto outType = output.getType();

    if (unrankedInput && isa<triton::PointerType, ptr::PtrType>(outType)) {
      // from_memref only takes ranked memref, cast the unranked memref to
      // ranked memref first.
      auto rankedMemref =
          memref::CastOp::create(
              rewriter, op.getLoc(),
              MemRefType::get({1}, unrankedInput.getElementType(),
                              MemRefLayoutAttrInterface(),
                              unrankedInput.getMemorySpace()),
              input)
              .getResult();

      Attribute targetSpace;
      if (auto ptrType = dyn_cast<ptr::PtrType>(outType)) {
        targetSpace = ptrType.getMemorySpace();
      } else if (auto ptrType = dyn_cast<triton::PointerType>(outType)) {
        // Fallback for triton pointer type, assuming default/generic space
        // behavior match or we shouldn't be here. But keeping it safe. Convert
        // triton ptr space (int) to Attribute if needed? For now, let's assume
        // we are targeting ptr::PtrType as per pass design.
      }

      Value memrefForPtr = rankedMemref;
      if (unrankedInput.getMemorySpace() != targetSpace) {
        auto targetMemrefType =
            MemRefType::get({1}, unrankedInput.getElementType(),
                            MemRefLayoutAttrInterface(), targetSpace);
        memrefForPtr = memref::MemorySpaceCastOp::create(
            rewriter, op.getLoc(), targetMemrefType, rankedMemref);
      }

      auto memrefToPtr =
          ptr::ToPtrOp::create(rewriter, op->getLoc(), outType, memrefForPtr)
              .getResult();

      rewriter.replaceAllUsesWith(output, memrefToPtr);
      rewriter.eraseOp(op);

      return success();
    }

    return failure();
  }
};

struct ToMemrefConverter : public OpRewritePattern<UnrealizedConversionCastOp> {
  ToMemrefConverter(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<UnrealizedConversionCastOp>(context, benefit) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOneCast(op)) {
      return failure();
    }
    auto input = op.getInputs().front();
    auto inType = input.getType();
    auto output = op.getResult(0);
    auto outUnrankedMemrefType = dyn_cast<UnrankedMemRefType>(output.getType());
    if (isa<triton::PointerType, ptr::PtrType>(inType) &&
        outUnrankedMemrefType) {
      // to_memref can only cast to ranked static shape memref, we have to cast
      // the resulting memref back to unranked
      auto elemType = outUnrankedMemrefType.getElementType();
      Attribute inSpace;
      if (auto ptrType = dyn_cast<ptr::PtrType>(inType)) {
        inSpace = ptrType.getMemorySpace();
      }

      auto ptrToMemref =
          ptr::FromPtrOp::create(rewriter, op->getLoc(),
                                 MemRefType::get({1}, elemType,
                                                 MemRefLayoutAttrInterface(),
                                                 inSpace),
                                 input, nullptr)
              .getResult();

      Value castedMemref = ptrToMemref;
      Attribute outSpace = outUnrankedMemrefType.getMemorySpace();

      if (inSpace != outSpace) {
        auto targetMemrefType = MemRefType::get(
            {1}, elemType, MemRefLayoutAttrInterface(), outSpace);
        castedMemref = memref::MemorySpaceCastOp::create(
            rewriter, op.getLoc(), targetMemrefType, ptrToMemref);
      }

      auto newUnrankedMemref = memref::CastOp::create(
          rewriter, op.getLoc(), outUnrankedMemrefType, castedMemref);

      rewriter.replaceAllUsesWith(output, newUnrankedMemref);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class ReconcilePtrCastsPass
    : public triton::impl::ReconcilePtrCastsBase<ReconcilePtrCastsPass> {
  using Base = triton::impl::ReconcilePtrCastsBase<ReconcilePtrCastsPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns
        .add<SimplifyUnrealizedCast, FromMemrefConverter, ToMemrefConverter>(
            &getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
