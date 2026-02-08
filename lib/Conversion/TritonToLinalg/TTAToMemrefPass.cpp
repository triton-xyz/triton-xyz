#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

#define DEBUG_TYPE "tta-to-memref"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TTATOMEMREF
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) -> Type {
      if (isa<RankedTensorType>(ptrType.getPointeeType())) {
        return ptrType;
      }
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    });
  }
};

struct AddressExpr {
  Value base;
  SmallVector<int64_t> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> shape;
  SmallVector<int32_t> order;
  Value indirectIndex;
  Value indirectMask;
  std::optional<int32_t> indirectDim;
};

static FailureOr<AddressExpr>
collectAddressExpr(Value address, Location loc,
                   ConversionPatternRewriter &rewriter) {
  if (auto makeAddr = address.getDefiningOp<tta::MakeAddrOp>()) {
    AddressExpr expr;
    expr.base = makeAddr.getBase();
    expr.sizes = llvm::to_vector(makeAddr.getSizes());
    expr.strides = makeAddr.getMixedStrides();
    expr.offsets = makeAddr.getMixedOffsets();
    expr.shape = makeAddr.getMixedShape();
    expr.order = llvm::to_vector(makeAddr.getOrder());
    return expr;
  }

  if (auto reindex = address.getDefiningOp<tta::ReindexOp>()) {
    FailureOr<AddressExpr> maybeExpr =
        collectAddressExpr(reindex.getAddress(), loc, rewriter);
    if (failed(maybeExpr)) {
      return failure();
    }

    AddressExpr expr = *maybeExpr;
    auto reindexOffsets = reindex.getMixedOffsets();
    if (expr.offsets.size() != reindexOffsets.size()) {
      return failure();
    }

    for (auto [i, off] : llvm::enumerate(reindexOffsets)) {
      expr.offsets[i] = addOFRs(expr.offsets[i], off, loc, rewriter);
    }

    if (Value indirect = reindex.getIndirectIndex()) {
      if (expr.indirectIndex) {
        return failure();
      }
      auto indirectDimAttr = reindex.getIndirectDimAttr();
      if (!indirectDimAttr) {
        return failure();
      }
      expr.indirectIndex = indirect;
      expr.indirectDim = indirectDimAttr.getInt();
      expr.indirectMask = reindex.getMask();
    }

    return expr;
  }

  if (auto advance = address.getDefiningOp<tta::AdvanceOp>()) {
    FailureOr<AddressExpr> maybeExpr =
        collectAddressExpr(advance.getAddress(), loc, rewriter);
    if (failed(maybeExpr)) {
      return failure();
    }

    AddressExpr expr = *maybeExpr;
    auto deltas = advance.getMixedDeltas();
    if (expr.offsets.size() != deltas.size()) {
      return failure();
    }

    for (auto [i, delta] : llvm::enumerate(deltas)) {
      expr.offsets[i] = addOFRs(expr.offsets[i], delta, loc, rewriter);
    }

    return expr;
  }

  return failure();
}

static FailureOr<Value>
buildTTSAddressFromTTA(Value address, Location loc,
                       ConversionPatternRewriter &rewriter) {
  if (address.getDefiningOp<tts::MakeTensorPtrOp>() ||
      address.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>() ||
      !address.getDefiningOp()) {
    return address;
  }

  FailureOr<AddressExpr> maybeExpr = collectAddressExpr(address, loc, rewriter);
  if (failed(maybeExpr)) {
    return failure();
  }

  AddressExpr expr = *maybeExpr;

  if (expr.indirectIndex) {
    if (!expr.indirectDim.has_value()) {
      return failure();
    }
    if (!expr.order.empty()) {
      return failure();
    }

    int32_t gatherDim = *expr.indirectDim;
    if (gatherDim < 0 ||
        static_cast<size_t>(gatherDim) >= expr.offsets.size()) {
      return failure();
    }

    if (!hasConstZero(expr.offsets[gatherDim])) {
      return failure();
    }

    if (expr.indirectMask) {
      auto makeTptr = tts::MakeGatherScatterTensorPtrOp::create(
          rewriter, loc, expr.base, expr.indirectIndex, expr.indirectMask,
          gatherDim, expr.sizes, expr.strides, expr.offsets);
      return makeTptr.getResult();
    }

    auto makeTptr = tts::MakeGatherScatterTensorPtrOp::create(
        rewriter, loc, expr.base, expr.indirectIndex, gatherDim, expr.sizes,
        expr.strides, expr.offsets);
    return makeTptr.getResult();
  }

  auto makeTptr = tts::MakeTensorPtrOp::create(
      rewriter, loc, expr.base, expr.sizes, expr.strides, expr.offsets,
      expr.shape, expr.order);
  return makeTptr.getResult();
}

static FailureOr<SmallVector<OpFoldResult>>
computeLoadStoreMaskDims(Value ptr, ArrayRef<OpFoldResult> originalMaskDims,
                         bool requireRankMask,
                         ConversionPatternRewriter &rewriter) {
  SmallVector<OpFoldResult> dims(originalMaskDims.begin(),
                                 originalMaskDims.end());

  auto gatherPtr = ptr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>();
  if (!gatherPtr) {
    return dims;
  }

  int64_t rank = gatherPtr.getSizes().size();
  int64_t gatherDim = gatherPtr.getGatherScatterDim();
  bool hasGatherMask = static_cast<bool>(gatherPtr.getGatherScatterMask());

  if (dims.empty() && (requireRankMask || hasGatherMask)) {
    dims.reserve(rank);
    for (int64_t size : gatherPtr.getSizes()) {
      dims.push_back(rewriter.getIndexAttr(size));
    }
  }

  if (!hasGatherMask) {
    return dims;
  }

  if (static_cast<int64_t>(dims.size()) != rank || gatherDim < 0 ||
      gatherDim >= rank) {
    return failure();
  }

  dims[gatherDim] = rewriter.getIndexAttr(0);
  return dims;
}

struct ConvertTTALoadPattern : public OpConversionPattern<tta::LoadOp> {
  using OpConversionPattern<tta::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tta::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loweredPtr =
        buildTTSAddressFromTTA(adaptor.getPtr(), op.getLoc(), rewriter);
    if (failed(loweredPtr)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to lower tta address chain");
    }

    auto maybeMaskDims =
        computeLoadStoreMaskDims(*loweredPtr, op.getMixedMaskDims(),
                                 static_cast<bool>(op.getOther()), rewriter);
    if (failed(maybeMaskDims)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize mask dims for indirect reindex load");
    }

    auto load = tts::LoadOp::create(rewriter, op.getLoc(), *loweredPtr,
                                    *maybeMaskDims, op.getOther());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct ConvertTTAStorePattern : public OpConversionPattern<tta::StoreOp> {
  using OpConversionPattern<tta::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tta::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loweredPtr =
        buildTTSAddressFromTTA(adaptor.getPtr(), op.getLoc(), rewriter);
    if (failed(loweredPtr)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to lower tta address chain");
    }

    auto maybeMaskDims =
        computeLoadStoreMaskDims(*loweredPtr, op.getMixedMaskDims(),
                                 /*requireRankMask=*/false, rewriter);
    if (failed(maybeMaskDims)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize mask dims for indirect reindex store");
    }

    rewriter.replaceOpWithNewOp<tts::StoreOp>(
        op, *loweredPtr, adaptor.getValue(), *maybeMaskDims);
    return success();
  }
};

class TTAToMemrefPass : public triton::impl::TTAToMemrefBase<TTAToMemrefPass> {
  using Base = triton::impl::TTAToMemrefBase<TTAToMemrefPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect, tta::TritonAddressDialect,
        tts::TritonStructuredDialect>();

    target.addIllegalOp<tta::LoadOp, tta::StoreOp, tts::LoadOp, tts::StoreOp,
                        tts::MakeTensorPtrOp,
                        tts::MakeGatherScatterTensorPtrOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    patterns.add<ConvertTTALoadPattern, ConvertTTAStorePattern>(
        typeConverter, patterns.getContext());

    triton::populateStructuredToMemrefConversionPatterns(patterns,
                                                         typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> deadOps;
      moduleOp.walk([&](Operation *op) {
        if (!op->use_empty()) {
          return;
        }
        if (isa<tta::AdvanceOp, tta::ReindexOp, tta::MakeAddrOp>(op)) {
          deadOps.push_back(op);
        }
      });

      for (Operation *op : deadOps) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace
