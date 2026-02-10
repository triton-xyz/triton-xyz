#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
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

struct LoadedAddressInfo {
  int64_t rank;
  Type elementType;
  SmallVector<int64_t> shape;
};

static std::optional<LoadedAddressInfo> getLoadedAddressInfo(Type type) {
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementPtrType =
        dyn_cast<triton::PointerType>(ptrTensorType.getElementType());
    if (!elementPtrType) {
      return std::nullopt;
    }

    return LoadedAddressInfo{ptrTensorType.getRank(),
                             elementPtrType.getPointeeType(),
                             SmallVector<int64_t>(ptrTensorType.getShape())};
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (!pointeeTensorType) {
      return std::nullopt;
    }

    return LoadedAddressInfo{
        pointeeTensorType.getRank(), pointeeTensorType.getElementType(),
        SmallVector<int64_t>(pointeeTensorType.getShape())};
  }

  if (auto addrType = dyn_cast<tta::AddrType>(type)) {
    SmallVector<int64_t> shape(addrType.getRank(), ShapedType::kDynamic);
    return LoadedAddressInfo{addrType.getRank(), addrType.getElementType(),
                             std::move(shape)};
  }

  return std::nullopt;
}

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

struct ForIterArgInfo {
  scf::ForOp forOp;
  unsigned iterArgIndex;
};

struct LoopProgression {
  int64_t lowerBound;
  int64_t step;
};

static FailureOr<ForIterArgInfo> getForIterArgInfo(Value value);
static FailureOr<LoopProgression>
getConstantLoopProgression(ForIterArgInfo info);
static bool isAddressSeedResolvable(Value value);
static bool isSupportedLoopStepExpr(Value value, Value iterArg);
static Value stripAddressViewLikeChain(Value address);
static bool hasUnsupportedLoopCarriedAddr(Value ptr);

static LogicalResult emitTTAToMemrefError(Operation *op, StringRef reason) {
  op->emitError("tta-to-memref: ") << reason;
  return failure();
}

static FailureOr<Value>
castIndirectIndexToIndex(Value indirectIndex, Location loc,
                         ConversionPatternRewriter &rewriter);

static FailureOr<RankedTensorType>
getMerged1DTensorType(RankedTensorType lhsType, RankedTensorType rhsType,
                      Type elementType) {
  if (!lhsType || !rhsType || lhsType.getRank() != 1 ||
      rhsType.getRank() != 1 || lhsType.getElementType() != elementType ||
      rhsType.getElementType() != elementType) {
    return failure();
  }

  int64_t lhsDim = lhsType.getShape()[0];
  int64_t rhsDim = rhsType.getShape()[0];
  int64_t mergedDim = lhsDim;
  if (lhsDim != rhsDim) {
    if (ShapedType::isDynamic(lhsDim) || ShapedType::isDynamic(rhsDim)) {
      mergedDim = ShapedType::kDynamic;
    } else {
      return failure();
    }
  }

  return RankedTensorType::get({mergedDim}, elementType);
}

static FailureOr<Value> castTensorToType(Value value,
                                         RankedTensorType targetType,
                                         Location loc,
                                         ConversionPatternRewriter &rewriter) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || tensorType.getRank() != targetType.getRank() ||
      tensorType.getElementType() != targetType.getElementType()) {
    return failure();
  }

  if (tensorType == targetType) {
    return value;
  }

  if (failed(verifyCompatibleShape(tensorType, targetType))) {
    return failure();
  }

  return tensor::CastOp::create(rewriter, loc, targetType, value).getResult();
}

static FailureOr<Value>
mergeIndirectIndex(Value lhsIndirectIndex, Value rhsIndirectIndex, Location loc,
                   ConversionPatternRewriter &rewriter) {
  FailureOr<Value> lhsIndex =
      castIndirectIndexToIndex(lhsIndirectIndex, loc, rewriter);
  if (failed(lhsIndex)) {
    return failure();
  }

  FailureOr<Value> rhsIndex =
      castIndirectIndexToIndex(rhsIndirectIndex, loc, rewriter);
  if (failed(rhsIndex)) {
    return failure();
  }

  auto lhsType = dyn_cast<RankedTensorType>((*lhsIndex).getType());
  auto rhsType = dyn_cast<RankedTensorType>((*rhsIndex).getType());
  FailureOr<RankedTensorType> maybeMergedType =
      getMerged1DTensorType(lhsType, rhsType, rewriter.getIndexType());
  if (failed(maybeMergedType)) {
    return failure();
  }

  FailureOr<Value> lhsMerged =
      castTensorToType(*lhsIndex, *maybeMergedType, loc, rewriter);
  FailureOr<Value> rhsMerged =
      castTensorToType(*rhsIndex, *maybeMergedType, loc, rewriter);
  if (failed(lhsMerged) || failed(rhsMerged)) {
    return failure();
  }

  return arith::AddIOp::create(rewriter, loc, *lhsMerged, *rhsMerged)
      .getResult();
}

static FailureOr<AddressExpr>
collectAddressExpr(Value address, Location loc,
                   ConversionPatternRewriter &rewriter,
                   std::optional<StringRef> *failureReason = nullptr) {
  if (auto imported = address.getDefiningOp<tta::FromTTPtrOp>()) {
    address = imported.getSource();
  }

  if (auto maybeIterArgInfo = getForIterArgInfo(address);
      succeeded(maybeIterArgInfo)) {
    ForIterArgInfo iterArgInfo = *maybeIterArgInfo;
    FailureOr<LoopProgression> maybeProgression =
        getConstantLoopProgression(iterArgInfo);
    if (failed(maybeProgression)) {
      if (failureReason) {
        *failureReason = StringRef("non-constant loop progression");
      }
      return failure();
    }
    LoopProgression progression = *maybeProgression;

    FailureOr<AddressExpr> maybeInitExpr = collectAddressExpr(
        iterArgInfo.forOp.getInitArgs()[iterArgInfo.iterArgIndex], loc,
        rewriter, failureReason);
    if (failed(maybeInitExpr)) {
      return failure();
    }

    AddressExpr expr = *maybeInitExpr;
    if (expr.indirectIndex || expr.indirectMask || expr.indirectDim) {
      if (failureReason) {
        *failureReason =
            StringRef("unsupported loop-carried indirect recurrence");
      }
      return failure();
    }

    auto yieldOp =
        cast<scf::YieldOp>(iterArgInfo.forOp.getBody()->getTerminator());
    Value yielded = yieldOp.getResults()[iterArgInfo.iterArgIndex];
    int64_t rank = static_cast<int64_t>(expr.offsets.size());

    auto collectLoopStepOffsets =
        [&](auto &&self, Value value) -> FailureOr<SmallVector<OpFoldResult>> {
      if (value == address) {
        SmallVector<OpFoldResult> zeros;
        zeros.reserve(rank);
        for (int64_t i = 0; i < rank; ++i) {
          zeros.push_back(rewriter.getIndexAttr(0));
        }
        return zeros;
      }

      if (auto advance = value.getDefiningOp<tta::AdvanceOp>()) {
        auto maybeOffsets = self(self, advance.getAddress());
        if (failed(maybeOffsets)) {
          return failure();
        }
        auto deltas = advance.getMixedDeltas();
        if (static_cast<int64_t>(deltas.size()) != rank) {
          return failure();
        }

        SmallVector<OpFoldResult> composed = *maybeOffsets;
        for (auto [i, delta] : llvm::enumerate(deltas)) {
          composed[i] = addOFRs(composed[i], delta, loc, rewriter);
        }
        return composed;
      }

      if (auto reindex = value.getDefiningOp<tta::ReindexOp>()) {
        if (reindex.getIndirectIndex() || reindex.getMask()) {
          return failure();
        }

        auto maybeOffsets = self(self, reindex.getAddress());
        if (failed(maybeOffsets)) {
          return failure();
        }
        auto reindexOffsets = reindex.getMixedOffsets();
        if (static_cast<int64_t>(reindexOffsets.size()) != rank) {
          return failure();
        }

        SmallVector<OpFoldResult> composed = *maybeOffsets;
        for (auto [i, offset] : llvm::enumerate(reindexOffsets)) {
          composed[i] = addOFRs(composed[i], offset, loc, rewriter);
        }
        return composed;
      }

      return failure();
    };

    auto maybeStepOffsets =
        collectLoopStepOffsets(collectLoopStepOffsets, yielded);
    if (failed(maybeStepOffsets)) {
      if (failureReason) {
        *failureReason =
            StringRef("unsupported loop-carried address step expression");
      }
      return failure();
    }

    Value inductionVar = iterArgInfo.forOp.getInductionVar();
    Value inductionVarInt;
    IntegerType workIntType;

    if (inductionVar.getType().isIndex()) {
      workIntType = rewriter.getI64Type();
      inductionVarInt =
          arith::IndexCastOp::create(rewriter, loc, workIntType, inductionVar)
              .getResult();
    } else if (auto intType = dyn_cast<IntegerType>(inductionVar.getType())) {
      workIntType = intType;
      inductionVarInt = inductionVar;
    } else {
      if (failureReason) {
        *failureReason = StringRef("unsupported induction variable type");
      }
      return failure();
    }

    Value loopIndexInt = inductionVarInt;
    if (progression.lowerBound != 0) {
      Value lowerBound =
          arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getIntegerAttr(workIntType, progression.lowerBound))
              .getResult();
      loopIndexInt =
          arith::SubIOp::create(rewriter, loc, loopIndexInt, lowerBound)
              .getResult();
    }
    if (progression.step != 1) {
      Value step = arith::ConstantOp::create(
                       rewriter, loc,
                       rewriter.getIntegerAttr(workIntType, progression.step))
                       .getResult();
      loopIndexInt =
          arith::DivSIOp::create(rewriter, loc, loopIndexInt, step).getResult();
    }

    Value iterCount = arith::IndexCastOp::create(
                          rewriter, loc, rewriter.getIndexType(), loopIndexInt)
                          .getResult();

    for (auto [i, stepOffset] : llvm::enumerate(*maybeStepOffsets)) {
      OpFoldResult scaled = mulOFRs(stepOffset, iterCount, loc, rewriter);
      expr.offsets[i] = addOFRs(expr.offsets[i], scaled, loc, rewriter);
    }

    return expr;
  }

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
        collectAddressExpr(reindex.getAddress(), loc, rewriter, failureReason);
    if (failed(maybeExpr)) {
      return failure();
    }

    AddressExpr expr = *maybeExpr;
    auto reindexOffsets = reindex.getMixedOffsets();
    if (expr.offsets.size() != reindexOffsets.size()) {
      if (failureReason) {
        *failureReason = StringRef("reindex offsets rank mismatch");
      }
      return failure();
    }

    for (auto [i, off] : llvm::enumerate(reindexOffsets)) {
      expr.offsets[i] = addOFRs(expr.offsets[i], off, loc, rewriter);
    }

    if (Value indirect = reindex.getIndirectIndex()) {
      auto indirectDimAttr = reindex.getIndirectDimAttr();
      if (!indirectDimAttr) {
        if (failureReason) {
          *failureReason = StringRef("indirect_dim is missing");
        }
        return failure();
      }
      int32_t indirectDim = indirectDimAttr.getInt();

      if (expr.indirectIndex) {
        if (!expr.indirectDim.has_value() || *expr.indirectDim != indirectDim) {
          if (failureReason) {
            *failureReason = StringRef("mixed_indirect_dim in address chain");
          }
          return failure();
        }

        FailureOr<Value> maybeMergedIndirect =
            mergeIndirectIndex(expr.indirectIndex, indirect, loc, rewriter);
        if (failed(maybeMergedIndirect)) {
          if (failureReason) {
            *failureReason = StringRef("indirect_index is not mergeable");
          }
          return failure();
        }
        expr.indirectIndex = *maybeMergedIndirect;

        if (Value mask = reindex.getMask()) {
          if (expr.indirectMask) {
            auto lhsMaskType =
                dyn_cast<RankedTensorType>(expr.indirectMask.getType());
            auto rhsMaskType = dyn_cast<RankedTensorType>(mask.getType());
            FailureOr<RankedTensorType> maybeMergedMaskType =
                getMerged1DTensorType(lhsMaskType, rhsMaskType,
                                      rewriter.getI1Type());
            if (failed(maybeMergedMaskType)) {
              if (failureReason) {
                *failureReason = StringRef("indirect_mask shape mismatch");
              }
              return failure();
            }

            FailureOr<Value> lhsMergedMask = castTensorToType(
                expr.indirectMask, *maybeMergedMaskType, loc, rewriter);
            FailureOr<Value> rhsMergedMask =
                castTensorToType(mask, *maybeMergedMaskType, loc, rewriter);
            if (failed(lhsMergedMask) || failed(rhsMergedMask)) {
              if (failureReason) {
                *failureReason = StringRef("indirect_mask shape mismatch");
              }
              return failure();
            }

            expr.indirectMask =
                arith::AndIOp::create(rewriter, loc, *lhsMergedMask,
                                      *rhsMergedMask)
                    .getResult();
          } else {
            expr.indirectMask = mask;
          }
        }
      } else {
        expr.indirectIndex = indirect;
        expr.indirectDim = indirectDim;
        expr.indirectMask = reindex.getMask();
      }
    }

    return expr;
  }

  if (auto advance = address.getDefiningOp<tta::AdvanceOp>()) {
    FailureOr<AddressExpr> maybeExpr =
        collectAddressExpr(advance.getAddress(), loc, rewriter, failureReason);
    if (failed(maybeExpr)) {
      return failure();
    }

    AddressExpr expr = *maybeExpr;
    auto deltas = advance.getMixedDeltas();
    if (expr.offsets.size() != deltas.size()) {
      if (failureReason) {
        *failureReason = StringRef("advance deltas rank mismatch");
      }
      return failure();
    }

    for (auto [i, delta] : llvm::enumerate(deltas)) {
      expr.offsets[i] = addOFRs(expr.offsets[i], delta, loc, rewriter);
    }

    return expr;
  }

  if (failureReason) {
    *failureReason = StringRef("unsupported address chain");
  }
  return failure();
}

static FailureOr<ForIterArgInfo> getForIterArgInfo(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg) {
    return failure();
  }

  Block *block = blockArg.getOwner();
  if (!block) {
    return failure();
  }

  auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());
  if (!forOp || block != &forOp.getRegion().front()) {
    return failure();
  }

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber == 0 || argNumber > forOp.getInitArgs().size()) {
    return failure();
  }

  return ForIterArgInfo{forOp, argNumber - 1};
}

static FailureOr<LoopProgression>
getConstantLoopProgression(ForIterArgInfo info) {
  auto lowerBound = getConstantIntValue(info.forOp.getLowerBound());
  auto step = getConstantIntValue(info.forOp.getStep());
  if (!lowerBound || !step || *step <= 0) {
    return failure();
  }

  return LoopProgression{*lowerBound, *step};
}

static Value stripAddressViewLikeChain(Value address) {
  while (true) {
    if (auto imported = address.getDefiningOp<tta::FromTTPtrOp>()) {
      address = imported.getSource();
      continue;
    }
    if (auto reindex = address.getDefiningOp<tta::ReindexOp>()) {
      address = reindex.getAddress();
      continue;
    }
    if (auto advance = address.getDefiningOp<tta::AdvanceOp>()) {
      address = advance.getAddress();
      continue;
    }
    return address;
  }
}

static bool isAddressSeedResolvable(Value value) {
  while (true) {
    if (auto imported = value.getDefiningOp<tta::FromTTPtrOp>()) {
      value = imported.getSource();
      continue;
    }

    if (auto reindex = value.getDefiningOp<tta::ReindexOp>()) {
      if (reindex.getIndirectIndex() || reindex.getMask()) {
        return false;
      }
      value = reindex.getAddress();
      continue;
    }

    if (auto advance = value.getDefiningOp<tta::AdvanceOp>()) {
      value = advance.getAddress();
      continue;
    }

    return static_cast<bool>(value.getDefiningOp<tta::MakeAddrOp>());
  }
}

static bool isSupportedLoopStepExpr(Value value, Value iterArg) {
  if (value == iterArg) {
    return true;
  }

  if (auto advance = value.getDefiningOp<tta::AdvanceOp>()) {
    return isSupportedLoopStepExpr(advance.getAddress(), iterArg);
  }

  if (auto reindex = value.getDefiningOp<tta::ReindexOp>()) {
    if (reindex.getIndirectIndex() || reindex.getMask()) {
      return false;
    }
    return isSupportedLoopStepExpr(reindex.getAddress(), iterArg);
  }

  return false;
}

static bool hasUnsupportedLoopCarriedAddr(Value ptr) {
  Value root = stripAddressViewLikeChain(ptr);
  if (!isa<tta::AddrType>(root.getType())) {
    return false;
  }

  auto maybeIterArgInfo = getForIterArgInfo(root);
  if (failed(maybeIterArgInfo)) {
    return false;
  }

  ForIterArgInfo iterArgInfo = *maybeIterArgInfo;
  if (failed(getConstantLoopProgression(iterArgInfo))) {
    return true;
  }

  Value initArg = iterArgInfo.forOp.getInitArgs()[iterArgInfo.iterArgIndex];
  if (!isAddressSeedResolvable(initArg)) {
    return true;
  }

  auto yieldOp =
      cast<scf::YieldOp>(iterArgInfo.forOp.getBody()->getTerminator());
  Value yielded = yieldOp.getResults()[iterArgInfo.iterArgIndex];
  return !isSupportedLoopStepExpr(yielded, root);
}

static FailureOr<SmallVector<int64_t>>
resolveLoadedShape(const LoadedAddressInfo &loadedInfo,
                   const AddressExpr &expr) {
  if (loadedInfo.shape.size() != expr.sizes.size()) {
    return failure();
  }

  SmallVector<int64_t> shape(loadedInfo.shape.begin(), loadedInfo.shape.end());
  for (auto [index, dim] : llvm::enumerate(shape)) {
    if (ShapedType::isDynamic(dim)) {
      shape[index] = expr.sizes[index];
    }
  }

  return shape;
}

static Value makeIndexConstant(Location loc, int64_t v,
                               ConversionPatternRewriter &rewriter) {
  return arith::ConstantIndexOp::create(rewriter, loc, v).getResult();
}

static SmallVector<OpFoldResult>
getZeroOffsets(int64_t rank, ConversionPatternRewriter &rewriter) {
  SmallVector<OpFoldResult> offsets;
  offsets.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    offsets.push_back(rewriter.getIndexAttr(0));
  }
  return offsets;
}

static SmallVector<OpFoldResult>
getOneStrides(int64_t rank, ConversionPatternRewriter &rewriter) {
  SmallVector<OpFoldResult> strides;
  strides.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    strides.push_back(rewriter.getIndexAttr(1));
  }
  return strides;
}

static SmallVector<OpFoldResult>
getMixedStaticSizes(ArrayRef<int64_t> sizes,
                    ConversionPatternRewriter &rewriter) {
  SmallVector<OpFoldResult> mixedSizes;
  mixedSizes.reserve(sizes.size());
  for (int64_t size : sizes) {
    mixedSizes.push_back(rewriter.getIndexAttr(size));
  }
  return mixedSizes;
}

static SmallVector<OpFoldResult>
getMixedStridesForMemref(ArrayRef<int64_t> sizes,
                         ArrayRef<OpFoldResult> strides,
                         ConversionPatternRewriter &rewriter) {
  SmallVector<OpFoldResult> normalized;
  int64_t accumulate = 1;

  for (auto [size, stride] : llvm::reverse(llvm::zip(sizes, strides))) {
    auto strideInt = getIntAttr(stride);
    if (size == 1 && strideInt && strideInt.value() == 0) {
      normalized.push_back(rewriter.getIndexAttr(accumulate));
    } else {
      normalized.push_back(stride);
    }
    accumulate *= size;
  }

  std::reverse(normalized.begin(), normalized.end());
  return normalized;
}

static OpFoldResult
accumulateTargetOffset(Location loc, ArrayRef<OpFoldResult> offsets,
                       ConversionPatternRewriter &rewriter) {
  OpFoldResult targetOffset = rewriter.getIndexAttr(0);
  for (auto offset : offsets) {
    targetOffset = addOFRs(targetOffset, offset, loc, rewriter);
  }
  return targetOffset;
}

static OpFoldResult
accumulateTargetOffset(Location loc, ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> strides, int64_t gatherDim,
                       ConversionPatternRewriter &rewriter) {
  OpFoldResult targetOffset = rewriter.getIndexAttr(0);
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    OpFoldResult current = offset;
    if (static_cast<int64_t>(i) == gatherDim) {
      current = mulOFRs(current, strides[i], loc, rewriter);
    }
    targetOffset = addOFRs(targetOffset, current, loc, rewriter);
  }
  return targetOffset;
}

static FailureOr<Value> getBaseMemref(Value base, Type elementType,
                                      Location loc,
                                      ConversionPatternRewriter &rewriter) {
  if (isa<BaseMemRefType>(base.getType())) {
    return base;
  }

  if (auto castOp = base.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (!castOp.getInputs().empty() &&
        isa<BaseMemRefType>(castOp.getInputs().front().getType())) {
      return castOp.getInputs().front();
    }
  }

  if (isa<triton::PointerType>(base.getType())) {
    auto memrefType = UnrankedMemRefType::get(elementType, 0);
    return UnrealizedConversionCastOp::create(rewriter, loc, memrefType, base)
        .getResult(0);
  }

  return failure();
}

static memref::SubViewOp
createSubview(Value source, ArrayRef<OpFoldResult> offsets,
              ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
              Location loc, ConversionPatternRewriter &rewriter) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto resultType =
      memref::SubViewOp::inferResultType(sourceType, offsets, sizes, strides);
  return memref::SubViewOp::create(rewriter, loc, cast<MemRefType>(resultType),
                                   source, offsets, sizes, strides);
}

static tensor::ExtractSliceOp
createExtractSlice(Value source, ArrayRef<OpFoldResult> offsets,
                   ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                   Location loc, ConversionPatternRewriter &rewriter) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto resultType = tensor::ExtractSliceOp::inferResultType(sourceType, sizes);
  return tensor::ExtractSliceOp::create(rewriter, loc,
                                        cast<RankedTensorType>(resultType),
                                        source, offsets, sizes, strides);
}

static FailureOr<Value> createReinterpretCast(
    Value baseMemref, Type elementType, ArrayRef<int64_t> resultShape,
    ArrayRef<int64_t> sourceSizes, ArrayRef<OpFoldResult> sourceOffsets,
    ArrayRef<OpFoldResult> sourceStrides, std::optional<int64_t> gatherDim,
    Location loc, ConversionPatternRewriter &rewriter) {
  if (sourceSizes.size() != sourceOffsets.size() ||
      sourceSizes.size() != sourceStrides.size()) {
    return failure();
  }

  auto mixedStrides =
      getMixedStridesForMemref(sourceSizes, sourceStrides, rewriter);

  OpFoldResult targetOffset = rewriter.getIndexAttr(0);
  if (gatherDim.has_value()) {
    targetOffset = accumulateTargetOffset(loc, sourceOffsets, mixedStrides,
                                          *gatherDim, rewriter);
  } else {
    targetOffset = accumulateTargetOffset(loc, sourceOffsets, rewriter);
  }

  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  int64_t staticOffset =
      getIntAttr(targetOffset).value_or(ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(rewriter.getContext(), staticOffset,
                                       staticStrides);
  auto resultType = MemRefType::get(resultShape, elementType, layout);

  auto sizes = getMixedStaticSizes(resultShape, rewriter);

  auto castOp = memref::ReinterpretCastOp::create(
      rewriter, loc, resultType, baseMemref, targetOffset, sizes, mixedStrides);
  return castOp.getResult();
}

static FailureOr<Value> getIndirectSize(Value indirectIndex, Location loc,
                                        ConversionPatternRewriter &rewriter) {
  auto tensorType = dyn_cast<RankedTensorType>(indirectIndex.getType());
  if (!tensorType || tensorType.getRank() != 1) {
    return failure();
  }

  if (tensorType.isDynamicDim(0)) {
    return tensor::DimOp::create(rewriter, loc, indirectIndex, 0).getResult();
  }

  return makeIndexConstant(loc, tensorType.getDimSize(0), rewriter);
}

static FailureOr<Value>
castIndirectIndexToIndex(Value indirectIndex, Location loc,
                         ConversionPatternRewriter &rewriter) {
  auto tensorType = dyn_cast<RankedTensorType>(indirectIndex.getType());
  if (!tensorType || tensorType.getRank() != 1) {
    return failure();
  }

  if (tensorType.getElementType().isIndex()) {
    return indirectIndex;
  }

  auto indexTensorType =
      RankedTensorType::get(tensorType.getShape(), rewriter.getIndexType());
  return arith::IndexCastOp::create(rewriter, loc, indexTensorType,
                                    indirectIndex)
      .getResult();
}

static FailureOr<Value>
castAtomicOffsetToIndex(Value offset, Location loc,
                        ConversionPatternRewriter &rewriter) {
  if (offset.getType().isIndex()) {
    return offset;
  }

  if (offset.getType().isIntOrIndex()) {
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                      offset)
        .getResult();
  }

  return failure();
}

static std::optional<Value>
buildAtomicUpdateFromKind(ConversionPatternRewriter &rewriter, Location loc,
                          StringRef kind, Value current, Value value) {
  Type elementType = current.getType();
  if (isa<FloatType>(elementType)) {
    if (kind == "add" || kind == "fadd") {
      return arith::AddFOp::create(rewriter, loc, current, value).getResult();
    }
    if (kind == "max") {
      return arith::MaximumFOp::create(rewriter, loc, current, value)
          .getResult();
    }
    if (kind == "min") {
      return arith::MinimumFOp::create(rewriter, loc, current, value)
          .getResult();
    }
    if (kind == "xchg") {
      return value;
    }
    return std::nullopt;
  }

  if (isa<IntegerType>(elementType)) {
    if (kind == "add") {
      return arith::AddIOp::create(rewriter, loc, current, value).getResult();
    }
    if (kind == "and") {
      return arith::AndIOp::create(rewriter, loc, current, value).getResult();
    }
    if (kind == "or") {
      return arith::OrIOp::create(rewriter, loc, current, value).getResult();
    }
    if (kind == "xor") {
      return arith::XOrIOp::create(rewriter, loc, current, value).getResult();
    }
    if (kind == "max") {
      return arith::MaxSIOp::create(rewriter, loc, current, value).getResult();
    }
    if (kind == "min") {
      return arith::MinSIOp::create(rewriter, loc, current, value).getResult();
    }
    if (kind == "xchg") {
      return value;
    }
    return std::nullopt;
  }

  return std::nullopt;
}

static Value computeIndirectUpperBound(Value offsetSize,
                                       ArrayRef<OpFoldResult> maskDims,
                                       int64_t gatherDim, bool hasIndirectMask,
                                       Location loc,
                                       ConversionPatternRewriter &rewriter) {
  if (maskDims.empty()) {
    return offsetSize;
  }

  if (gatherDim < 0 || gatherDim >= static_cast<int64_t>(maskDims.size())) {
    return offsetSize;
  }

  OpFoldResult gatherMaskDim = maskDims[gatherDim];
  if (auto intAttr = getIntAttr(gatherMaskDim)) {
    int64_t dim = intAttr.value();
    if (dim == 0 && hasIndirectMask) {
      return offsetSize;
    }
    Value dimValue = makeIndexConstant(loc, dim, rewriter);
    return arith::MinSIOp::create(rewriter, loc, dimValue, offsetSize)
        .getResult();
  }

  Value dimValue = ofrToIndexValue(gatherMaskDim, loc, rewriter);
  return arith::MinSIOp::create(rewriter, loc, dimValue, offsetSize)
      .getResult();
}

static Value computeIndirectUpperBound(Value offsetSize,
                                       ArrayRef<OpFoldResult> maskDims,
                                       int64_t gatherDim, bool hasIndirectMask,
                                       int64_t offsetStaticSize, Location loc,
                                       ConversionPatternRewriter &rewriter) {
  if (maskDims.empty()) {
    return offsetSize;
  }

  if (gatherDim < 0 || gatherDim >= static_cast<int64_t>(maskDims.size())) {
    return offsetSize;
  }

  OpFoldResult gatherMaskDim = maskDims[gatherDim];
  if (auto intAttr = getIntAttr(gatherMaskDim)) {
    int64_t dim = intAttr.value();
    if (dim == 0 && hasIndirectMask) {
      return offsetSize;
    }
    return makeIndexConstant(loc, std::min<int64_t>(offsetStaticSize, dim),
                             rewriter);
  }

  Value dimValue = ofrToIndexValue(gatherMaskDim, loc, rewriter);
  return arith::MinSIOp::create(rewriter, loc, dimValue, offsetSize)
      .getResult();
}

static void maybeFillWithOther(Value alloc, Value other, Location loc,
                               ConversionPatternRewriter &rewriter) {
  if (!other) {
    return;
  }
  linalg::FillOp::create(rewriter, loc, ValueRange{other}, ValueRange{alloc});
}

struct ConvertTTALoadPattern : public OpConversionPattern<tta::LoadOp> {
  using OpConversionPattern<tta::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tta::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadedInfo = getLoadedAddressInfo(op.getPtr().getType());
    if (!loadedInfo) {
      return rewriter.notifyMatchFailure(op, "unsupported pointer-like type");
    }

    std::optional<StringRef> collectFailureReason;
    FailureOr<AddressExpr> maybeExpr = collectAddressExpr(
        adaptor.getPtr(), op.getLoc(), rewriter, &collectFailureReason);
    if (failed(maybeExpr)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    AddressExpr expr = *maybeExpr;

    auto maybeLoadedShape = resolveLoadedShape(*loadedInfo, expr);
    if (failed(maybeLoadedShape)) {
      return rewriter.notifyMatchFailure(op, "failed to resolve loaded shape");
    }
    SmallVector<int64_t> loadedShape = *maybeLoadedShape;

    auto maybeBaseMemref = getBaseMemref(expr.base, loadedInfo->elementType,
                                         op.getLoc(), rewriter);
    if (failed(maybeBaseMemref)) {
      return rewriter.notifyMatchFailure(op, "failed to get base memref");
    }

    auto allocType = MemRefType::get(loadedShape, loadedInfo->elementType);
    Value alloc =
        memref::AllocOp::create(rewriter, op.getLoc(), allocType).getResult();

    auto ones = getOneStrides(loadedInfo->rank, rewriter);
    auto zeros = getZeroOffsets(loadedInfo->rank, rewriter);
    SmallVector<OpFoldResult> maskDims = op.getMixedMaskDims();

    if (!expr.indirectIndex) {
      auto maybeSrc = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, loadedShape, expr.sizes,
          expr.offsets, expr.strides,
          /*gatherDim=*/std::nullopt, op.getLoc(), rewriter);
      if (failed(maybeSrc)) {
        return rewriter.notifyMatchFailure(op,
                                           "failed to create source memref");
      }

      maybeFillWithOther(alloc, op.getOther(), op.getLoc(), rewriter);

      if (maskDims.empty()) {
        memref::CopyOp::create(rewriter, op.getLoc(), *maybeSrc, alloc);
      } else {
        auto srcSubview = createSubview(*maybeSrc, zeros, maskDims, ones,
                                        op.getLoc(), rewriter);
        auto dstSubview =
            createSubview(alloc, zeros, maskDims, ones, op.getLoc(), rewriter);
        memref::CopyOp::create(rewriter, op.getLoc(), srcSubview, dstSubview);
      }

      Value resultTensor = bufferization::ToTensorOp::create(
                               rewriter, op.getLoc(),
                               cast<RankedTensorType>(op.getType()), alloc,
                               /*restrict=*/true,
                               /*writable=*/true)
                               .getResult();
      rewriter.replaceOp(op, resultTensor);
      return success();
    }

    if (!expr.indirectDim.has_value()) {
      return rewriter.notifyMatchFailure(op, "indirect dim is missing");
    }
    int64_t gatherDim = *expr.indirectDim;
    if (gatherDim < 0 || gatherDim >= loadedInfo->rank) {
      return rewriter.notifyMatchFailure(op, "indirect dim out of bounds");
    }
    if (!expr.order.empty()) {
      return emitTTAToMemrefError(
          op.getOperation(),
          "indirect reindex on block pointer is unsupported");
    }
    OpFoldResult gatherBaseOffset = expr.offsets[gatherDim];

    auto maybeIndirectIndex =
        castIndirectIndexToIndex(expr.indirectIndex, op.getLoc(), rewriter);
    if (failed(maybeIndirectIndex)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index");
    }

    auto maybeOffsetSize =
        getIndirectSize(*maybeIndirectIndex, op.getLoc(), rewriter);
    if (failed(maybeOffsetSize)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index size");
    }

    if (maskDims.empty() && (op.getOther() || expr.indirectMask)) {
      maskDims = getMixedStaticSizes(loadedShape, rewriter);
    }

    SmallVector<int64_t> sliceShape(loadedShape.begin(), loadedShape.end());
    sliceShape[gatherDim] = 1;
    SmallVector<OpFoldResult> sliceSizes =
        getMixedStaticSizes(sliceShape, rewriter);
    if (!maskDims.empty()) {
      if (static_cast<int64_t>(maskDims.size()) != loadedInfo->rank) {
        return rewriter.notifyMatchFailure(op, "mask rank mismatch");
      }
      sliceSizes = maskDims;
      sliceSizes[gatherDim] = rewriter.getIndexAttr(1);
    }

    maybeFillWithOther(alloc, op.getOther(), op.getLoc(), rewriter);

    Value lowerBound = makeIndexConstant(op.getLoc(), 0, rewriter);

    auto indirectIndexType =
        cast<RankedTensorType>((*maybeIndirectIndex).getType());
    Value upperBound;
    if (indirectIndexType.isDynamicDim(0)) {
      upperBound = computeIndirectUpperBound(
          *maybeOffsetSize, maskDims, gatherDim,
          static_cast<bool>(expr.indirectMask), op.getLoc(), rewriter);
    } else {
      upperBound = computeIndirectUpperBound(
          *maybeOffsetSize, maskDims, gatherDim,
          static_cast<bool>(expr.indirectMask), indirectIndexType.getShape()[0],
          op.getLoc(), rewriter);
    }

    Value step = makeIndexConstant(op.getLoc(), 1, rewriter);
    auto loop =
        scf::ForOp::create(rewriter, op.getLoc(), lowerBound, upperBound, step);

    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();

    auto copyBody = [&](ConversionPatternRewriter &rw) -> LogicalResult {
      Value gatherOffset = tensor::ExtractOp::create(
          rw, op.getLoc(), *maybeIndirectIndex, ValueRange{iv});
      if (!hasConstZero(gatherBaseOffset)) {
        Value gatherBaseOffsetValue =
            ofrToIndexValue(gatherBaseOffset, op.getLoc(), rw);
        gatherOffset = arith::AddIOp::create(rw, op.getLoc(), gatherOffset,
                                             gatherBaseOffsetValue)
                           .getResult();
      }

      SmallVector<OpFoldResult> gatheredOffsets(expr.offsets.begin(),
                                                expr.offsets.end());
      gatheredOffsets[gatherDim] = gatherOffset;

      auto maybeSrc = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, sliceShape, expr.sizes,
          gatheredOffsets, expr.strides, gatherDim, op.getLoc(), rw);
      if (failed(maybeSrc)) {
        return failure();
      }

      auto allocOffsets = getZeroOffsets(loadedInfo->rank, rw);
      allocOffsets[gatherDim] = iv;

      auto dstSubview =
          createSubview(alloc, allocOffsets, sliceSizes, ones, op.getLoc(), rw);

      if (maskDims.empty()) {
        memref::CopyOp::create(rw, op.getLoc(), *maybeSrc, dstSubview);
      } else {
        auto srcSubview =
            createSubview(*maybeSrc, zeros, sliceSizes, ones, op.getLoc(), rw);
        memref::CopyOp::create(rw, op.getLoc(), srcSubview, dstSubview);
      }

      return success();
    };

    if (expr.indirectMask) {
      Value maskValue = tensor::ExtractOp::create(
          rewriter, op.getLoc(), expr.indirectMask, ValueRange{iv});
      auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      if (failed(copyBody(rewriter))) {
        return rewriter.notifyMatchFailure(
            op, "failed to lower indirect load body");
      }
      rewriter.setInsertionPointAfter(ifOp);
    } else {
      if (failed(copyBody(rewriter))) {
        return rewriter.notifyMatchFailure(
            op, "failed to lower indirect load body");
      }
    }

    rewriter.setInsertionPointAfter(loop);

    Value resultTensor =
        bufferization::ToTensorOp::create(
            rewriter, op.getLoc(), cast<RankedTensorType>(op.getType()), alloc,
            /*restrict=*/true,
            /*writable=*/true)
            .getResult();
    rewriter.replaceOp(op, resultTensor);
    return success();
  }
};

struct ConvertTTAStorePattern : public OpConversionPattern<tta::StoreOp> {
  using OpConversionPattern<tta::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tta::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadedInfo = getLoadedAddressInfo(op.getPtr().getType());
    if (!loadedInfo) {
      return rewriter.notifyMatchFailure(op, "unsupported pointer-like type");
    }

    std::optional<StringRef> collectFailureReason;
    FailureOr<AddressExpr> maybeExpr = collectAddressExpr(
        adaptor.getPtr(), op.getLoc(), rewriter, &collectFailureReason);
    if (failed(maybeExpr)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    AddressExpr expr = *maybeExpr;

    auto maybeLoadedShape = resolveLoadedShape(*loadedInfo, expr);
    if (failed(maybeLoadedShape)) {
      return rewriter.notifyMatchFailure(op, "failed to resolve loaded shape");
    }
    SmallVector<int64_t> loadedShape = *maybeLoadedShape;

    auto maybeBaseMemref = getBaseMemref(expr.base, loadedInfo->elementType,
                                         op.getLoc(), rewriter);
    if (failed(maybeBaseMemref)) {
      return rewriter.notifyMatchFailure(op, "failed to get base memref");
    }

    auto ones = getOneStrides(loadedInfo->rank, rewriter);
    auto zeros = getZeroOffsets(loadedInfo->rank, rewriter);
    SmallVector<OpFoldResult> maskDims = op.getMixedMaskDims();

    if (!expr.indirectIndex) {
      auto maybeDst = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, loadedShape, expr.sizes,
          expr.offsets, expr.strides,
          /*gatherDim=*/std::nullopt, op.getLoc(), rewriter);
      if (failed(maybeDst)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create destination memref");
      }

      if (maskDims.empty()) {
        auto storeOp = bufferization::MaterializeInDestinationOp::create(
            rewriter, op.getLoc(), adaptor.getValue(), *maybeDst);
        storeOp.setWritable(true);
      } else {
        auto srcSlice = createExtractSlice(adaptor.getValue(), zeros, maskDims,
                                           ones, op.getLoc(), rewriter);
        auto dstSubview = createSubview(*maybeDst, zeros, maskDims, ones,
                                        op.getLoc(), rewriter);
        auto storeOp = bufferization::MaterializeInDestinationOp::create(
            rewriter, op.getLoc(), srcSlice.getResult(),
            dstSubview.getResult());
        storeOp.setWritable(true);
      }

      rewriter.eraseOp(op);
      return success();
    }

    if (!expr.indirectDim.has_value()) {
      return rewriter.notifyMatchFailure(op, "indirect dim is missing");
    }
    int64_t gatherDim = *expr.indirectDim;
    if (gatherDim < 0 || gatherDim >= loadedInfo->rank) {
      return rewriter.notifyMatchFailure(op, "indirect dim out of bounds");
    }
    if (!expr.order.empty()) {
      return emitTTAToMemrefError(
          op.getOperation(),
          "indirect reindex on block pointer is unsupported");
    }
    OpFoldResult gatherBaseOffset = expr.offsets[gatherDim];

    auto maybeIndirectIndex =
        castIndirectIndexToIndex(expr.indirectIndex, op.getLoc(), rewriter);
    if (failed(maybeIndirectIndex)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index");
    }

    auto maybeOffsetSize =
        getIndirectSize(*maybeIndirectIndex, op.getLoc(), rewriter);
    if (failed(maybeOffsetSize)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index size");
    }

    if (maskDims.empty() && expr.indirectMask) {
      maskDims = getMixedStaticSizes(loadedShape, rewriter);
    }

    SmallVector<int64_t> sliceShape(loadedShape.begin(), loadedShape.end());
    sliceShape[gatherDim] = 1;
    SmallVector<OpFoldResult> sliceSizes =
        getMixedStaticSizes(sliceShape, rewriter);
    if (!maskDims.empty()) {
      if (static_cast<int64_t>(maskDims.size()) != loadedInfo->rank) {
        return rewriter.notifyMatchFailure(op, "mask rank mismatch");
      }
      sliceSizes = maskDims;
      sliceSizes[gatherDim] = rewriter.getIndexAttr(1);
    }

    Value lowerBound = makeIndexConstant(op.getLoc(), 0, rewriter);

    auto indirectIndexType =
        cast<RankedTensorType>((*maybeIndirectIndex).getType());
    Value upperBound;
    if (indirectIndexType.isDynamicDim(0)) {
      upperBound = computeIndirectUpperBound(
          *maybeOffsetSize, maskDims, gatherDim,
          static_cast<bool>(expr.indirectMask), op.getLoc(), rewriter);
    } else {
      upperBound = computeIndirectUpperBound(
          *maybeOffsetSize, maskDims, gatherDim,
          static_cast<bool>(expr.indirectMask), indirectIndexType.getShape()[0],
          op.getLoc(), rewriter);
    }

    Value step = makeIndexConstant(op.getLoc(), 1, rewriter);
    auto loop =
        scf::ForOp::create(rewriter, op.getLoc(), lowerBound, upperBound, step);

    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();

    auto storeBody = [&](ConversionPatternRewriter &rw) -> LogicalResult {
      Value gatherOffset = tensor::ExtractOp::create(
          rw, op.getLoc(), *maybeIndirectIndex, ValueRange{iv});
      if (!hasConstZero(gatherBaseOffset)) {
        Value gatherBaseOffsetValue =
            ofrToIndexValue(gatherBaseOffset, op.getLoc(), rw);
        gatherOffset = arith::AddIOp::create(rw, op.getLoc(), gatherOffset,
                                             gatherBaseOffsetValue)
                           .getResult();
      }

      SmallVector<OpFoldResult> gatheredOffsets(expr.offsets.begin(),
                                                expr.offsets.end());
      gatheredOffsets[gatherDim] = gatherOffset;

      auto maybeDst = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, sliceShape, expr.sizes,
          gatheredOffsets, expr.strides, gatherDim, op.getLoc(), rw);
      if (failed(maybeDst)) {
        return failure();
      }

      auto valueOffsets = getZeroOffsets(loadedInfo->rank, rw);
      valueOffsets[gatherDim] = iv;
      auto srcSlice = createExtractSlice(adaptor.getValue(), valueOffsets,
                                         sliceSizes, ones, op.getLoc(), rw);

      Value destination = *maybeDst;
      if (!maskDims.empty()) {
        destination =
            createSubview(*maybeDst, zeros, sliceSizes, ones, op.getLoc(), rw)
                .getResult();
      }

      auto storeOp = bufferization::MaterializeInDestinationOp::create(
          rw, op.getLoc(), srcSlice.getResult(), destination);
      storeOp.setWritable(true);
      return success();
    };

    if (expr.indirectMask) {
      Value maskValue = tensor::ExtractOp::create(
          rewriter, op.getLoc(), expr.indirectMask, ValueRange{iv});
      auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      if (failed(storeBody(rewriter))) {
        return rewriter.notifyMatchFailure(
            op, "failed to lower indirect store body");
      }
      rewriter.setInsertionPointAfter(ifOp);
    } else {
      if (failed(storeBody(rewriter))) {
        return rewriter.notifyMatchFailure(
            op, "failed to lower indirect store body");
      }
    }

    rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTTAAtomicPattern : public OpConversionPattern<tta::AtomicOp> {
  using OpConversionPattern<tta::AtomicOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tta::AtomicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tensor tta.atomic is unsupported");
    }

    if (!isa<tta::AddrType>(op.getPtr().getType())) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic pointer must be scalar addr");
    }

    std::optional<StringRef> collectFailureReason;
    FailureOr<AddressExpr> maybeExpr = collectAddressExpr(
        adaptor.getPtr(), op.getLoc(), rewriter, &collectFailureReason);
    if (failed(maybeExpr)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    AddressExpr expr = *maybeExpr;

    if (expr.indirectIndex || expr.indirectMask || expr.indirectDim) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "indirect tta.atomic is unsupported");
    }
    if (!expr.order.empty()) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "block pointer tta.atomic is unsupported");
    }
    if (expr.offsets.size() != 1 || expr.strides.size() != 1) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic pointer must have rank 1");
    }

    std::optional<int64_t> stride = getIntAttr(expr.strides[0]);
    if (!stride || *stride != 1) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic requires unit stride");
    }

    auto maybeBaseMemref = getBaseMemref(expr.base, op.getValue().getType(),
                                         op.getLoc(), rewriter);
    if (failed(maybeBaseMemref)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "failed to get base memref for tta.atomic");
    }

    auto maybeOffset =
        castAtomicOffsetToIndex(adaptor.getOffset(), op.getLoc(), rewriter);
    if (failed(maybeOffset)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic offset must be int/index scalar");
    }

    Value totalOffset = *maybeOffset;
    if (!hasConstZero(expr.offsets[0])) {
      Value baseOffset =
          ofrToIndexValue(expr.offsets[0], op.getLoc(), rewriter);
      totalOffset =
          arith::AddIOp::create(rewriter, op.getLoc(), baseOffset, totalOffset)
              .getResult();
    }

    auto rankedType =
        MemRefType::get({ShapedType::kDynamic}, op.getValue().getType());
    Value rankedMemref = memref::CastOp::create(rewriter, op.getLoc(),
                                                rankedType, *maybeBaseMemref)
                             .getResult();

    auto generic = memref::GenericAtomicRMWOp::create(
        rewriter, op.getLoc(), rankedMemref, totalOffset);
    Block &body = generic.getRegion().front();
    rewriter.setInsertionPointToStart(&body);

    Value current = body.getArgument(0);
    auto maybeUpdated = buildAtomicUpdateFromKind(
        rewriter, op.getLoc(), adaptor.getKindAttr().getValue(), current,
        adaptor.getValue());
    if (!maybeUpdated.has_value()) {
      rewriter.eraseOp(generic);
      std::string message = "atomic kind is unsupported: ";
      message += adaptor.getKindAttr().getValue().str();
      return emitTTAToMemrefError(op.getOperation(), message);
    }

    Value finalValue = *maybeUpdated;
    if (Value mask = adaptor.getMask()) {
      if (!mask.getType().isInteger(1)) {
        rewriter.eraseOp(generic);
        return emitTTAToMemrefError(op.getOperation(),
                                    "tta.atomic mask must be scalar i1");
      }
      finalValue = arith::SelectOp::create(rewriter, op.getLoc(), mask,
                                           finalValue, current)
                       .getResult();
    }

    memref::AtomicYieldOp::create(rewriter, op.getLoc(), finalValue);
    rewriter.replaceOp(op, generic.getResult());
    return success();
  }
};

struct ConvertTTAAtomicCASPattern
    : public OpConversionPattern<tta::AtomicCASOp> {
  using OpConversionPattern<tta::AtomicCASOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tta::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tensor tta.atomic_cas is unsupported");
    }

    if (!isa<triton::PointerType>(op.getPtr().getType())) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic_cas pointer must be scalar ptr");
    }

    auto maybeBaseMemref = getBaseMemref(
        adaptor.getPtr(), op.getValue().getType(), op.getLoc(), rewriter);
    if (failed(maybeBaseMemref)) {
      return emitTTAToMemrefError(
          op.getOperation(), "failed to get base memref for tta.atomic_cas");
    }

    auto maybeOffset =
        castAtomicOffsetToIndex(adaptor.getOffset(), op.getLoc(), rewriter);
    if (failed(maybeOffset)) {
      return emitTTAToMemrefError(
          op.getOperation(), "tta.atomic_cas offset must be int/index scalar");
    }

    auto rankedType =
        MemRefType::get({ShapedType::kDynamic}, op.getValue().getType());
    Value rankedMemref = memref::CastOp::create(rewriter, op.getLoc(),
                                                rankedType, *maybeBaseMemref)
                             .getResult();

    auto generic = memref::GenericAtomicRMWOp::create(
        rewriter, op.getLoc(), rankedMemref, *maybeOffset);
    Block &body = generic.getRegion().front();
    rewriter.setInsertionPointToStart(&body);

    Value current = body.getArgument(0);
    Value equal;
    if (isa<FloatType>(current.getType())) {
      equal = arith::CmpFOp::create(rewriter, op.getLoc(),
                                    arith::CmpFPredicate::OEQ, current,
                                    adaptor.getCompare())
                  .getResult();
    } else if (isa<IntegerType>(current.getType())) {
      equal =
          arith::CmpIOp::create(rewriter, op.getLoc(), arith::CmpIPredicate::eq,
                                current, adaptor.getCompare())
              .getResult();
    } else {
      rewriter.eraseOp(generic);
      return emitTTAToMemrefError(
          op.getOperation(),
          "tta.atomic_cas only supports integer or floating-point values");
    }

    Value finalValue = arith::SelectOp::create(rewriter, op.getLoc(), equal,
                                               adaptor.getValue(), current)
                           .getResult();
    memref::AtomicYieldOp::create(rewriter, op.getLoc(), finalValue);
    rewriter.replaceOp(op, generic.getResult());
    return success();
  }
};

class TTAToMemrefPass : public triton::impl::TTAToMemrefBase<TTAToMemrefPass> {
  using Base = triton::impl::TTAToMemrefBase<TTAToMemrefPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();

    bool hasUnsupportedControlFlow = false;
    moduleOp.walk([&](tta::LoadOp op) {
      if (!hasUnsupportedLoopCarriedAddr(op.getPtr())) {
        return;
      }
      op.emitOpError("unsupported loop-carried !tta.addr recurrence in "
                     "scf.for iter_args");
      hasUnsupportedControlFlow = true;
    });
    moduleOp.walk([&](tta::StoreOp op) {
      if (!hasUnsupportedLoopCarriedAddr(op.getPtr())) {
        return;
      }
      op.emitOpError("unsupported loop-carried !tta.addr recurrence in "
                     "scf.for iter_args");
      hasUnsupportedControlFlow = true;
    });
    if (hasUnsupportedControlFlow) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect, tta::TritonAddressDialect>();

    target.addIllegalOp<tta::LoadOp, tta::StoreOp, tta::AtomicOp,
                        tta::AtomicCASOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    patterns.add<ConvertTTALoadPattern, ConvertTTAStorePattern,
                 ConvertTTAAtomicPattern, ConvertTTAAtomicCASPattern>(
        typeConverter, patterns.getContext());

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
        if (isa<tta::AdvanceOp, tta::ReindexOp, tta::MakeAddrOp,
                tta::FromTTPtrOp>(op)) {
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
