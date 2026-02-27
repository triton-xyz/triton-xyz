#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
#include "triton-shared/Analysis/AnalysisAddress.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h" // IWYU pragma: keep
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
#include "triton-shared/Conversion/TritonToLinalgTTA/Passes.h.inc"
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

using AddressDescriptor = mlir::triton::address::AddressDescriptor;
using AddressFeatures = mlir::triton::address::AddressFeatures;
using DimRule = mlir::triton::address::DimRule;
using LayoutKind = mlir::triton::address::LayoutKind;

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

struct IndirectDimInfo {
  int64_t dim;
  Value index;
  Value mask;
};

struct IndirectInfo {
  SmallVector<IndirectDimInfo> dims;
};

enum class AddressLoweringPath {
  Direct,
  Indirect,
  WrapAware,
};

struct ForIterArgInfo {
  scf::ForOp forOp;
  unsigned iterArgIndex;
};

struct LoopProgression {
  int64_t lowerBound;
  int64_t step;
};

struct AddressStepInfo {
  SmallVector<OpFoldResult> offsets;
  SmallVector<IndirectDimInfo> indirects;
};

static FailureOr<ForIterArgInfo> getForIterArgInfo(Value value);
static FailureOr<LoopProgression>
getConstantLoopProgression(ForIterArgInfo info);
static FailureOr<SmallVector<OpFoldResult>>
collectAddressOffsetDeltas(Value value, Value root, int64_t rank, Location loc,
                           ConversionPatternRewriter &rewriter);
static FailureOr<AddressStepInfo>
collectAddressStepInfo(Value value, Value root, int64_t rank, Location loc,
                       ConversionPatternRewriter &rewriter);
static bool isAddressSeedResolvable(Value value);
static bool isSupportedLoopStepExpr(Value value, Value iterArg);
static Value stripAddressViewLikeChain(Value address);
static bool hasUnsupportedLoopCarriedAddr(Value ptr);

static LogicalResult emitTTAToMemrefError(Operation *op, StringRef reason) {
  op->emitError("tta-to-memref: ") << reason;
  return failure();
}

static void setFailureReason(std::optional<StringRef> *failureReason,
                             StringRef reason) {
  if (failureReason) {
    *failureReason = reason;
  }
}

static FailureOr<Value>
castIndirectIndexToIndex(Value indirectIndex, Location loc,
                         ConversionPatternRewriter &rewriter);

static FailureOr<Value>
buildIdentityIndexTensor(Value indexTensor, Location loc,
                         ConversionPatternRewriter &rewriter) {
  auto tensorType = dyn_cast<RankedTensorType>(indexTensor.getType());
  if (!tensorType || tensorType.getRank() != 1 ||
      !tensorType.getElementType().isIndex()) {
    return failure();
  }

  if (!tensorType.isDynamicDim(0)) {
    int64_t size = tensorType.getDimSize(0);
    SmallVector<int64_t> values;
    values.reserve(size);
    for (int64_t i = 0; i < size; ++i) {
      values.push_back(i);
    }
    auto attr = DenseIntElementsAttr::get(tensorType, values);
    return arith::ConstantOp::create(rewriter, loc, attr).getResult();
  }

  Value extent = tensor::DimOp::create(rewriter, loc, indexTensor, /*index=*/0)
                     .getResult();
  auto generate = tensor::GenerateOp::create(
      rewriter, loc, tensorType, ValueRange{extent},
      [&](OpBuilder &builder, Location bodyLoc, ValueRange indices) {
        tensor::YieldOp::create(builder, bodyLoc, indices.front());
      });
  return generate.getResult();
}

static FailureOr<Value>
buildAllTrueMaskTensor(Value maskTensor, Location loc,
                       ConversionPatternRewriter &rewriter) {
  auto tensorType = dyn_cast<RankedTensorType>(maskTensor.getType());
  if (!tensorType || tensorType.getRank() != 1 ||
      !tensorType.getElementType().isInteger(1)) {
    return failure();
  }

  if (!tensorType.isDynamicDim(0)) {
    auto attr = DenseElementsAttr::get(tensorType, true);
    return arith::ConstantOp::create(rewriter, loc, attr).getResult();
  }

  Value extent =
      tensor::DimOp::create(rewriter, loc, maskTensor, /*index=*/0).getResult();
  auto generate = tensor::GenerateOp::create(
      rewriter, loc, tensorType, ValueRange{extent},
      [&](OpBuilder &builder, Location bodyLoc, ValueRange) {
        Value trueValue = arith::ConstantOp::create(builder, bodyLoc,
                                                    builder.getBoolAttr(true))
                              .getResult();
        tensor::YieldOp::create(builder, bodyLoc, trueValue);
      });
  return generate.getResult();
}

static bool isAddressChainRootedAtMakeAddr(Value value) {
  while (true) {
    if (auto reindex = value.getDefiningOp<tta::ReindexOp>()) {
      value = reindex.getAddress();
      continue;
    }
    if (auto reindex = value.getDefiningOp<tta::IndirectReindexOp>()) {
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

static FailureOr<IndirectInfo>
collectIndirectInfo(const AddressDescriptor &descriptor,
                    std::optional<StringRef> *failureReason = nullptr) {
  (void)failureReason;
  IndirectInfo info;
  for (auto [dim, rule] : llvm::enumerate(descriptor.dims)) {
    if (!rule.indirect.has_value()) {
      continue;
    }
    info.dims.push_back(IndirectDimInfo{static_cast<int64_t>(dim),
                                        rule.indirect->indexTensor,
                                        rule.indirect->maskTensor});
  }
  return info;
}

static AddressFeatures
collectAddressFeatures(const AddressDescriptor &descriptor) {
  AddressFeatures features =
      mlir::triton::address::getAddressFeatures(descriptor);
  // A zero wrap boundary is semantically disabled.
  if (features.hasWrapBoundary) {
    features.hasWrapBoundary =
        llvm::any_of(descriptor.dims, [](const DimRule &rule) {
          return rule.wrapBoundary.has_value() &&
                 !hasConstZero(rule.wrapBoundary->boundary);
        });
  }
  return features;
}

static bool hasAnyIndirectAccess(const IndirectInfo &indirectInfo) {
  return !indirectInfo.dims.empty();
}

static const IndirectDimInfo *
getSingleIndirectDimInfo(const IndirectInfo &indirectInfo) {
  if (indirectInfo.dims.size() != 1) {
    return nullptr;
  }
  return &indirectInfo.dims.front();
}

struct NormalizedIndirectDimInfo {
  int64_t dim;
  Value index;
  Value mask;
};

static FailureOr<SmallVector<NormalizedIndirectDimInfo>>
normalizeIndirectInfo(const IndirectInfo &indirectInfo, Location loc,
                      ConversionPatternRewriter &rewriter) {
  SmallVector<NormalizedIndirectDimInfo> normalized;
  normalized.reserve(indirectInfo.dims.size());
  for (const IndirectDimInfo &dimInfo : indirectInfo.dims) {
    auto maybeIndex = castIndirectIndexToIndex(dimInfo.index, loc, rewriter);
    if (failed(maybeIndex)) {
      return failure();
    }
    normalized.push_back(
        NormalizedIndirectDimInfo{dimInfo.dim, *maybeIndex, dimInfo.mask});
  }
  return normalized;
}

static const NormalizedIndirectDimInfo *findNormalizedIndirectDim(
    ArrayRef<NormalizedIndirectDimInfo> normalizedIndirect, int64_t dim) {
  for (const NormalizedIndirectDimInfo &entry : normalizedIndirect) {
    if (entry.dim == dim) {
      return &entry;
    }
  }
  return nullptr;
}

static AddressLoweringPath
classifyAddressLoweringPath(const AddressFeatures &features,
                            const IndirectInfo &indirectInfo) {
  if (features.hasWrapBoundary) {
    return AddressLoweringPath::WrapAware;
  }
  if (features.hasIndirect || hasAnyIndirectAccess(indirectInfo)) {
    return AddressLoweringPath::Indirect;
  }
  return AddressLoweringPath::Direct;
}

static FailureOr<SmallVector<int64_t>>
collectAddressStaticSizes(const AddressDescriptor &descriptor,
                          std::optional<StringRef> *failureReason = nullptr) {
  SmallVector<int64_t> sizes;
  sizes.reserve(descriptor.dims.size());
  for (const DimRule &rule : descriptor.dims) {
    auto maybeSize = getIntAttr(rule.size);
    if (!maybeSize.has_value()) {
      if (failureReason) {
        *failureReason = StringRef("address size is not constant");
      }
      return failure();
    }
    sizes.push_back(*maybeSize);
  }
  return sizes;
}

static SmallVector<OpFoldResult>
collectAddressOffsets(const AddressDescriptor &descriptor) {
  SmallVector<OpFoldResult> offsets;
  offsets.reserve(descriptor.dims.size());
  for (const DimRule &rule : descriptor.dims) {
    offsets.push_back(rule.offset);
  }
  return offsets;
}

static SmallVector<OpFoldResult>
collectAddressStrides(const AddressDescriptor &descriptor) {
  SmallVector<OpFoldResult> strides;
  strides.reserve(descriptor.dims.size());
  for (const DimRule &rule : descriptor.dims) {
    strides.push_back(rule.stride);
  }
  return strides;
}

static LogicalResult
validateAddressWrapBoundaries(const AddressDescriptor &descriptor,
                              Operation *op) {
  if (descriptor.layoutKind == LayoutKind::Block) {
    return success();
  }

  for (const DimRule &rule : descriptor.dims) {
    if (!rule.wrapBoundary.has_value() ||
        hasConstZero(rule.wrapBoundary->boundary)) {
      continue;
    }

    auto maybeBoundary = getIntAttr(rule.wrapBoundary->boundary);
    if (maybeBoundary && *maybeBoundary <= 0) {
      return emitTTAToMemrefError(op,
                                  "wrap boundary must be greater than zero");
    }
  }

  return success();
}

static std::optional<int64_t>
getStaticLoopUpperBound(ArrayRef<int64_t> loadedShape,
                        ArrayRef<OpFoldResult> maskDims, int64_t dim) {
  if (dim < 0 || dim >= static_cast<int64_t>(loadedShape.size())) {
    return std::nullopt;
  }
  if (maskDims.empty()) {
    int64_t extent = loadedShape[dim];
    if (ShapedType::isDynamic(extent)) {
      return std::nullopt;
    }
    return extent;
  }
  if (maskDims.size() != loadedShape.size()) {
    return std::nullopt;
  }
  return getIntAttr(maskDims[dim]);
}

static bool canProveNoWrapAccess(const AddressDescriptor &descriptor,
                                 const IndirectInfo &indirectInfo,
                                 ArrayRef<int64_t> loadedShape,
                                 ArrayRef<OpFoldResult> maskDims) {
  if (descriptor.layoutKind == LayoutKind::Block) {
    return false;
  }
  if (hasAnyIndirectAccess(indirectInfo)) {
    return false;
  }
  if (descriptor.dims.size() != loadedShape.size()) {
    return false;
  }

  for (auto [dim, rule] : llvm::enumerate(descriptor.dims)) {
    if (!rule.wrapBoundary.has_value() ||
        hasConstZero(rule.wrapBoundary->boundary)) {
      continue;
    }

    auto maybeBoundary = getIntAttr(rule.wrapBoundary->boundary);
    auto maybeOffset = getIntAttr(rule.offset);
    auto maybeStride = getIntAttr(rule.stride);
    auto maybeExtent = getStaticLoopUpperBound(loadedShape, maskDims, dim);
    if (!maybeBoundary || !maybeOffset || !maybeStride || !maybeExtent) {
      return false;
    }
    if (*maybeBoundary <= 0) {
      return false;
    }
    if (*maybeExtent <= 0) {
      continue;
    }

    // Fast-path check assumes normal Triton index ranges where 64-bit affine
    // arithmetic does not overflow.
    int64_t first = *maybeOffset;
    int64_t delta = (*maybeExtent - 1) * (*maybeStride);
    int64_t last = first + delta;
    int64_t lower = first < last ? first : last;
    int64_t upper = first < last ? last : first;
    if (lower < 0 || upper >= *maybeBoundary) {
      return false;
    }
  }

  return true;
}

struct Rank1WrapSegments {
  int64_t firstSourceOffset = 0;
  int64_t firstSize = 0;
  int64_t secondSourceOffset = 0;
  int64_t secondSize = 0;
};

static FailureOr<Rank1WrapSegments> buildRank1WrapSegments(
    const AddressDescriptor &descriptor, const IndirectInfo &indirectInfo,
    ArrayRef<int64_t> loadedShape, ArrayRef<OpFoldResult> maskDims) {
  if (descriptor.layoutKind == LayoutKind::Block ||
      descriptor.dims.size() != 1 || loadedShape.size() != 1 ||
      !maskDims.empty()) {
    return failure();
  }
  if (hasAnyIndirectAccess(indirectInfo)) {
    return failure();
  }
  if (ShapedType::isDynamic(loadedShape[0]) || loadedShape[0] <= 0) {
    return failure();
  }

  const DimRule &dim = descriptor.dims.front();
  if (!dim.wrapBoundary.has_value() ||
      hasConstZero(dim.wrapBoundary->boundary)) {
    return failure();
  }

  auto maybeStride = getIntAttr(dim.stride);
  auto maybeOffset = getIntAttr(dim.offset);
  auto maybeBoundary = getIntAttr(dim.wrapBoundary->boundary);
  if (!maybeStride || !maybeOffset || !maybeBoundary || *maybeBoundary <= 0 ||
      *maybeStride != 1) {
    return failure();
  }

  int64_t totalSize = loadedShape[0];
  if (totalSize > *maybeBoundary) {
    return failure();
  }

  int64_t start = *maybeOffset % *maybeBoundary;
  if (start < 0) {
    start += *maybeBoundary;
  }
  if (start + totalSize <= *maybeBoundary) {
    return failure();
  }

  Rank1WrapSegments segments;
  segments.firstSourceOffset = start;
  segments.firstSize = *maybeBoundary - start;
  segments.secondSourceOffset = 0;
  segments.secondSize = totalSize - segments.firstSize;
  if (segments.firstSize <= 0 || segments.secondSize <= 0) {
    return failure();
  }
  return segments;
}

static FailureOr<SmallVector<OpFoldResult>>
collectAddressOffsetDeltas(Value value, Value root, int64_t rank, Location loc,
                           ConversionPatternRewriter &rewriter) {
  auto collectDeltas =
      [&](auto &&self, Value current) -> FailureOr<SmallVector<OpFoldResult>> {
    if (current == root) {
      SmallVector<OpFoldResult> zeros;
      zeros.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        zeros.push_back(rewriter.getIndexAttr(0));
      }
      return zeros;
    }

    if (auto imported = current.getDefiningOp<tta::FromTTPtrOp>()) {
      return self(self, imported.getSource());
    }

    if (auto advance = current.getDefiningOp<tta::AdvanceOp>()) {
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

    if (auto reindex = current.getDefiningOp<tta::ReindexOp>()) {
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

    if (current.getDefiningOp<tta::IndirectReindexOp>()) {
      return failure();
    }

    return failure();
  };

  return collectDeltas(collectDeltas, value);
}

static FailureOr<AddressStepInfo>
collectAddressStepInfo(Value value, Value root, int64_t rank, Location loc,
                       ConversionPatternRewriter &rewriter) {
  auto collectStep = [&](auto &&self,
                         Value current) -> FailureOr<AddressStepInfo> {
    if (current == root) {
      AddressStepInfo info;
      info.offsets.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        info.offsets.push_back(rewriter.getIndexAttr(0));
      }
      return info;
    }

    if (auto imported = current.getDefiningOp<tta::FromTTPtrOp>()) {
      return self(self, imported.getSource());
    }

    if (auto advance = current.getDefiningOp<tta::AdvanceOp>()) {
      auto maybeInfo = self(self, advance.getAddress());
      if (failed(maybeInfo)) {
        return failure();
      }
      auto deltas = advance.getMixedDeltas();
      if (static_cast<int64_t>(deltas.size()) != rank) {
        return failure();
      }
      for (auto [i, delta] : llvm::enumerate(deltas)) {
        maybeInfo->offsets[i] =
            addOFRs(maybeInfo->offsets[i], delta, loc, rewriter);
      }
      return *maybeInfo;
    }

    if (auto reindex = current.getDefiningOp<tta::ReindexOp>()) {
      auto maybeInfo = self(self, reindex.getAddress());
      if (failed(maybeInfo)) {
        return failure();
      }
      auto offsets = reindex.getMixedOffsets();
      if (static_cast<int64_t>(offsets.size()) != rank) {
        return failure();
      }
      for (auto [i, offset] : llvm::enumerate(offsets)) {
        maybeInfo->offsets[i] =
            addOFRs(maybeInfo->offsets[i], offset, loc, rewriter);
      }
      return *maybeInfo;
    }

    if (auto indirect = current.getDefiningOp<tta::IndirectReindexOp>()) {
      auto maybeInfo = self(self, indirect.getAddress());
      if (failed(maybeInfo)) {
        return failure();
      }
      int64_t dim = indirect.getIndirectDimAttr().getInt();
      if (dim < 0 || dim >= rank) {
        return failure();
      }
      maybeInfo->indirects.push_back(IndirectDimInfo{
          dim, indirect.getIndirectIndex(), indirect.getMask()});
      return *maybeInfo;
    }

    return failure();
  };

  return collectStep(collectStep, value);
}

static FailureOr<AddressDescriptor> collectAddressDescriptorWithCommonAnalysis(
    Value address, Location loc, ConversionPatternRewriter &rewriter,
    std::optional<StringRef> *failureReason = nullptr) {
  if (!isAddressChainRootedAtMakeAddr(address)) {
    if (failureReason) {
      *failureReason = StringRef("unsupported address chain");
    }
    return failure();
  }

  mlir::triton::address::AnalysisAddress analysis;
  auto maybeDescriptor =
      analysis.analyzeDescriptor(address, loc, rewriter, failureReason);
  if (failed(maybeDescriptor)) {
    if (failureReason && !failureReason->has_value()) {
      *failureReason = StringRef("unsupported address chain");
    }
    return failure();
  }
  return *maybeDescriptor;
}

static FailureOr<AddressDescriptor>
collectAddressDescriptor(Value address, Location loc,
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

    FailureOr<AddressDescriptor> maybeInitDescriptor = collectAddressDescriptor(
        iterArgInfo.forOp.getInitArgs()[iterArgInfo.iterArgIndex], loc,
        rewriter, failureReason);
    if (failed(maybeInitDescriptor)) {
      return failure();
    }

    auto yieldOp =
        cast<scf::YieldOp>(iterArgInfo.forOp.getBody()->getTerminator());
    Value yielded = yieldOp.getResults()[iterArgInfo.iterArgIndex];
    int64_t rank = static_cast<int64_t>(maybeInitDescriptor->dims.size());
    auto maybeStepInfo =
        collectAddressStepInfo(yielded, address, rank, loc, rewriter);
    if (failed(maybeStepInfo)) {
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

    for (auto [i, stepOffset] : llvm::enumerate(maybeStepInfo->offsets)) {
      OpFoldResult scaled = mulOFRs(stepOffset, iterCount, loc, rewriter);
      maybeInitDescriptor->dims[i].offset =
          addOFRs(maybeInitDescriptor->dims[i].offset, scaled, loc, rewriter);
    }

    if (!maybeStepInfo->indirects.empty()) {
      SmallVector<Value> mergedStepIndices(rank);
      SmallVector<Value> mergedStepMasks(rank);
      SmallVector<RankedTensorType> mergedStepTypes(rank);

      for (const IndirectDimInfo &info : maybeStepInfo->indirects) {
        int64_t dim = info.dim;
        if (dim < 0 || dim >= rank) {
          setFailureReason(
              failureReason,
              "loop-carried indirect recurrence indirect_dim out of bounds");
          return failure();
        }

        auto maybeStepIndex =
            castIndirectIndexToIndex(info.index, loc, rewriter);
        if (failed(maybeStepIndex)) {
          setFailureReason(
              failureReason,
              "loop-carried indirect recurrence requires 1D int/index step "
              "index");
          return failure();
        }
        auto stepType = dyn_cast<RankedTensorType>((*maybeStepIndex).getType());
        if (!stepType || stepType.getRank() != 1 ||
            !stepType.getElementType().isIndex()) {
          setFailureReason(
              failureReason,
              "loop-carried indirect recurrence requires 1D index");
          return failure();
        }

        if (!mergedStepTypes[dim]) {
          mergedStepTypes[dim] = stepType;
          mergedStepIndices[dim] = *maybeStepIndex;
        } else {
          if (stepType.getShape() != mergedStepTypes[dim].getShape() ||
              stepType.getElementType() !=
                  mergedStepTypes[dim].getElementType()) {
            setFailureReason(failureReason,
                             "loop-carried indirect recurrence step index "
                             "shape/type mismatch on same dim");
            return failure();
          }
          mergedStepIndices[dim] =
              arith::AddIOp::create(rewriter, loc, mergedStepIndices[dim],
                                    *maybeStepIndex)
                  .getResult();
        }

        if (Value mask = info.mask) {
          auto maskType = dyn_cast<RankedTensorType>(mask.getType());
          if (!maskType || maskType.getRank() != 1 ||
              !maskType.getElementType().isInteger(1)) {
            setFailureReason(failureReason,
                             "loop-carried indirect recurrence requires 1D "
                             "mask");
            return failure();
          }
          if (maskType.getShape() != stepType.getShape()) {
            setFailureReason(
                failureReason,
                "loop-carried indirect recurrence step mask shape mismatch "
                "with step index on same dim");
            return failure();
          }
          if (!mergedStepMasks[dim]) {
            mergedStepMasks[dim] = mask;
          } else {
            mergedStepMasks[dim] =
                arith::AndIOp::create(rewriter, loc, mergedStepMasks[dim], mask)
                    .getResult();
          }
        }
      }

      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
      Value isIterZero =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                iterCount, zero)
              .getResult();

      for (int64_t recurrenceDim = 0; recurrenceDim < rank; ++recurrenceDim) {
        RankedTensorType expectedType = mergedStepTypes[recurrenceDim];
        if (!expectedType) {
          continue;
        }
        Value mergedStepIndex = mergedStepIndices[recurrenceDim];
        Value mergedStepMask = mergedStepMasks[recurrenceDim];

        SmallVector<Value> iterCountDynamicSizes;
        if (expectedType.isDynamicDim(0)) {
          iterCountDynamicSizes.push_back(
              tensor::DimOp::create(rewriter, loc, mergedStepIndex, /*index=*/0)
                  .getResult());
        }
        Value iterCountTensor =
            tensor::SplatOp::create(rewriter, loc, iterCount, expectedType,
                                    iterCountDynamicSizes)
                .getResult();
        Value scaledStepIndex =
            arith::MulIOp::create(rewriter, loc, mergedStepIndex,
                                  iterCountTensor)
                .getResult();

        auto &dimRule = maybeInitDescriptor->dims[recurrenceDim];
        if (dimRule.indirect.has_value()) {
          auto maybeSeedIndex = castIndirectIndexToIndex(
              dimRule.indirect->indexTensor, loc, rewriter);
          if (failed(maybeSeedIndex)) {
            setFailureReason(
                failureReason,
                "loop-carried indirect recurrence requires 1D int/index seed "
                "index");
            return failure();
          }

          auto seedType =
              dyn_cast<RankedTensorType>((*maybeSeedIndex).getType());
          if (!seedType || seedType.getRank() != 1 ||
              !seedType.getElementType().isIndex()) {
            setFailureReason(failureReason,
                             "loop-carried indirect recurrence requires 1D "
                             "index");
            return failure();
          }
          if (seedType.getShape() != expectedType.getShape() ||
              seedType.getElementType() != expectedType.getElementType()) {
            setFailureReason(
                failureReason,
                "loop-carried indirect recurrence seed/step index shape/type "
                "mismatch on same dim");
            return failure();
          }

          dimRule.indirect->indexTensor =
              arith::AddIOp::create(rewriter, loc, *maybeSeedIndex,
                                    scaledStepIndex)
                  .getResult();

          if (mergedStepMask) {
            Value seedMask = dimRule.indirect->maskTensor;
            Value mergedMask = mergedStepMask;
            if (seedMask) {
              auto seedMaskType =
                  dyn_cast<RankedTensorType>(seedMask.getType());
              auto stepMaskType =
                  dyn_cast<RankedTensorType>(mergedStepMask.getType());
              if (!seedMaskType || !stepMaskType ||
                  seedMaskType.getRank() != 1 || stepMaskType.getRank() != 1 ||
                  !seedMaskType.getElementType().isInteger(1) ||
                  !stepMaskType.getElementType().isInteger(1)) {
                setFailureReason(
                    failureReason,
                    "loop-carried indirect recurrence requires 1D mask");
                return failure();
              }
              if (seedMaskType.getShape() != stepMaskType.getShape()) {
                setFailureReason(
                    failureReason,
                    "loop-carried indirect recurrence seed/step mask shape "
                    "mismatch on same dim");
                return failure();
              }
              Value andMask =
                  arith::AndIOp::create(rewriter, loc, seedMask, mergedStepMask)
                      .getResult();
              auto ifOp = scf::IfOp::create(rewriter, loc, seedMask.getType(),
                                            isIterZero, true);
              rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
              scf::YieldOp::create(rewriter, loc, ValueRange{seedMask});
              rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
              scf::YieldOp::create(rewriter, loc, ValueRange{andMask});
              rewriter.setInsertionPointAfter(ifOp);
              mergedMask = ifOp.getResult(0);
            } else {
              auto maskType =
                  dyn_cast<RankedTensorType>(mergedStepMask.getType());
              if (!maskType || maskType.getRank() != 1 ||
                  !maskType.getElementType().isInteger(1)) {
                setFailureReason(
                    failureReason,
                    "loop-carried indirect recurrence requires 1D mask");
                return failure();
              }
              auto maybeAllTrueMask =
                  buildAllTrueMaskTensor(mergedStepMask, loc, rewriter);
              if (failed(maybeAllTrueMask)) {
                setFailureReason(
                    failureReason,
                    "loop-carried indirect recurrence requires 1D mask");
                return failure();
              }
              auto ifOp = scf::IfOp::create(
                  rewriter, loc, mergedStepMask.getType(), isIterZero, true);
              rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
              scf::YieldOp::create(rewriter, loc,
                                   ValueRange{*maybeAllTrueMask});
              rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
              scf::YieldOp::create(rewriter, loc, ValueRange{mergedStepMask});
              rewriter.setInsertionPointAfter(ifOp);
              mergedMask = ifOp.getResult(0);
            }
            dimRule.indirect->maskTensor = mergedMask;
          }
          continue;
        }

        auto maybeStepOffset =
            getIntAttr(maybeStepInfo->offsets[recurrenceDim]);
        if (!maybeStepOffset || *maybeStepOffset != 0) {
          setFailureReason(
              failureReason,
              "loop-carried indirect recurrence without seed requires "
              "zero direct step on same dim");
          return failure();
        }

        auto maybeIdentityIndex =
            buildIdentityIndexTensor(mergedStepIndex, loc, rewriter);
        if (failed(maybeIdentityIndex)) {
          setFailureReason(failureReason, "loop-carried indirect recurrence "
                                          "without seed requires 1D index");
          return failure();
        }

        auto ifOp = scf::IfOp::create(rewriter, loc, expectedType, isIterZero,
                                      /*withElseRegion=*/true);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        scf::YieldOp::create(rewriter, loc, ValueRange{*maybeIdentityIndex});
        rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
        scf::YieldOp::create(rewriter, loc, ValueRange{scaledStepIndex});
        rewriter.setInsertionPointAfter(ifOp);

        dimRule.indirect = mlir::triton::address::IndirectIndexRule{
            ifOp.getResult(0), Value()};

        if (mergedStepMask) {
          auto maskType = dyn_cast<RankedTensorType>(mergedStepMask.getType());
          if (!maskType || maskType.getRank() != 1 ||
              !maskType.getElementType().isInteger(1)) {
            setFailureReason(failureReason,
                             "loop-carried indirect recurrence requires 1D "
                             "mask");
            return failure();
          }
          auto maybeAllTrueMask =
              buildAllTrueMaskTensor(mergedStepMask, loc, rewriter);
          if (failed(maybeAllTrueMask)) {
            setFailureReason(failureReason,
                             "loop-carried indirect recurrence requires 1D "
                             "mask");
            return failure();
          }
          auto maskIfOp = scf::IfOp::create(
              rewriter, loc, mergedStepMask.getType(), isIterZero, true);
          rewriter.setInsertionPointToStart(&maskIfOp.getThenRegion().front());
          scf::YieldOp::create(rewriter, loc, ValueRange{*maybeAllTrueMask});
          rewriter.setInsertionPointToStart(&maskIfOp.getElseRegion().front());
          scf::YieldOp::create(rewriter, loc, ValueRange{mergedStepMask});
          rewriter.setInsertionPointAfter(maskIfOp);
          dimRule.indirect->maskTensor = maskIfOp.getResult(0);
        }
      }
    }

    return *maybeInitDescriptor;
  }

  Value root = stripAddressViewLikeChain(address);
  if (root != address) {
    auto maybeIterArgInfo = getForIterArgInfo(root);
    if (succeeded(maybeIterArgInfo)) {
      auto maybeRootDescriptor =
          collectAddressDescriptor(root, loc, rewriter, failureReason);
      if (failed(maybeRootDescriptor)) {
        return failure();
      }

      int64_t rank = static_cast<int64_t>(maybeRootDescriptor->dims.size());
      auto maybeViewOffsets =
          collectAddressOffsetDeltas(address, root, rank, loc, rewriter);
      if (succeeded(maybeViewOffsets)) {
        for (auto [i, offset] : llvm::enumerate(*maybeViewOffsets)) {
          maybeRootDescriptor->dims[i].offset = addOFRs(
              maybeRootDescriptor->dims[i].offset, offset, loc, rewriter);
        }
        return *maybeRootDescriptor;
      }
    }
  }

  if (auto maybeDescriptor = collectAddressDescriptorWithCommonAnalysis(
          address, loc, rewriter, failureReason);
      succeeded(maybeDescriptor)) {
    return *maybeDescriptor;
  }

  if (failureReason && !failureReason->has_value()) {
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
    if (auto reindex = address.getDefiningOp<tta::IndirectReindexOp>()) {
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
      value = reindex.getAddress();
      continue;
    }

    if (auto reindex = value.getDefiningOp<tta::IndirectReindexOp>()) {
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
    return isSupportedLoopStepExpr(reindex.getAddress(), iterArg);
  }

  if (auto reindex = value.getDefiningOp<tta::IndirectReindexOp>()) {
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
                   const AddressDescriptor &descriptor,
                   std::optional<StringRef> *failureReason = nullptr) {
  auto maybeSizes = collectAddressStaticSizes(descriptor, failureReason);
  if (failed(maybeSizes)) {
    return failure();
  }

  if (loadedInfo.shape.size() != maybeSizes->size()) {
    return failure();
  }

  SmallVector<int64_t> shape(loadedInfo.shape.begin(), loadedInfo.shape.end());
  for (auto [index, dim] : llvm::enumerate(shape)) {
    if (ShapedType::isDynamic(dim)) {
      shape[index] = (*maybeSizes)[index];
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

static FailureOr<Value>
castBaseMemrefToLinear(Value baseMemref, Type elementType, Location loc,
                       ConversionPatternRewriter &rewriter);

static LogicalResult lowerRank1WrappedLoadByCopy(
    tta::LoadOp op, const AddressDescriptor &descriptor,
    const IndirectInfo &indirectInfo, ArrayRef<int64_t> loadedShape,
    ArrayRef<OpFoldResult> maskDims, ArrayRef<int64_t> sourceSizes,
    Value baseMemref, Value alloc, ConversionPatternRewriter &rewriter) {
  auto maybeSegments =
      buildRank1WrapSegments(descriptor, indirectInfo, loadedShape, maskDims);
  if (failed(maybeSegments)) {
    return failure();
  }

  auto elementType = cast<MemRefType>(alloc.getType()).getElementType();
  auto maybeLinearBase =
      castBaseMemrefToLinear(baseMemref, elementType, op.getLoc(), rewriter);
  if (failed(maybeLinearBase)) {
    return failure();
  }

  SmallVector<OpFoldResult> sourceStrides = collectAddressStrides(descriptor);
  SmallVector<OpFoldResult> unitStride = getOneStrides(/*rank=*/1, rewriter);
  const Rank1WrapSegments &segments = *maybeSegments;

  {
    SmallVector<int64_t> firstShape{segments.firstSize};
    SmallVector<OpFoldResult> firstSrcOffsets{
        rewriter.getIndexAttr(segments.firstSourceOffset)};
    auto maybeFirstSrc = createReinterpretCast(
        *maybeLinearBase, elementType, firstShape, sourceSizes, firstSrcOffsets,
        sourceStrides, /*gatherDim=*/std::nullopt, op.getLoc(), rewriter);
    if (failed(maybeFirstSrc)) {
      return failure();
    }

    SmallVector<OpFoldResult> dstOffsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> dstSizes{
        rewriter.getIndexAttr(segments.firstSize)};
    auto firstDst = createSubview(alloc, dstOffsets, dstSizes, unitStride,
                                  op.getLoc(), rewriter);
    memref::CopyOp::create(rewriter, op.getLoc(), *maybeFirstSrc, firstDst);
  }

  {
    SmallVector<int64_t> secondShape{segments.secondSize};
    SmallVector<OpFoldResult> secondSrcOffsets{
        rewriter.getIndexAttr(segments.secondSourceOffset)};
    auto maybeSecondSrc = createReinterpretCast(
        *maybeLinearBase, elementType, secondShape, sourceSizes,
        secondSrcOffsets, sourceStrides, /*gatherDim=*/std::nullopt,
        op.getLoc(), rewriter);
    if (failed(maybeSecondSrc)) {
      return failure();
    }

    SmallVector<OpFoldResult> dstOffsets{
        rewriter.getIndexAttr(segments.firstSize)};
    SmallVector<OpFoldResult> dstSizes{
        rewriter.getIndexAttr(segments.secondSize)};
    auto secondDst = createSubview(alloc, dstOffsets, dstSizes, unitStride,
                                   op.getLoc(), rewriter);
    memref::CopyOp::create(rewriter, op.getLoc(), *maybeSecondSrc, secondDst);
  }

  return success();
}

static LogicalResult lowerRank1WrappedStoreByCopy(
    tta::StoreOp op, const AddressDescriptor &descriptor,
    const IndirectInfo &indirectInfo, ArrayRef<int64_t> loadedShape,
    ArrayRef<OpFoldResult> maskDims, ArrayRef<int64_t> sourceSizes,
    Value baseMemref, Value valueTensor, ConversionPatternRewriter &rewriter) {
  auto maybeSegments =
      buildRank1WrapSegments(descriptor, indirectInfo, loadedShape, maskDims);
  if (failed(maybeSegments)) {
    return failure();
  }

  auto elementType =
      cast<RankedTensorType>(valueTensor.getType()).getElementType();
  auto maybeLinearBase =
      castBaseMemrefToLinear(baseMemref, elementType, op.getLoc(), rewriter);
  if (failed(maybeLinearBase)) {
    return failure();
  }

  SmallVector<OpFoldResult> sourceStrides = collectAddressStrides(descriptor);
  SmallVector<OpFoldResult> unitStride = getOneStrides(/*rank=*/1, rewriter);
  const Rank1WrapSegments &segments = *maybeSegments;

  {
    SmallVector<int64_t> firstShape{segments.firstSize};
    SmallVector<OpFoldResult> firstDstOffsets{
        rewriter.getIndexAttr(segments.firstSourceOffset)};
    auto maybeFirstDst = createReinterpretCast(
        *maybeLinearBase, elementType, firstShape, sourceSizes, firstDstOffsets,
        sourceStrides, /*gatherDim=*/std::nullopt, op.getLoc(), rewriter);
    if (failed(maybeFirstDst)) {
      return failure();
    }

    SmallVector<OpFoldResult> srcOffsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> srcSizes{
        rewriter.getIndexAttr(segments.firstSize)};
    auto firstSlice = createExtractSlice(valueTensor, srcOffsets, srcSizes,
                                         unitStride, op.getLoc(), rewriter);
    auto firstStore = bufferization::MaterializeInDestinationOp::create(
        rewriter, op.getLoc(), firstSlice.getResult(), *maybeFirstDst);
    firstStore.setWritable(true);
  }

  {
    SmallVector<int64_t> secondShape{segments.secondSize};
    SmallVector<OpFoldResult> secondDstOffsets{
        rewriter.getIndexAttr(segments.secondSourceOffset)};
    auto maybeSecondDst = createReinterpretCast(
        *maybeLinearBase, elementType, secondShape, sourceSizes,
        secondDstOffsets, sourceStrides, /*gatherDim=*/std::nullopt,
        op.getLoc(), rewriter);
    if (failed(maybeSecondDst)) {
      return failure();
    }

    SmallVector<OpFoldResult> srcOffsets{
        rewriter.getIndexAttr(segments.firstSize)};
    SmallVector<OpFoldResult> srcSizes{
        rewriter.getIndexAttr(segments.secondSize)};
    auto secondSlice = createExtractSlice(valueTensor, srcOffsets, srcSizes,
                                          unitStride, op.getLoc(), rewriter);
    auto secondStore = bufferization::MaterializeInDestinationOp::create(
        rewriter, op.getLoc(), secondSlice.getResult(), *maybeSecondDst);
    secondStore.setWritable(true);
  }

  return success();
}

static FailureOr<Value>
castBaseMemrefToLinear(Value baseMemref, Type elementType, Location loc,
                       ConversionPatternRewriter &rewriter) {
  auto linearType = MemRefType::get({ShapedType::kDynamic}, elementType);

  if (auto unranked = dyn_cast<UnrankedMemRefType>(baseMemref.getType())) {
    if (unranked.getElementType() != elementType) {
      return failure();
    }
    return memref::CastOp::create(rewriter, loc, linearType, baseMemref)
        .getResult();
  }

  if (auto ranked = dyn_cast<MemRefType>(baseMemref.getType())) {
    if (ranked.getElementType() != elementType || ranked.getRank() != 1) {
      return failure();
    }
    if (ranked == linearType) {
      return baseMemref;
    }
    return memref::CastOp::create(rewriter, loc, linearType, baseMemref)
        .getResult();
  }

  return failure();
}

static FailureOr<SmallVector<Value>> buildWrapAwareLoopUpperBounds(
    const AddressDescriptor &descriptor, ArrayRef<int64_t> loadedShape,
    ArrayRef<OpFoldResult> maskDims,
    ArrayRef<NormalizedIndirectDimInfo> normalizedIndirect, Location loc,
    ConversionPatternRewriter &rewriter) {
  int64_t rank = loadedShape.size();
  if (rank <= 0 || static_cast<int64_t>(descriptor.dims.size()) != rank) {
    return failure();
  }

  SmallVector<Value> upperBounds;
  upperBounds.reserve(rank);

  if (maskDims.empty()) {
    for (int64_t size : loadedShape) {
      if (ShapedType::isDynamic(size)) {
        return failure();
      }
      upperBounds.push_back(makeIndexConstant(loc, size, rewriter));
    }
  } else {
    if (static_cast<int64_t>(maskDims.size()) != rank) {
      return failure();
    }
    for (OpFoldResult dim : maskDims) {
      upperBounds.push_back(ofrToIndexValue(dim, loc, rewriter));
    }
  }

  if (normalizedIndirect.empty()) {
    return upperBounds;
  }

  for (const NormalizedIndirectDimInfo &indirectDim : normalizedIndirect) {
    if (indirectDim.dim < 0 || indirectDim.dim >= rank) {
      return failure();
    }
    auto maybeIndirectSize = getIndirectSize(indirectDim.index, loc, rewriter);
    if (failed(maybeIndirectSize)) {
      return failure();
    }
    upperBounds[indirectDim.dim] =
        arith::MinSIOp::create(rewriter, loc, upperBounds[indirectDim.dim],
                               *maybeIndirectSize)
            .getResult();
  }
  return upperBounds;
}

static Value buildEuclideanModulo(Value dividend, Value divisor, Location loc,
                                  ConversionPatternRewriter &rewriter) {
  Value rem =
      arith::RemSIOp::create(rewriter, loc, dividend, divisor).getResult();
  Value zero = makeIndexConstant(loc, 0, rewriter);
  Value isNegative =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, rem, zero)
          .getResult();
  Value adjusted =
      arith::AddIOp::create(rewriter, loc, rem, divisor).getResult();
  return arith::SelectOp::create(rewriter, loc, isNegative, adjusted, rem)
      .getResult();
}

static FailureOr<Value>
buildWrappedLinearizedTerm(const DimRule &rule, Value logicalIndex,
                           Location loc, ConversionPatternRewriter &rewriter) {
  Value stride = ofrToIndexValue(rule.stride, loc, rewriter);
  Value offset = ofrToIndexValue(rule.offset, loc, rewriter);
  Value scaled =
      arith::MulIOp::create(rewriter, loc, logicalIndex, stride).getResult();
  Value term = arith::AddIOp::create(rewriter, loc, offset, scaled).getResult();

  if (!rule.wrapBoundary.has_value() ||
      hasConstZero(rule.wrapBoundary->boundary)) {
    return term;
  }

  auto maybeBoundary = getIntAttr(rule.wrapBoundary->boundary);
  if (maybeBoundary && *maybeBoundary <= 0) {
    return failure();
  }

  Value boundary = ofrToIndexValue(rule.wrapBoundary->boundary, loc, rewriter);
  if (!maybeBoundary) {
    Value zero = makeIndexConstant(loc, 0, rewriter);
    Value isPositive =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                              boundary, zero)
            .getResult();
    cf::AssertOp::create(rewriter, loc, isPositive,
                         "tta-to-memref: wrap boundary must be > 0");
  }
  return buildEuclideanModulo(term, boundary, loc, rewriter);
}

static FailureOr<Value> buildWrapAwareLinearOffset(
    const AddressDescriptor &descriptor,
    ArrayRef<NormalizedIndirectDimInfo> normalizedIndirect, ArrayRef<Value> ivs,
    Location loc, ConversionPatternRewriter &rewriter) {
  int64_t rank = descriptor.dims.size();
  if (static_cast<int64_t>(ivs.size()) != rank) {
    return failure();
  }

  for (const NormalizedIndirectDimInfo &indirectDim : normalizedIndirect) {
    if (indirectDim.dim < 0 || indirectDim.dim >= rank) {
      return failure();
    }
  }

  Value total = makeIndexConstant(loc, 0, rewriter);
  for (auto [dim, iv] : llvm::enumerate(ivs)) {
    Value logicalIndex = iv;
    if (const NormalizedIndirectDimInfo *indirectDim =
            findNormalizedIndirectDim(normalizedIndirect,
                                      static_cast<int64_t>(dim))) {
      logicalIndex = tensor::ExtractOp::create(
                         rewriter, loc, indirectDim->index, ValueRange{iv})
                         .getResult();
    }

    FailureOr<Value> maybeTerm = buildWrappedLinearizedTerm(
        descriptor.dims[dim], logicalIndex, loc, rewriter);
    if (failed(maybeTerm)) {
      return failure();
    }
    total = arith::AddIOp::create(rewriter, loc, total, *maybeTerm).getResult();
  }
  return total;
}

static FailureOr<Value> buildIndirectExecutionMask(
    ArrayRef<NormalizedIndirectDimInfo> normalizedIndirect, ArrayRef<Value> ivs,
    Location loc, ConversionPatternRewriter &rewriter) {
  Value combinedMask;
  for (const NormalizedIndirectDimInfo &indirectDim : normalizedIndirect) {
    if (!indirectDim.mask) {
      continue;
    }
    if (indirectDim.dim < 0 ||
        indirectDim.dim >= static_cast<int64_t>(ivs.size())) {
      return failure();
    }
    Value currentMask =
        tensor::ExtractOp::create(rewriter, loc, indirectDim.mask,
                                  ValueRange{ivs[indirectDim.dim]})
            .getResult();
    if (!combinedMask) {
      combinedMask = currentMask;
      continue;
    }
    combinedMask =
        arith::AndIOp::create(rewriter, loc, combinedMask, currentMask)
            .getResult();
  }
  return combinedMask;
}

static LogicalResult lowerWrapAwareLoad(tta::LoadOp op,
                                        const AddressDescriptor &descriptor,
                                        const IndirectInfo &indirectInfo,
                                        ArrayRef<int64_t> loadedShape,
                                        ArrayRef<OpFoldResult> maskDims,
                                        Value baseMemref, Value alloc,
                                        ConversionPatternRewriter &rewriter) {
  auto maybeLinearBase = castBaseMemrefToLinear(
      baseMemref, cast<MemRefType>(alloc.getType()).getElementType(),
      op.getLoc(), rewriter);
  if (failed(maybeLinearBase)) {
    return failure();
  }

  auto maybeNormalizedIndirect =
      normalizeIndirectInfo(indirectInfo, op.getLoc(), rewriter);
  if (failed(maybeNormalizedIndirect)) {
    return failure();
  }

  auto maybeUpperBounds = buildWrapAwareLoopUpperBounds(
      descriptor, loadedShape, maskDims, *maybeNormalizedIndirect, op.getLoc(),
      rewriter);
  if (failed(maybeUpperBounds)) {
    return failure();
  }

  int64_t rank = loadedShape.size();
  Value lowerBound = makeIndexConstant(op.getLoc(), 0, rewriter);
  Value step = makeIndexConstant(op.getLoc(), 1, rewriter);
  SmallVector<Value> ivs;
  ivs.reserve(rank);

  auto emitLoopNest = [&](auto &&self, int64_t dim) -> LogicalResult {
    if (dim == rank) {
      auto emitOne = [&](ConversionPatternRewriter &rw) -> LogicalResult {
        auto maybeLinearOffset = buildWrapAwareLinearOffset(
            descriptor, *maybeNormalizedIndirect, ivs, op.getLoc(), rw);
        if (failed(maybeLinearOffset)) {
          return failure();
        }
        Value loaded = memref::LoadOp::create(rw, op.getLoc(), *maybeLinearBase,
                                              ValueRange{*maybeLinearOffset})
                           .getResult();
        memref::StoreOp::create(rw, op.getLoc(), loaded, alloc,
                                ValueRange(ivs));
        return success();
      };

      auto maybeMask = buildIndirectExecutionMask(*maybeNormalizedIndirect, ivs,
                                                  op.getLoc(), rewriter);
      if (failed(maybeMask)) {
        return failure();
      }
      if (*maybeMask) {
        Value maskValue = *maybeMask;
        auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), maskValue);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        if (failed(emitOne(rewriter))) {
          return failure();
        }
        rewriter.setInsertionPointAfter(ifOp);
      } else {
        if (failed(emitOne(rewriter))) {
          return failure();
        }
      }
      return success();
    }

    auto loop = scf::ForOp::create(rewriter, op.getLoc(), lowerBound,
                                   (*maybeUpperBounds)[dim], step);
    rewriter.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
    if (failed(self(self, dim + 1))) {
      return failure();
    }
    ivs.pop_back();
    rewriter.setInsertionPointAfter(loop);
    return success();
  };

  return emitLoopNest(emitLoopNest, 0);
}

static LogicalResult lowerWrapAwareStore(tta::StoreOp op,
                                         const AddressDescriptor &descriptor,
                                         const IndirectInfo &indirectInfo,
                                         ArrayRef<int64_t> loadedShape,
                                         ArrayRef<OpFoldResult> maskDims,
                                         Value baseMemref, Value valueTensor,
                                         ConversionPatternRewriter &rewriter) {
  auto maybeLinearBase = castBaseMemrefToLinear(
      baseMemref,
      cast<RankedTensorType>(valueTensor.getType()).getElementType(),
      op.getLoc(), rewriter);
  if (failed(maybeLinearBase)) {
    return failure();
  }

  auto maybeNormalizedIndirect =
      normalizeIndirectInfo(indirectInfo, op.getLoc(), rewriter);
  if (failed(maybeNormalizedIndirect)) {
    return failure();
  }

  auto maybeUpperBounds = buildWrapAwareLoopUpperBounds(
      descriptor, loadedShape, maskDims, *maybeNormalizedIndirect, op.getLoc(),
      rewriter);
  if (failed(maybeUpperBounds)) {
    return failure();
  }

  int64_t rank = loadedShape.size();
  Value lowerBound = makeIndexConstant(op.getLoc(), 0, rewriter);
  Value step = makeIndexConstant(op.getLoc(), 1, rewriter);
  SmallVector<Value> ivs;
  ivs.reserve(rank);

  auto emitLoopNest = [&](auto &&self, int64_t dim) -> LogicalResult {
    if (dim == rank) {
      auto emitOne = [&](ConversionPatternRewriter &rw) -> LogicalResult {
        auto maybeLinearOffset = buildWrapAwareLinearOffset(
            descriptor, *maybeNormalizedIndirect, ivs, op.getLoc(), rw);
        if (failed(maybeLinearOffset)) {
          return failure();
        }
        Value scalar = tensor::ExtractOp::create(rw, op.getLoc(), valueTensor,
                                                 ValueRange(ivs))
                           .getResult();
        memref::StoreOp::create(rw, op.getLoc(), scalar, *maybeLinearBase,
                                ValueRange{*maybeLinearOffset});
        return success();
      };

      auto maybeMask = buildIndirectExecutionMask(*maybeNormalizedIndirect, ivs,
                                                  op.getLoc(), rewriter);
      if (failed(maybeMask)) {
        return failure();
      }
      if (*maybeMask) {
        Value maskValue = *maybeMask;
        auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), maskValue);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        if (failed(emitOne(rewriter))) {
          return failure();
        }
        rewriter.setInsertionPointAfter(ifOp);
      } else {
        if (failed(emitOne(rewriter))) {
          return failure();
        }
      }
      return success();
    }

    auto loop = scf::ForOp::create(rewriter, op.getLoc(), lowerBound,
                                   (*maybeUpperBounds)[dim], step);
    rewriter.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
    if (failed(self(self, dim + 1))) {
      return failure();
    }
    ivs.pop_back();
    rewriter.setInsertionPointAfter(loop);
    return success();
  };

  return emitLoopNest(emitLoopNest, 0);
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
    FailureOr<AddressDescriptor> maybeDescriptor = collectAddressDescriptor(
        adaptor.getPtr(), op.getLoc(), rewriter, &collectFailureReason);
    if (failed(maybeDescriptor)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    AddressDescriptor descriptor = *maybeDescriptor;
    if (failed(validateAddressWrapBoundaries(descriptor, op.getOperation()))) {
      return failure();
    }
    AddressFeatures addressFeatures = collectAddressFeatures(descriptor);

    auto maybeIndirectInfo =
        collectIndirectInfo(descriptor, &collectFailureReason);
    if (failed(maybeIndirectInfo)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    const IndirectDimInfo *singleIndirect =
        getSingleIndirectDimInfo(*maybeIndirectInfo);

    auto maybeLoadedShape =
        resolveLoadedShape(*loadedInfo, descriptor, &collectFailureReason);
    if (failed(maybeLoadedShape)) {
      return rewriter.notifyMatchFailure(op, "failed to resolve loaded shape");
    }
    SmallVector<int64_t> loadedShape = *maybeLoadedShape;

    auto maybeSizes =
        collectAddressStaticSizes(descriptor, &collectFailureReason);
    if (failed(maybeSizes)) {
      return rewriter.notifyMatchFailure(op, "failed to resolve address sizes");
    }

    auto maybeBaseMemref = getBaseMemref(
        descriptor.base, loadedInfo->elementType, op.getLoc(), rewriter);
    if (failed(maybeBaseMemref)) {
      return rewriter.notifyMatchFailure(op, "failed to get base memref");
    }

    auto allocType = MemRefType::get(loadedShape, loadedInfo->elementType);
    Value alloc =
        memref::AllocOp::create(rewriter, op.getLoc(), allocType).getResult();

    auto ones = getOneStrides(loadedInfo->rank, rewriter);
    auto zeros = getZeroOffsets(loadedInfo->rank, rewriter);
    SmallVector<OpFoldResult> maskDims = op.getMixedMaskDims();
    SmallVector<OpFoldResult> sourceOffsets = collectAddressOffsets(descriptor);
    SmallVector<OpFoldResult> sourceStrides = collectAddressStrides(descriptor);

    AddressLoweringPath loweringPath =
        classifyAddressLoweringPath(addressFeatures, *maybeIndirectInfo);
    if (loweringPath == AddressLoweringPath::WrapAware &&
        !hasAnyIndirectAccess(*maybeIndirectInfo) &&
        canProveNoWrapAccess(descriptor, *maybeIndirectInfo, loadedShape,
                             maskDims)) {
      loweringPath = AddressLoweringPath::Direct;
    }

    if (loweringPath == AddressLoweringPath::WrapAware &&
        !hasAnyIndirectAccess(*maybeIndirectInfo) &&
        succeeded(lowerRank1WrappedLoadByCopy(
            op, descriptor, *maybeIndirectInfo, loadedShape, maskDims,
            *maybeSizes, *maybeBaseMemref, alloc, rewriter))) {
      Value resultTensor = bufferization::ToTensorOp::create(
                               rewriter, op.getLoc(),
                               cast<RankedTensorType>(op.getType()), alloc,
                               /*restrict=*/true,
                               /*writable=*/true)
                               .getResult();
      rewriter.replaceOp(op, resultTensor);
      return success();
    }

    if (loweringPath == AddressLoweringPath::WrapAware) {
      maybeFillWithOther(alloc, op.getOther(), op.getLoc(), rewriter);
      if (failed(lowerWrapAwareLoad(op, descriptor, *maybeIndirectInfo,
                                    loadedShape, maskDims, *maybeBaseMemref,
                                    alloc, rewriter))) {
        return rewriter.notifyMatchFailure(op, "failed to lower wrapped load");
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

    if (loweringPath == AddressLoweringPath::Direct) {
      auto maybeSrc = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, loadedShape, *maybeSizes,
          sourceOffsets, sourceStrides,
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

    if (hasAnyIndirectAccess(*maybeIndirectInfo) && !singleIndirect) {
      if (addressFeatures.hasBlockLayout) {
        return emitTTAToMemrefError(
            op.getOperation(),
            "indirect reindex on block pointer is unsupported");
      }
      maybeFillWithOther(alloc, op.getOther(), op.getLoc(), rewriter);
      if (failed(lowerWrapAwareLoad(op, descriptor, *maybeIndirectInfo,
                                    loadedShape, maskDims, *maybeBaseMemref,
                                    alloc, rewriter))) {
        return rewriter.notifyMatchFailure(
            op, "failed to lower multi-indirect load");
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

    if (!singleIndirect) {
      return rewriter.notifyMatchFailure(op, "indirect dim is missing");
    }
    int64_t gatherDim = singleIndirect->dim;
    if (gatherDim < 0 || gatherDim >= loadedInfo->rank) {
      return rewriter.notifyMatchFailure(op, "indirect dim out of bounds");
    }
    if (addressFeatures.hasBlockLayout) {
      return emitTTAToMemrefError(
          op.getOperation(),
          "indirect reindex on block pointer is unsupported");
    }
    OpFoldResult gatherBaseOffset = descriptor.dims[gatherDim].offset;

    auto maybeIndirectIndex =
        castIndirectIndexToIndex(singleIndirect->index, op.getLoc(), rewriter);
    if (failed(maybeIndirectIndex)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index");
    }

    auto maybeOffsetSize =
        getIndirectSize(*maybeIndirectIndex, op.getLoc(), rewriter);
    if (failed(maybeOffsetSize)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index size");
    }

    if (maskDims.empty() && (op.getOther() || singleIndirect->mask)) {
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
          static_cast<bool>(singleIndirect->mask), op.getLoc(), rewriter);
    } else {
      upperBound = computeIndirectUpperBound(
          *maybeOffsetSize, maskDims, gatherDim,
          static_cast<bool>(singleIndirect->mask),
          indirectIndexType.getShape()[0], op.getLoc(), rewriter);
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

      SmallVector<OpFoldResult> gatheredOffsets(sourceOffsets.begin(),
                                                sourceOffsets.end());
      gatheredOffsets[gatherDim] = gatherOffset;

      auto maybeSrc = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, sliceShape, *maybeSizes,
          gatheredOffsets, sourceStrides, gatherDim, op.getLoc(), rw);
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

    if (singleIndirect->mask) {
      Value maskValue =
          tensor::ExtractOp::create(rewriter, op.getLoc(), singleIndirect->mask,
                                    ValueRange{iv})
              .getResult();
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
    FailureOr<AddressDescriptor> maybeDescriptor = collectAddressDescriptor(
        adaptor.getPtr(), op.getLoc(), rewriter, &collectFailureReason);
    if (failed(maybeDescriptor)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    AddressDescriptor descriptor = *maybeDescriptor;
    if (failed(validateAddressWrapBoundaries(descriptor, op.getOperation()))) {
      return failure();
    }
    AddressFeatures addressFeatures = collectAddressFeatures(descriptor);

    auto maybeIndirectInfo =
        collectIndirectInfo(descriptor, &collectFailureReason);
    if (failed(maybeIndirectInfo)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    const IndirectDimInfo *singleIndirect =
        getSingleIndirectDimInfo(*maybeIndirectInfo);

    auto maybeLoadedShape =
        resolveLoadedShape(*loadedInfo, descriptor, &collectFailureReason);
    if (failed(maybeLoadedShape)) {
      return rewriter.notifyMatchFailure(op, "failed to resolve loaded shape");
    }
    SmallVector<int64_t> loadedShape = *maybeLoadedShape;

    auto maybeSizes =
        collectAddressStaticSizes(descriptor, &collectFailureReason);
    if (failed(maybeSizes)) {
      return rewriter.notifyMatchFailure(op, "failed to resolve address sizes");
    }

    auto maybeBaseMemref = getBaseMemref(
        descriptor.base, loadedInfo->elementType, op.getLoc(), rewriter);
    if (failed(maybeBaseMemref)) {
      return rewriter.notifyMatchFailure(op, "failed to get base memref");
    }

    auto ones = getOneStrides(loadedInfo->rank, rewriter);
    auto zeros = getZeroOffsets(loadedInfo->rank, rewriter);
    SmallVector<OpFoldResult> maskDims = op.getMixedMaskDims();
    SmallVector<OpFoldResult> sourceOffsets = collectAddressOffsets(descriptor);
    SmallVector<OpFoldResult> sourceStrides = collectAddressStrides(descriptor);

    AddressLoweringPath loweringPath =
        classifyAddressLoweringPath(addressFeatures, *maybeIndirectInfo);
    if (loweringPath == AddressLoweringPath::WrapAware &&
        !hasAnyIndirectAccess(*maybeIndirectInfo) &&
        canProveNoWrapAccess(descriptor, *maybeIndirectInfo, loadedShape,
                             maskDims)) {
      loweringPath = AddressLoweringPath::Direct;
    }

    if (loweringPath == AddressLoweringPath::WrapAware &&
        !hasAnyIndirectAccess(*maybeIndirectInfo) &&
        succeeded(lowerRank1WrappedStoreByCopy(
            op, descriptor, *maybeIndirectInfo, loadedShape, maskDims,
            *maybeSizes, *maybeBaseMemref, adaptor.getValue(), rewriter))) {
      rewriter.eraseOp(op);
      return success();
    }

    if (loweringPath == AddressLoweringPath::WrapAware) {
      if (failed(lowerWrapAwareStore(op, descriptor, *maybeIndirectInfo,
                                     loadedShape, maskDims, *maybeBaseMemref,
                                     adaptor.getValue(), rewriter))) {
        return rewriter.notifyMatchFailure(op, "failed to lower wrapped store");
      }
      rewriter.eraseOp(op);
      return success();
    }

    if (loweringPath == AddressLoweringPath::Direct) {
      auto maybeDst = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, loadedShape, *maybeSizes,
          sourceOffsets, sourceStrides,
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

    if (hasAnyIndirectAccess(*maybeIndirectInfo) && !singleIndirect) {
      if (addressFeatures.hasBlockLayout) {
        return emitTTAToMemrefError(
            op.getOperation(),
            "indirect reindex on block pointer is unsupported");
      }
      if (failed(lowerWrapAwareStore(op, descriptor, *maybeIndirectInfo,
                                     loadedShape, maskDims, *maybeBaseMemref,
                                     adaptor.getValue(), rewriter))) {
        return rewriter.notifyMatchFailure(
            op, "failed to lower multi-indirect store");
      }
      rewriter.eraseOp(op);
      return success();
    }

    if (!singleIndirect) {
      return rewriter.notifyMatchFailure(op, "indirect dim is missing");
    }
    int64_t gatherDim = singleIndirect->dim;
    if (gatherDim < 0 || gatherDim >= loadedInfo->rank) {
      return rewriter.notifyMatchFailure(op, "indirect dim out of bounds");
    }
    if (addressFeatures.hasBlockLayout) {
      return emitTTAToMemrefError(
          op.getOperation(),
          "indirect reindex on block pointer is unsupported");
    }
    OpFoldResult gatherBaseOffset = descriptor.dims[gatherDim].offset;

    auto maybeIndirectIndex =
        castIndirectIndexToIndex(singleIndirect->index, op.getLoc(), rewriter);
    if (failed(maybeIndirectIndex)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index");
    }

    auto maybeOffsetSize =
        getIndirectSize(*maybeIndirectIndex, op.getLoc(), rewriter);
    if (failed(maybeOffsetSize)) {
      return rewriter.notifyMatchFailure(op, "invalid indirect index size");
    }

    if (maskDims.empty() && singleIndirect->mask) {
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
          static_cast<bool>(singleIndirect->mask), op.getLoc(), rewriter);
    } else {
      upperBound = computeIndirectUpperBound(
          *maybeOffsetSize, maskDims, gatherDim,
          static_cast<bool>(singleIndirect->mask),
          indirectIndexType.getShape()[0], op.getLoc(), rewriter);
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

      SmallVector<OpFoldResult> gatheredOffsets(sourceOffsets.begin(),
                                                sourceOffsets.end());
      gatheredOffsets[gatherDim] = gatherOffset;

      auto maybeDst = createReinterpretCast(
          *maybeBaseMemref, loadedInfo->elementType, sliceShape, *maybeSizes,
          gatheredOffsets, sourceStrides, gatherDim, op.getLoc(), rw);
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

    if (singleIndirect->mask) {
      Value maskValue =
          tensor::ExtractOp::create(rewriter, op.getLoc(), singleIndirect->mask,
                                    ValueRange{iv})
              .getResult();
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
    FailureOr<AddressDescriptor> maybeDescriptor = collectAddressDescriptor(
        adaptor.getPtr(), op.getLoc(), rewriter, &collectFailureReason);
    if (failed(maybeDescriptor)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    AddressDescriptor descriptor = *maybeDescriptor;
    if (failed(validateAddressWrapBoundaries(descriptor, op.getOperation()))) {
      return failure();
    }
    AddressFeatures addressFeatures = collectAddressFeatures(descriptor);

    auto maybeIndirectInfo =
        collectIndirectInfo(descriptor, &collectFailureReason);
    if (failed(maybeIndirectInfo)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }

    if (addressFeatures.hasBlockLayout) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "block pointer tta.atomic is unsupported");
    }
    if (addressFeatures.hasIndirect ||
        hasAnyIndirectAccess(*maybeIndirectInfo)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "indirect tta.atomic is unsupported");
    }
    if (descriptor.dims.size() != 1) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic pointer must have rank 1");
    }

    std::optional<int64_t> stride = getIntAttr(descriptor.dims[0].stride);
    if (!stride || *stride != 1) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic requires unit stride");
    }

    auto maybeBaseMemref = getBaseMemref(
        descriptor.base, op.getValue().getType(), op.getLoc(), rewriter);
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

    auto maybeTotalOffset = buildWrappedLinearizedTerm(
        descriptor.dims[0], *maybeOffset, op.getLoc(), rewriter);
    if (failed(maybeTotalOffset)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "failed to linearize tta.atomic offset");
    }
    Value totalOffset = *maybeTotalOffset;

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

    if (!isa<tta::AddrType>(op.getPtr().getType())) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic_cas pointer must be scalar addr");
    }

    std::optional<StringRef> collectFailureReason;
    FailureOr<AddressDescriptor> maybeDescriptor = collectAddressDescriptor(
        adaptor.getPtr(), op.getLoc(), rewriter, &collectFailureReason);
    if (failed(maybeDescriptor)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }
    AddressDescriptor descriptor = *maybeDescriptor;
    if (failed(validateAddressWrapBoundaries(descriptor, op.getOperation()))) {
      return failure();
    }
    AddressFeatures addressFeatures = collectAddressFeatures(descriptor);

    auto maybeIndirectInfo =
        collectIndirectInfo(descriptor, &collectFailureReason);
    if (failed(maybeIndirectInfo)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  collectFailureReason
                                      ? *collectFailureReason
                                      : "failed to collect address chain");
    }

    if (addressFeatures.hasBlockLayout) {
      return emitTTAToMemrefError(
          op.getOperation(), "block pointer tta.atomic_cas is unsupported");
    }
    if (addressFeatures.hasIndirect ||
        hasAnyIndirectAccess(*maybeIndirectInfo)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "indirect tta.atomic_cas is unsupported");
    }
    if (descriptor.dims.size() != 1) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic_cas pointer must have rank 1");
    }

    std::optional<int64_t> stride = getIntAttr(descriptor.dims[0].stride);
    if (!stride || *stride != 1) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "tta.atomic_cas requires unit stride");
    }

    auto maybeBaseMemref = getBaseMemref(
        descriptor.base, op.getValue().getType(), op.getLoc(), rewriter);
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

    auto maybeTotalOffset = buildWrappedLinearizedTerm(
        descriptor.dims[0], *maybeOffset, op.getLoc(), rewriter);
    if (failed(maybeTotalOffset)) {
      return emitTTAToMemrefError(op.getOperation(),
                                  "failed to linearize tta.atomic_cas offset");
    }
    Value totalOffset = *maybeTotalOffset;

    auto generic = memref::GenericAtomicRMWOp::create(
        rewriter, op.getLoc(), rankedMemref, totalOffset);
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
        if (isa<tta::AdvanceOp, tta::ReindexOp, tta::IndirectReindexOp,
                tta::MakeAddrOp, tta::FromTTPtrOp>(op)) {
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
