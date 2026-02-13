#include "triton-shared/AnalysisAddress/AnalysisAddress.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace triton {
namespace address {

namespace {

static void setFailureReason(std::optional<StringRef> *failureReason,
                             StringRef reason) {
  if (failureReason &&
      (!failureReason->has_value() || (*failureReason)->empty())) {
    *failureReason = reason;
  }
}

static FailureOr<AddressDescriptor>
toAddressDescriptor(const ptrexpr::PtrState &state,
                    std::optional<StringRef> *failureReason = nullptr) {
  if (state.isEmpty() || !state.source || state.getRank() <= 0) {
    setFailureReason(failureReason, "invalid_ptr_state");
    return failure();
  }

  auto baseType = dyn_cast<triton::PointerType>(state.source.getType());
  if (!baseType) {
    setFailureReason(failureReason, "unsupported address base type");
    return failure();
  }

  AddressDescriptor descriptor;
  descriptor.base = state.source;
  descriptor.elementType = baseType.getPointeeType();
  descriptor.addressSpace = baseType.getAddressSpace();
  descriptor.rank = state.getRank();
  descriptor.layoutKind =
      state.isBlockPtr() ? LayoutKind::Block : LayoutKind::Strided;

  descriptor.dims.reserve(state.getRank());
  for (int32_t i = 0; i < state.getRank(); ++i) {
    DimRule dim;
    dim.size = state.sizes[i];
    dim.stride = state.strides[i];
    dim.offset = state.offsets[i];
    if (!state.isBlockPtr() && !hasConstZero(state.shape[i])) {
      dim.wrapBoundary = WrapBoundary{state.shape[i]};
    }
    descriptor.dims.push_back(std::move(dim));
  }

  if (state.isBlockPtr()) {
    BlockLayout blockLayout;
    blockLayout.parentShape = SmallVector<OpFoldResult>(state.shape);
    blockLayout.order = SmallVector<int32_t>(state.order);
    descriptor.blockLayout = std::move(blockLayout);
  }

  return descriptor;
}

static FailureOr<AddressDescriptor>
toAddressDescriptor(tta::MakeAddrOp makeAddr,
                    std::optional<StringRef> *failureReason = nullptr) {
  auto baseType = dyn_cast<triton::PointerType>(makeAddr.getBase().getType());
  if (!baseType) {
    setFailureReason(failureReason, "unsupported address base type");
    return failure();
  }

  auto sizes = makeAddr.getMixedSizes();
  auto strides = makeAddr.getMixedStrides();
  auto offsets = makeAddr.getMixedOffsets();
  auto layout = makeAddr.getMixedLayout();
  StringRef layoutKind = makeAddr.getLayoutKindString();
  bool isBlock = layoutKind == "block";

  if (layoutKind != "strided" && layoutKind != "block") {
    setFailureReason(failureReason, "unsupported layout_kind");
    return failure();
  }

  if (sizes.size() != strides.size() || sizes.size() != offsets.size() ||
      sizes.size() != layout.size()) {
    setFailureReason(failureReason, "make_addr rank mismatch");
    return failure();
  }

  AddressDescriptor descriptor;
  descriptor.base = makeAddr.getBase();
  descriptor.elementType = baseType.getPointeeType();
  descriptor.addressSpace = baseType.getAddressSpace();
  descriptor.rank = sizes.size();
  descriptor.layoutKind = isBlock ? LayoutKind::Block : LayoutKind::Strided;

  descriptor.dims.reserve(descriptor.rank);
  for (auto [idx, size] : llvm::enumerate(sizes)) {
    DimRule dim;
    dim.size = size;
    dim.stride = strides[idx];
    dim.offset = offsets[idx];
    if (!isBlock && !hasConstZero(layout[idx])) {
      dim.wrapBoundary = WrapBoundary{layout[idx]};
    }
    descriptor.dims.push_back(std::move(dim));
  }

  if (isBlock) {
    SmallVector<int32_t> order = makeAddr.getLayoutOrder();
    if (order.size() != layout.size()) {
      setFailureReason(failureReason, "block layout order rank mismatch");
      return failure();
    }
    BlockLayout blockLayout;
    blockLayout.parentShape = SmallVector<OpFoldResult>(layout);
    blockLayout.order = std::move(order);
    descriptor.blockLayout = std::move(blockLayout);
  }

  return descriptor;
}

static LogicalResult applyAdvanceOffsets(
    AddressDescriptor &descriptor, ArrayRef<OpFoldResult> deltas, Location loc,
    OpBuilder &builder, std::optional<StringRef> *failureReason = nullptr) {
  if (static_cast<int64_t>(deltas.size()) != descriptor.rank) {
    setFailureReason(failureReason, "advance rank mismatch");
    return failure();
  }

  for (auto [index, delta] : llvm::enumerate(deltas)) {
    descriptor.dims[index].offset =
        addOFRs(descriptor.dims[index].offset, delta, loc, builder);
  }

  return success();
}

static LogicalResult
applyReindex(AddressDescriptor &descriptor, tta::ReindexOp reindex,
             Location loc, OpBuilder &builder,
             std::optional<StringRef> *failureReason = nullptr) {
  auto offsets = reindex.getMixedOffsets();
  if (static_cast<int64_t>(offsets.size()) != descriptor.rank) {
    setFailureReason(failureReason, "reindex rank mismatch");
    return failure();
  }

  for (auto [index, offset] : llvm::enumerate(offsets)) {
    descriptor.dims[index].offset =
        addOFRs(descriptor.dims[index].offset, offset, loc, builder);
  }

  return success();
}

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
                                         Location loc, OpBuilder &builder) {
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

  return tensor::CastOp::create(builder, loc, targetType, value).getResult();
}

static FailureOr<Value> castIndirectIndexToIndex(Value indirectIndex,
                                                 Location loc,
                                                 OpBuilder &builder) {
  auto tensorType = dyn_cast<RankedTensorType>(indirectIndex.getType());
  if (!tensorType || tensorType.getRank() != 1) {
    return failure();
  }

  if (tensorType.getElementType().isIndex()) {
    return indirectIndex;
  }

  auto intType = dyn_cast<IntegerType>(tensorType.getElementType());
  if (!intType || (intType.getWidth() != 32 && intType.getWidth() != 64)) {
    return failure();
  }

  auto indexTensorType =
      RankedTensorType::get(tensorType.getShape(), builder.getIndexType());
  return arith::IndexCastOp::create(builder, loc, indexTensorType,
                                    indirectIndex)
      .getResult();
}

static FailureOr<Value>
mergeIndirectIndex(Value lhsIndex, Value rhsIndex, Location loc,
                   OpBuilder &builder,
                   std::optional<StringRef> *failureReason = nullptr) {
  FailureOr<Value> lhsIndexTensor =
      castIndirectIndexToIndex(lhsIndex, loc, builder);
  FailureOr<Value> rhsIndexTensor =
      castIndirectIndexToIndex(rhsIndex, loc, builder);
  if (failed(lhsIndexTensor) || failed(rhsIndexTensor)) {
    if (failureReason) {
      *failureReason =
          StringRef("indirect_index merge rank/type is unsupported");
    }
    return failure();
  }

  auto lhsType = dyn_cast<RankedTensorType>((*lhsIndexTensor).getType());
  auto rhsType = dyn_cast<RankedTensorType>((*rhsIndexTensor).getType());
  FailureOr<RankedTensorType> maybeMergedType =
      getMerged1DTensorType(lhsType, rhsType, builder.getIndexType());
  if (failed(maybeMergedType)) {
    if (failureReason) {
      *failureReason = StringRef("indirect_index merge shape mismatch");
    }
    return failure();
  }

  FailureOr<Value> lhsMerged =
      castTensorToType(*lhsIndexTensor, *maybeMergedType, loc, builder);
  FailureOr<Value> rhsMerged =
      castTensorToType(*rhsIndexTensor, *maybeMergedType, loc, builder);
  if (failed(lhsMerged) || failed(rhsMerged)) {
    if (failureReason) {
      *failureReason = StringRef("indirect_index merge shape mismatch");
    }
    return failure();
  }

  return arith::AddIOp::create(builder, loc, *lhsMerged, *rhsMerged)
      .getResult();
}

static FailureOr<Value>
mergeIndirectMask(Value lhsMask, Value rhsMask, Location loc,
                  OpBuilder &builder,
                  std::optional<StringRef> *failureReason = nullptr) {
  auto lhsMaskType = dyn_cast<RankedTensorType>(lhsMask.getType());
  auto rhsMaskType = dyn_cast<RankedTensorType>(rhsMask.getType());
  FailureOr<RankedTensorType> maybeMergedMaskType =
      getMerged1DTensorType(lhsMaskType, rhsMaskType, builder.getI1Type());
  if (failed(maybeMergedMaskType)) {
    if (failureReason) {
      *failureReason = StringRef("indirect_mask shape mismatch");
    }
    return failure();
  }

  FailureOr<Value> lhsMergedMask =
      castTensorToType(lhsMask, *maybeMergedMaskType, loc, builder);
  FailureOr<Value> rhsMergedMask =
      castTensorToType(rhsMask, *maybeMergedMaskType, loc, builder);
  if (failed(lhsMergedMask) || failed(rhsMergedMask)) {
    if (failureReason) {
      *failureReason = StringRef("indirect_mask shape mismatch");
    }
    return failure();
  }

  return arith::AndIOp::create(builder, loc, *lhsMergedMask, *rhsMergedMask)
      .getResult();
}

static LogicalResult applyIndirectReindex(
    AddressDescriptor &descriptor, tta::IndirectReindexOp reindex, Location loc,
    OpBuilder &builder, std::optional<StringRef> *failureReason = nullptr) {
  int32_t indirectDim = reindex.getIndirectDimAttr().getInt();
  if (indirectDim < 0 || indirectDim >= descriptor.rank) {
    if (failureReason) {
      *failureReason = StringRef("indirect_dim out of bounds");
    }
    return failure();
  }

  DimRule &dim = descriptor.dims[indirectDim];
  if (!dim.indirect.has_value()) {
    dim.indirect =
        IndirectIndexRule{reindex.getIndirectIndex(), reindex.getMask()};
    return success();
  }

  FailureOr<Value> maybeMergedIndex =
      mergeIndirectIndex(dim.indirect->indexTensor, reindex.getIndirectIndex(),
                         loc, builder, failureReason);
  if (failed(maybeMergedIndex)) {
    return failure();
  }

  Value mergedMask = dim.indirect->maskTensor;
  Value rhsMask = reindex.getMask();
  if (!mergedMask) {
    mergedMask = rhsMask;
  } else if (rhsMask) {
    FailureOr<Value> maybeMergedMask =
        mergeIndirectMask(mergedMask, rhsMask, loc, builder, failureReason);
    if (failed(maybeMergedMask)) {
      return failure();
    }
    mergedMask = *maybeMergedMask;
  }

  dim.indirect = IndirectIndexRule{*maybeMergedIndex, mergedMask};
  return success();
}

static bool isUnstructuredOffset(OpFoldResult offset) {
  auto value = dyn_cast<Value>(offset);
  return value && isa<ShapedType>(value.getType());
}

static FailureOr<Value> normalizeIndirectIndexTensorForTTA(
    Value indexTensor, Location loc, OpBuilder &builder,
    std::optional<StringRef> *failureReason = nullptr) {
  auto tensorType = dyn_cast<RankedTensorType>(indexTensor.getType());
  if (!tensorType) {
    if (failureReason) {
      *failureReason = StringRef("indirect_index must be ranked tensor");
    }
    return failure();
  }
  if (tensorType.getRank() != 1) {
    std::optional<int64_t> nonSingletonDim;
    for (auto [dim, size] : llvm::enumerate(tensorType.getShape())) {
      if (size == 1) {
        continue;
      }
      if (ShapedType::isDynamic(size)) {
        if (failureReason) {
          *failureReason = StringRef("indirect_index must be 1D tensor");
        }
        return failure();
      }
      if (nonSingletonDim.has_value()) {
        if (failureReason) {
          *failureReason = StringRef("indirect_index must be 1D tensor");
        }
        return failure();
      }
      nonSingletonDim = static_cast<int64_t>(dim);
    }
    if (!nonSingletonDim.has_value()) {
      nonSingletonDim = 0;
    }

    int64_t collapsedSize = tensorType.getShape()[*nonSingletonDim];
    auto collapsedType =
        RankedTensorType::get({collapsedSize}, tensorType.getElementType());
    SmallVector<ReassociationIndices> reassociation(1);
    reassociation.front().reserve(tensorType.getRank());
    for (int64_t dim = 0; dim < tensorType.getRank(); ++dim) {
      reassociation.front().push_back(static_cast<int32_t>(dim));
    }
    indexTensor = tensor::CollapseShapeOp::create(builder, loc, collapsedType,
                                                  indexTensor, reassociation)
                      .getResult();
    tensorType = collapsedType;
  }

  Type elementType = tensorType.getElementType();
  if (elementType.isInteger(32) || elementType.isInteger(64)) {
    return indexTensor;
  }

  auto i64TensorType =
      RankedTensorType::get(tensorType.getShape(), builder.getI64Type());
  if (elementType.isIndex()) {
    return arith::IndexCastOp::create(builder, loc, i64TensorType, indexTensor)
        .getResult();
  }

  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType) {
    if (failureReason) {
      *failureReason = StringRef("indirect_index must use int/index elements");
    }
    return failure();
  }

  if (intType.getWidth() < 64) {
    return arith::ExtSIOp::create(builder, loc, i64TensorType, indexTensor)
        .getResult();
  }
  return arith::TruncIOp::create(builder, loc, i64TensorType, indexTensor)
      .getResult();
}

static Value peelBroadcastForZeroStrideDims(Value indexTensor,
                                            int64_t indirectDim,
                                            const ptrexpr::PtrState &state) {
  Value current = indexTensor;
  while (auto broadcast = current.getDefiningOp<triton::BroadcastOp>()) {
    auto srcType = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
    auto dstType = dyn_cast<RankedTensorType>(broadcast.getType());
    if (!srcType || !dstType || srcType.getRank() != dstType.getRank() ||
        srcType.getRank() != state.getRank()) {
      break;
    }

    bool canPeel = true;
    for (int64_t dim = 0; dim < dstType.getRank(); ++dim) {
      int64_t srcSize = srcType.getShape()[dim];
      int64_t dstSize = dstType.getShape()[dim];
      if (srcSize == dstSize) {
        continue;
      }

      auto maybeStride = getIntAttr(state.strides[dim]);
      if (ShapedType::isDynamic(srcSize) || ShapedType::isDynamic(dstSize) ||
          srcSize != 1 || dim == indirectDim || !maybeStride ||
          *maybeStride != 0) {
        canPeel = false;
        break;
      }
    }

    if (!canPeel) {
      break;
    }
    current = broadcast.getSrc();
  }

  return current;
}

static FailureOr<AddressDescriptor> toAddressDescriptorFromSingleIndirectState(
    const ptrexpr::PtrState &state, Location loc, OpBuilder &builder,
    std::optional<StringRef> *failureReason = nullptr) {
  if (state.isEmpty() || !state.source || state.getRank() <= 0 ||
      state.isStructured()) {
    return failure();
  }

  auto baseType = dyn_cast<triton::PointerType>(state.source.getType());
  if (!baseType) {
    if (failureReason) {
      *failureReason = StringRef("unsupported address base type");
    }
    return failure();
  }

  if (state.isBlockPtr()) {
    if (failureReason) {
      *failureReason = StringRef("indirect block pointer is unsupported");
    }
    return failure();
  }

  int64_t indirectDim = 0;
  std::optional<int64_t> maybeIndirectDim;
  for (int64_t i = 0; i < state.getRank(); ++i) {
    if (!isUnstructuredOffset(state.offsets[i])) {
      continue;
    }
    if (maybeIndirectDim.has_value()) {
      if (failureReason) {
        *failureReason = StringRef("mixed_indirect_dim in address chain");
      }
      return failure();
    }
    maybeIndirectDim = i;
  }
  if (!maybeIndirectDim.has_value()) {
    return failure();
  }
  indirectDim = *maybeIndirectDim;

  auto maybeStride = getIntAttr(state.strides[indirectDim]);
  if (!maybeStride || *maybeStride != 1) {
    if (failureReason) {
      *failureReason = StringRef("indirect_dim stride must be 1");
    }
    return failure();
  }
  if (!hasConstZero(state.shape[indirectDim])) {
    if (failureReason) {
      *failureReason = StringRef("indirect shape with modulo is unsupported");
    }
    return failure();
  }

  auto maybeIndexTensor = dyn_cast<Value>(state.offsets[indirectDim]);
  if (!maybeIndexTensor) {
    if (failureReason) {
      *failureReason = StringRef("indirect_index must be tensor value");
    }
    return failure();
  }

  maybeIndexTensor =
      peelBroadcastForZeroStrideDims(maybeIndexTensor, indirectDim, state);

  auto normalizedIndex = normalizeIndirectIndexTensorForTTA(
      maybeIndexTensor, loc, builder, failureReason);
  if (failed(normalizedIndex)) {
    return failure();
  }

  for (int64_t i = 0; i < state.getRank(); ++i) {
    if (i == indirectDim) {
      continue;
    }

    auto maybeSize = getIntAttr(state.sizes[i]);
    if (!maybeSize) {
      if (failureReason) {
        *failureReason =
            StringRef("indirect non-gather dim size is unsupported");
      }
      return failure();
    }

    if (!hasConstZero(state.offsets[i]) || !hasConstZero(state.shape[i])) {
      if (failureReason) {
        *failureReason =
            StringRef("indirect non-gather dim state is unsupported");
      }
      return failure();
    }

    auto maybeStructuredStride = getIntAttr(state.strides[i]);
    bool isSingleton = *maybeSize == 1;
    bool isBroadcastStructured =
        maybeStructuredStride && *maybeStructuredStride == 0;
    if (!isSingleton && !isBroadcastStructured) {
      if (failureReason) {
        *failureReason =
            StringRef("indirect non-gather dim must be singleton or broadcast");
      }
      return failure();
    }
  }

  AddressDescriptor descriptor;
  descriptor.base = state.source;
  descriptor.elementType = baseType.getPointeeType();
  descriptor.addressSpace = baseType.getAddressSpace();
  descriptor.rank = state.getRank();
  descriptor.layoutKind = LayoutKind::Strided;
  descriptor.dims.reserve(descriptor.rank);

  for (int64_t i = 0; i < descriptor.rank; ++i) {
    DimRule dim;
    dim.size = state.sizes[i];
    dim.stride = state.strides[i];
    dim.offset = state.offsets[i];
    if (!hasConstZero(state.shape[i])) {
      dim.wrapBoundary = WrapBoundary{state.shape[i]};
    }
    if (i == indirectDim) {
      dim.offset = builder.getIndexAttr(0);
      dim.indirect = IndirectIndexRule{*normalizedIndex, Value()};
    }
    descriptor.dims.push_back(std::move(dim));
  }

  return descriptor;
}

} // namespace

AddressFeatures getAddressFeatures(const AddressDescriptor &descriptor) {
  AddressFeatures features;
  features.hasBlockLayout = descriptor.layoutKind == LayoutKind::Block;
  features.hasIndirect = llvm::any_of(descriptor.dims, [](const DimRule &dim) {
    return dim.indirect.has_value();
  });
  features.hasWrapBoundary =
      llvm::any_of(descriptor.dims, [](const DimRule &dim) {
        return dim.wrapBoundary.has_value();
      });
  return features;
}

AddressClass classifyAddress(const AddressDescriptor &descriptor) {
  AddressFeatures features = getAddressFeatures(descriptor);
  if (features.hasBlockLayout) {
    return AddressClass::BlockPtr;
  }
  if (features.hasIndirect) {
    return AddressClass::MixedPtr;
  }
  if (features.hasWrapBoundary) {
    return AddressClass::SplitPtr;
  }
  return AddressClass::StructuredPtr;
}

FailureOr<AddressDescriptor>
AnalysisAddress::analyzeDescriptor(Value ptrLike, Location loc,
                                   OpBuilder &builder,
                                   std::optional<StringRef> *failureReason) {
  if (auto imported = ptrLike.getDefiningOp<tta::FromTTPtrOp>()) {
    return analyzeDescriptor(imported.getSource(), loc, builder, failureReason);
  }

  if (auto makeAddr = ptrLike.getDefiningOp<tta::MakeAddrOp>()) {
    return toAddressDescriptor(makeAddr, failureReason);
  }

  if (auto reindex = ptrLike.getDefiningOp<tta::ReindexOp>()) {
    auto maybeDescriptor =
        analyzeDescriptor(reindex.getAddress(), loc, builder, failureReason);
    if (failed(maybeDescriptor)) {
      return failure();
    }

    if (failed(applyReindex(*maybeDescriptor, reindex, loc, builder,
                            failureReason))) {
      return failure();
    }
    return maybeDescriptor;
  }

  if (auto reindex = ptrLike.getDefiningOp<tta::IndirectReindexOp>()) {
    auto maybeDescriptor =
        analyzeDescriptor(reindex.getAddress(), loc, builder, failureReason);
    if (failed(maybeDescriptor)) {
      return failure();
    }

    if (failed(applyIndirectReindex(*maybeDescriptor, reindex, loc, builder,
                                    failureReason))) {
      return failure();
    }
    return maybeDescriptor;
  }

  if (auto advanceOp = ptrLike.getDefiningOp<tta::AdvanceOp>()) {
    auto maybeDescriptor =
        analyzeDescriptor(advanceOp.getAddress(), loc, builder, failureReason);
    if (failed(maybeDescriptor)) {
      return failure();
    }

    if (failed(applyAdvanceOffsets(*maybeDescriptor, advanceOp.getMixedDeltas(),
                                   loc, builder, failureReason))) {
      return failure();
    }
    return maybeDescriptor;
  }

  if (auto advanceOp = ptrLike.getDefiningOp<triton::AdvanceOp>()) {
    auto maybeDescriptor =
        analyzeDescriptor(advanceOp.getPtr(), loc, builder, failureReason);
    if (failed(maybeDescriptor)) {
      return failure();
    }

    SmallVector<OpFoldResult> deltas;
    deltas.reserve(advanceOp.getOffsets().size());
    for (Value delta : advanceOp.getOffsets()) {
      if (delta.getType().isIndex()) {
        deltas.push_back(delta);
        continue;
      }
      if (!delta.getType().isIntOrIndex()) {
        setFailureReason(failureReason, "advance delta type unsupported");
        return failure();
      }
      Value cast = arith::IndexCastOp::create(builder, loc,
                                              builder.getIndexType(), delta)
                       .getResult();
      deltas.push_back(cast);
    }

    if (static_cast<int64_t>(deltas.size()) != maybeDescriptor->rank) {
      setFailureReason(failureReason, "advance rank mismatch");
      return failure();
    }

    for (auto [index, delta] : llvm::enumerate(deltas)) {
      OpFoldResult scaledDelta =
          mulOFRs(delta, maybeDescriptor->dims[index].stride, loc, builder);
      maybeDescriptor->dims[index].offset = addOFRs(
          maybeDescriptor->dims[index].offset, scaledDelta, loc, builder);
    }

    return maybeDescriptor;
  }

  ptrexpr::PtrState state;
  if (auto makeTensorPtr = ptrLike.getDefiningOp<triton::MakeTensorPtrOp>()) {
    if (failed(ptrAnalysis.visitOperandMakeTensorPtr(makeTensorPtr, state, loc,
                                                     builder))) {
      setFailureReason(failureReason, "ptr_expr_analysis_failed");
      return failure();
    }
  } else if (auto makeTPtr = ptrLike.getDefiningOp<tts::MakeTensorPtrOp>()) {
    state.source = makeTPtr.getBase();
    state.offsets = makeTPtr.getMixedOffsets();
    state.sizes = makeTPtr.getMixedSizes();
    state.strides = makeTPtr.getMixedStrides();
    state.shape = makeTPtr.getMixedShape();
    state.order = SmallVector<int32_t>(makeTPtr.getOrder());
  } else if (failed(ptrAnalysis.visitOperand(ptrLike, state, loc, builder))) {
    setFailureReason(failureReason, "ptr_expr_analysis_failed");
    return failure();
  }

  if (state.isStructured()) {
    return toAddressDescriptor(state, failureReason);
  }

  auto maybeDescriptor = toAddressDescriptorFromSingleIndirectState(
      state, loc, builder, failureReason);
  if (failed(maybeDescriptor)) {
    return failure();
  }
  return *maybeDescriptor;
}

FailureOr<Value>
TTAEmitter::emitAddress(const AddressDescriptor &descriptor, Location loc,
                        OpBuilder &builder,
                        std::optional<StringRef> *failureReason) {
  AddressDescriptor baseDescriptor = descriptor;
  std::optional<int64_t> indirectDim;
  Value indirectIndex;
  Value indirectMask;

  for (auto [dim, rule] : llvm::enumerate(baseDescriptor.dims)) {
    if (!rule.indirect.has_value()) {
      continue;
    }

    if (indirectDim.has_value()) {
      if (failureReason) {
        *failureReason = StringRef("multiple indirect dims are unsupported");
      }
      return failure();
    }

    indirectDim = static_cast<int64_t>(dim);
    indirectIndex = rule.indirect->indexTensor;
    indirectMask = rule.indirect->maskTensor;
    rule.indirect.reset();
  }

  auto maybeBase = emitMakeAddr(baseDescriptor, loc, builder, failureReason);
  if (failed(maybeBase)) {
    setFailureReason(failureReason, "emit make_addr failed");
    return failure();
  }
  if (!indirectDim.has_value()) {
    return *maybeBase;
  }

  auto normalizedIndex = normalizeIndirectIndexTensorForTTA(
      indirectIndex, loc, builder, failureReason);
  if (failed(normalizedIndex)) {
    return failure();
  }

  if (indirectMask) {
    return tta::IndirectReindexOp::create(builder, loc, *maybeBase,
                                          *normalizedIndex, indirectMask,
                                          *indirectDim)
        .getResult();
  }
  return tta::IndirectReindexOp::create(builder, loc, *maybeBase,
                                        *normalizedIndex, *indirectDim)
      .getResult();
}

FailureOr<Value>
TTAEmitter::emitMakeAddr(const AddressDescriptor &descriptor, Location loc,
                         OpBuilder &builder,
                         std::optional<StringRef> *failureReason) {
  auto baseType = dyn_cast<triton::PointerType>(descriptor.base.getType());
  if (!baseType) {
    setFailureReason(failureReason, "emit_make_addr base must be pointer");
    return failure();
  }

  if (descriptor.rank <= 0 ||
      descriptor.rank != static_cast<int64_t>(descriptor.dims.size())) {
    setFailureReason(failureReason, "emit_make_addr rank mismatch");
    return failure();
  }

  if (descriptor.elementType != baseType.getPointeeType() ||
      descriptor.addressSpace != baseType.getAddressSpace()) {
    setFailureReason(failureReason, "emit_make_addr type mismatch");
    return failure();
  }

  SmallVector<int64_t> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> layout;
  DictionaryAttr layoutPayload;
  StringRef layoutKind;
  sizes.reserve(descriptor.rank);
  strides.reserve(descriptor.rank);
  offsets.reserve(descriptor.rank);
  layout.reserve(descriptor.rank);

  for (const DimRule &dim : descriptor.dims) {
    auto sizeAttr = getIntAttr(dim.size);
    if (!sizeAttr.has_value() || dim.indirect.has_value()) {
      setFailureReason(failureReason,
                       "emit_make_addr requires static sizes and no indirect");
      return failure();
    }
    sizes.push_back(sizeAttr.value());
    strides.push_back(dim.stride);
    offsets.push_back(dim.offset);
  }

  if (descriptor.layoutKind == LayoutKind::Block) {
    if (!descriptor.blockLayout.has_value()) {
      setFailureReason(failureReason, "emit_make_addr missing block layout");
      return failure();
    }
    if (descriptor.blockLayout->parentShape.size() != descriptor.rank ||
        descriptor.blockLayout->order.size() != descriptor.rank) {
      setFailureReason(failureReason,
                       "emit_make_addr block layout rank mismatch");
      return failure();
    }
    layout = SmallVector<OpFoldResult>(descriptor.blockLayout->parentShape);
    NamedAttrList payloadAttrs;
    payloadAttrs.append(builder.getNamedAttr(
        "order", builder.getDenseI32ArrayAttr(descriptor.blockLayout->order)));
    layoutPayload = DictionaryAttr::get(builder.getContext(), payloadAttrs);
    layoutKind = "block";
  } else {
    for (const DimRule &dim : descriptor.dims) {
      if (dim.wrapBoundary.has_value()) {
        layout.push_back(dim.wrapBoundary->boundary);
      } else {
        layout.push_back(builder.getIndexAttr(0));
      }
    }
    layoutKind = "strided";
  }

  auto makeAddr =
      tta::MakeAddrOp::create(builder, loc, descriptor.base, sizes, strides,
                              offsets, layoutKind, layout, layoutPayload);
  return makeAddr.getResult();
}

FailureOr<SmallVector<OpFoldResult>>
TTAEmitter::analyzeMaskDims(Value mask, Location loc, OpBuilder &builder,
                            bool useUnsafeMask) {
  if (!mask) {
    return SmallVector<OpFoldResult>{};
  }

  mlir::triton::MaskState mstate(useUnsafeMask);
  if (failed(mstate.parse(mask, loc, builder))) {
    return failure();
  }

  if (!mstate.getUnstructuredMasks().empty()) {
    return failure();
  }

  return SmallVector<OpFoldResult>(mstate.dims.begin(), mstate.dims.end());
}

FailureOr<Value> TTAEmitter::getScalarOther(Value other, Location loc,
                                            OpBuilder &builder) {
  if (!other) {
    return Value();
  }

  Value scalar = tts::utils::getScalarValue(other, loc, builder);
  if (!scalar) {
    return failure();
  }

  return scalar;
}

} // namespace address
} // namespace triton
} // namespace mlir
