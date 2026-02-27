#include "triton-shared/Analysis/AnalysisAddress.h"

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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

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

static void clearFailureReason(std::optional<StringRef> *failureReason) {
  if (failureReason) {
    *failureReason = std::nullopt;
  }
}

static void printOpFoldResult(raw_ostream &os, OpFoldResult ofr) {
  if (Attribute attr = dyn_cast<Attribute>(ofr)) {
    attr.print(os);
    return;
  }
  cast<Value>(ofr).print(os);
}

static const char *layoutKindToString(LayoutKind kind) {
  switch (kind) {
  case LayoutKind::Strided:
    return "strided";
  case LayoutKind::Block:
    return "block";
  }
  return "unknown";
}

static std::optional<int64_t> getSplatIntConstant(Value value) {
  auto cstOp = value.getDefiningOp<arith::ConstantOp>();
  if (!cstOp) {
    return std::nullopt;
  }

  Attribute valueAttr = cstOp.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    return intAttr.getValue().getSExtValue();
  }

  auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!dense || !dense.getElementType().isIntOrIndex()) {
    return std::nullopt;
  }
  if (dense.isSplat()) {
    return dense.getSplatValue<APInt>().getSExtValue();
  }

  auto values = dense.getValues<APInt>();
  auto it = values.begin();
  if (it == values.end()) {
    return std::nullopt;
  }
  APInt first = *it;
  ++it;
  for (; it != values.end(); ++it) {
    if (*it != first) {
      return std::nullopt;
    }
  }
  return first.getSExtValue();
}

static std::optional<int64_t> getRefinerConstantFromExpr(Value value) {
  if (auto cst = getSplatIntConstant(value)) {
    return cst;
  }
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
    return getRefinerConstantFromExpr(ext.getIn());
  }
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
    return getRefinerConstantFromExpr(cast.getIn());
  }
  if (auto trunc = value.getDefiningOp<arith::TruncIOp>()) {
    return getRefinerConstantFromExpr(trunc.getIn());
  }
  if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    return getRefinerConstantFromExpr(expand.getSrc());
  }
  if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    return getRefinerConstantFromExpr(broadcast.getSrc());
  }
  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    return getRefinerConstantFromExpr(splat.getSrc());
  }
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = getRefinerConstantFromExpr(add.getLhs());
    auto rhs = getRefinerConstantFromExpr(add.getRhs());
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return *lhs + *rhs;
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = getRefinerConstantFromExpr(mul.getLhs());
    auto rhs = getRefinerConstantFromExpr(mul.getRhs());
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return *lhs * *rhs;
  }
  return std::nullopt;
}

static Value stripRefinerViewLikeAndCastOps(Value value) {
  Value current = value;
  while (true) {
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = cast.getIn();
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = trunc.getIn();
      continue;
    }
    if (auto expand = current.getDefiningOp<triton::ExpandDimsOp>()) {
      current = expand.getSrc();
      continue;
    }
    if (auto broadcast = current.getDefiningOp<triton::BroadcastOp>()) {
      current = broadcast.getSrc();
      continue;
    }
    if (auto splat = current.getDefiningOp<triton::SplatOp>()) {
      current = splat.getSrc();
      continue;
    }
    if (auto collapse = current.getDefiningOp<tensor::CollapseShapeOp>()) {
      current = collapse.getSrc();
      continue;
    }
    if (auto cast = current.getDefiningOp<tensor::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    return current;
  }
}

struct RefinerLinearExpr {
  int64_t constant = 0;
  DenseMap<Value, int64_t> terms;
};

static void addLinearTerm(RefinerLinearExpr &expr, Value leaf, int64_t coeff) {
  if (coeff == 0) {
    return;
  }
  int64_t merged = coeff;
  auto it = expr.terms.find(leaf);
  if (it != expr.terms.end()) {
    merged += it->second;
  }
  if (merged == 0) {
    expr.terms.erase(leaf);
    return;
  }
  expr.terms[leaf] = merged;
}

static LogicalResult
buildRefinerLinearExpr(Value value, int64_t scale, RefinerLinearExpr &expr,
                       std::optional<StringRef> *failureReason = nullptr) {
  if (scale == 0) {
    return success();
  }

  if (auto cst = getRefinerConstantFromExpr(value)) {
    expr.constant += scale * *cst;
    return success();
  }

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (failed(
            buildRefinerLinearExpr(add.getLhs(), scale, expr, failureReason)) ||
        failed(
            buildRefinerLinearExpr(add.getRhs(), scale, expr, failureReason))) {
      return failure();
    }
    return success();
  }

  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhsCst = getRefinerConstantFromExpr(mul.getLhs());
    auto rhsCst = getRefinerConstantFromExpr(mul.getRhs());
    if (lhsCst && rhsCst) {
      expr.constant += scale * *lhsCst * *rhsCst;
      return success();
    }
    if (lhsCst) {
      return buildRefinerLinearExpr(mul.getRhs(), scale * *lhsCst, expr,
                                    failureReason);
    }
    if (rhsCst) {
      return buildRefinerLinearExpr(mul.getLhs(), scale * *rhsCst, expr,
                                    failureReason);
    }
    setFailureReason(failureReason, "refiner_v1 unsupported muli pattern");
    return failure();
  }

  if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
    return buildRefinerLinearExpr(ext.getIn(), scale, expr, failureReason);
  }

  if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
    return buildRefinerLinearExpr(cast.getIn(), scale, expr, failureReason);
  }

  if (auto trunc = value.getDefiningOp<arith::TruncIOp>()) {
    return buildRefinerLinearExpr(trunc.getIn(), scale, expr, failureReason);
  }

  if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    return buildRefinerLinearExpr(expand.getSrc(), scale, expr, failureReason);
  }

  if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    return buildRefinerLinearExpr(broadcast.getSrc(), scale, expr,
                                  failureReason);
  }

  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    return buildRefinerLinearExpr(splat.getSrc(), scale, expr, failureReason);
  }

  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    addLinearTerm(expr, range.getResult(), scale);
    return success();
  }

  Value leaf = stripRefinerViewLikeAndCastOps(value);
  if (auto cst = getSplatIntConstant(leaf)) {
    expr.constant += scale * *cst;
    return success();
  }

  Type leafType = leaf.getType();
  if (leafType.isIntOrIndex()) {
    addLinearTerm(expr, leaf, scale);
    return success();
  }
  if (auto tensorType = dyn_cast<RankedTensorType>(leafType)) {
    if (tensorType.getElementType().isIntOrIndex()) {
      addLinearTerm(expr, leaf, scale);
      return success();
    }
  }

  setFailureReason(failureReason, "refiner_v1 unsupported expression leaf");
  return failure();
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
    setFailureReason(failureReason,
                     "indirect_index merge rank/type is unsupported");
    return failure();
  }

  auto lhsType = dyn_cast<RankedTensorType>((*lhsIndexTensor).getType());
  auto rhsType = dyn_cast<RankedTensorType>((*rhsIndexTensor).getType());
  FailureOr<RankedTensorType> maybeMergedType =
      getMerged1DTensorType(lhsType, rhsType, builder.getIndexType());
  if (failed(maybeMergedType)) {
    setFailureReason(failureReason, "indirect_index merge shape mismatch");
    return failure();
  }

  FailureOr<Value> lhsMerged =
      castTensorToType(*lhsIndexTensor, *maybeMergedType, loc, builder);
  FailureOr<Value> rhsMerged =
      castTensorToType(*rhsIndexTensor, *maybeMergedType, loc, builder);
  if (failed(lhsMerged) || failed(rhsMerged)) {
    setFailureReason(failureReason, "indirect_index merge shape mismatch");
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
    setFailureReason(failureReason, "indirect_mask shape mismatch");
    return failure();
  }

  FailureOr<Value> lhsMergedMask =
      castTensorToType(lhsMask, *maybeMergedMaskType, loc, builder);
  FailureOr<Value> rhsMergedMask =
      castTensorToType(rhsMask, *maybeMergedMaskType, loc, builder);
  if (failed(lhsMergedMask) || failed(rhsMergedMask)) {
    setFailureReason(failureReason, "indirect_mask shape mismatch");
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
    setFailureReason(failureReason, "indirect_dim out of bounds");
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
    setFailureReason(failureReason, "indirect_index must be ranked tensor");
    return failure();
  }
  if (tensorType.getRank() != 1) {
    std::optional<int64_t> nonSingletonDim;
    for (auto [dim, size] : llvm::enumerate(tensorType.getShape())) {
      if (size == 1) {
        continue;
      }
      if (ShapedType::isDynamic(size)) {
        setFailureReason(failureReason, "indirect_index must be 1D tensor");
        return failure();
      }
      if (nonSingletonDim.has_value()) {
        setFailureReason(failureReason, "indirect_index must be 1D tensor");
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
    setFailureReason(failureReason,
                     "indirect_index must use int/index elements");
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

static FailureOr<AddressDescriptor> toAddressDescriptorFromIndirectState(
    const ptrexpr::PtrState &state, Location loc, OpBuilder &builder,
    bool enableRelaxedSingleIndirectNonGatherDims,
    std::optional<StringRef> *failureReason = nullptr) {
  if (state.isEmpty() || !state.source || state.getRank() <= 0 ||
      state.isStructured()) {
    return failure();
  }

  auto baseType = dyn_cast<triton::PointerType>(state.source.getType());
  if (!baseType) {
    setFailureReason(failureReason, "unsupported address base type");
    return failure();
  }

  if (state.isBlockPtr()) {
    setFailureReason(failureReason, "indirect block pointer is unsupported");
    return failure();
  }

  SmallVector<int64_t> indirectDims;
  for (int64_t i = 0; i < state.getRank(); ++i) {
    if (!isUnstructuredOffset(state.offsets[i])) {
      continue;
    }
    indirectDims.push_back(i);
  }
  if (indirectDims.empty()) {
    return failure();
  }
  DenseMap<int64_t, Value> indirectIndexByDim;
  for (int64_t indirectDim : indirectDims) {
    auto maybeStride = getIntAttr(state.strides[indirectDim]);
    if (!maybeStride || *maybeStride != 1) {
      setFailureReason(failureReason, "indirect_dim stride must be 1");
      return failure();
    }
    if (!hasConstZero(state.shape[indirectDim])) {
      setFailureReason(failureReason,
                       "indirect shape with modulo is unsupported");
      return failure();
    }

    auto maybeIndexTensor = dyn_cast<Value>(state.offsets[indirectDim]);
    if (!maybeIndexTensor) {
      setFailureReason(failureReason, "indirect_index must be tensor value");
      return failure();
    }

    maybeIndexTensor =
        peelBroadcastForZeroStrideDims(maybeIndexTensor, indirectDim, state);

    auto normalizedIndex = normalizeIndirectIndexTensorForTTA(
        maybeIndexTensor, loc, builder, failureReason);
    if (failed(normalizedIndex)) {
      return failure();
    }
    indirectIndexByDim[indirectDim] = *normalizedIndex;
  }

  for (int64_t i = 0; i < state.getRank(); ++i) {
    if (indirectIndexByDim.count(i)) {
      continue;
    }

    auto maybeSize = getIntAttr(state.sizes[i]);
    if (!maybeSize) {
      setFailureReason(failureReason,
                       "indirect non-gather dim size is unsupported");
      return failure();
    }

    if (!hasConstZero(state.offsets[i]) || !hasConstZero(state.shape[i])) {
      setFailureReason(failureReason,
                       "indirect non-gather dim state is unsupported");
      return failure();
    }

    if (!enableRelaxedSingleIndirectNonGatherDims) {
      auto maybeStructuredStride = getIntAttr(state.strides[i]);
      bool isSingleton = *maybeSize == 1;
      bool isBroadcastStructured =
          maybeStructuredStride && *maybeStructuredStride == 0;
      if (!isSingleton && !isBroadcastStructured) {
        setFailureReason(
            failureReason,
            "indirect non-gather dim must be singleton or broadcast");
        return failure();
      }
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
    if (auto it = indirectIndexByDim.find(i); it != indirectIndexByDim.end()) {
      dim.offset = builder.getIndexAttr(0);
      dim.indirect = IndirectIndexRule{it->second, Value()};
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

FailureOr<AddressAnalysisResult> AnalysisAddress::analyzeDescriptorWithOptions(
    Value ptrLike, Location loc, OpBuilder &builder,
    const AddressAnalysisOptions &options,
    std::optional<StringRef> *failureReason) {
  clearFailureReason(failureReason);
  auto maybeDescriptor =
      analyzeSeedDescriptor(ptrLike, loc, builder, options, failureReason);
  if (failed(maybeDescriptor)) {
    return failure();
  }

  if (options.enableDescriptorDebugDump) {
    dumpDescriptor(*maybeDescriptor, "seed");
  }

  AddressDescriptor descriptor = *maybeDescriptor;
  if (options.enableRefine &&
      failed(refineDescriptor(descriptor, ptrLike, loc, builder, options,
                              failureReason))) {
    return failure();
  }
  if (options.enableDescriptorDebugDump) {
    dumpDescriptor(descriptor, "refine");
  }

  if (options.enableValidation &&
      failed(validateDescriptor(descriptor, options, failureReason))) {
    return failure();
  }
  if (options.enableDescriptorDebugDump) {
    dumpDescriptor(descriptor, "validate");
  }

  AddressAnalysisResult result;
  result.descriptor = std::move(descriptor);
  result.features = getAddressFeatures(result.descriptor);
  result.addressClass = classifyAddress(result.descriptor);
  return result;
}

FailureOr<AddressDescriptor>
AnalysisAddress::analyzeDescriptor(Value ptrLike, Location loc,
                                   OpBuilder &builder,
                                   std::optional<StringRef> *failureReason) {
  auto maybeResult = analyzeDescriptorWithOptions(
      ptrLike, loc, builder, AddressAnalysisOptions(), failureReason);
  if (failed(maybeResult)) {
    return failure();
  }
  return maybeResult->descriptor;
}

FailureOr<AddressDescriptor> AnalysisAddress::analyzeSeedDescriptor(
    Value ptrLike, Location loc, OpBuilder &builder,
    const AddressAnalysisOptions &options,
    std::optional<StringRef> *failureReason) {
  std::optional<StringRef> chainFailureReason;
  auto maybeDescriptor =
      analyzeFromTTAChain(ptrLike, loc, builder, options, &chainFailureReason);
  if (succeeded(maybeDescriptor)) {
    return maybeDescriptor;
  }
  if (chainFailureReason.has_value()) {
    setFailureReason(failureReason, *chainFailureReason);
    return failure();
  }

  return analyzeFromPtrStateSeed(ptrLike, loc, builder, options, failureReason);
}

FailureOr<AddressDescriptor>
AnalysisAddress::analyzeFromTTAChain(Value ptrLike, Location loc,
                                     OpBuilder &builder,
                                     const AddressAnalysisOptions &options,
                                     std::optional<StringRef> *failureReason) {
  if (auto imported = ptrLike.getDefiningOp<tta::FromTTPtrOp>()) {
    return analyzeSeedDescriptor(imported.getSource(), loc, builder, options,
                                 failureReason);
  }

  if (auto makeAddr = ptrLike.getDefiningOp<tta::MakeAddrOp>()) {
    return toAddressDescriptor(makeAddr, failureReason);
  }

  if (auto reindex = ptrLike.getDefiningOp<tta::ReindexOp>()) {
    auto maybeDescriptor = analyzeSeedDescriptor(
        reindex.getAddress(), loc, builder, options, failureReason);
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
    auto maybeDescriptor = analyzeSeedDescriptor(
        reindex.getAddress(), loc, builder, options, failureReason);
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
    auto maybeDescriptor = analyzeSeedDescriptor(
        advanceOp.getAddress(), loc, builder, options, failureReason);
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
    auto maybeDescriptor = analyzeSeedDescriptor(
        advanceOp.getPtr(), loc, builder, options, failureReason);
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

  return failure();
}

FailureOr<AddressDescriptor> AnalysisAddress::analyzeFromPtrStateSeed(
    Value ptrLike, Location loc, OpBuilder &builder,
    const AddressAnalysisOptions &options,
    std::optional<StringRef> *failureReason) {
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

  auto maybeDescriptor = toAddressDescriptorFromIndirectState(
      state, loc, builder, options.enableRelaxedSingleIndirectNonGatherDims,
      failureReason);
  if (failed(maybeDescriptor)) {
    return failure();
  }
  return *maybeDescriptor;
}

LogicalResult
AnalysisAddress::refineDescriptor(AddressDescriptor &descriptor, Value ptrLike,
                                  Location loc, OpBuilder &builder,
                                  const AddressAnalysisOptions &options,
                                  std::optional<StringRef> *failureReason) {
  (void)options;

  if (descriptor.layoutKind != LayoutKind::Strided) {
    return success();
  }

  SmallVector<int64_t> indirectDims;
  for (auto [dim, rule] : llvm::enumerate(descriptor.dims)) {
    if (!rule.indirect.has_value()) {
      continue;
    }
    indirectDims.push_back(static_cast<int64_t>(dim));
  }
  if (indirectDims.empty()) {
    return success();
  }
  if (indirectDims.size() != 1) {
    return success();
  }
  int64_t indirectDim = indirectDims.front();

  auto addPtr = ptrLike.getDefiningOp<triton::AddPtrOp>();
  if (!addPtr) {
    return success();
  }

  auto maybeOldStride = getIntAttr(descriptor.dims[indirectDim].stride);
  if (!maybeOldStride) {
    setFailureReason(failureReason,
                     "refiner_v1 indirect stride must be static");
    return failure();
  }

  RefinerLinearExpr offsetExpr;
  if (failed(buildRefinerLinearExpr(addPtr.getOffset(), /*scale=*/1, offsetExpr,
                                    failureReason))) {
    return failure();
  }

  Value seedLeaf = stripRefinerViewLikeAndCastOps(
      descriptor.dims[indirectDim].indirect->indexTensor);
  auto seedCoeffIt = offsetExpr.terms.find(seedLeaf);
  if (seedCoeffIt == offsetExpr.terms.end()) {
    return success();
  }
  int64_t seedCoeff = seedCoeffIt->second;
  if (seedCoeff <= 0) {
    setFailureReason(failureReason,
                     "refiner_v1 indirect coefficient must be positive");
    return failure();
  }

  if (offsetExpr.constant != 0) {
    setFailureReason(failureReason,
                     "refiner_v1 non-zero constant offset is unsupported");
    return failure();
  }

  for (const auto &term : offsetExpr.terms) {
    if (term.first == seedLeaf) {
      continue;
    }
    if (!term.first.getDefiningOp<triton::MakeRangeOp>()) {
      setFailureReason(failureReason,
                       "refiner_v1 unsupported structured expression term");
      return failure();
    }
  }

  // v1 refinement: recover the gather-axis scaling that ptrAnalysis may drop
  // in mixed structured/unstructured expressions.
  descriptor.dims[indirectDim].stride =
      builder.getIndexAttr(*maybeOldStride * seedCoeff);
  return success();
}

LogicalResult
AnalysisAddress::validateDescriptor(const AddressDescriptor &descriptor,
                                    const AddressAnalysisOptions &options,
                                    std::optional<StringRef> *failureReason) {
  (void)options;

  if (!descriptor.base) {
    setFailureReason(failureReason, "descriptor base is missing");
    return failure();
  }

  auto baseType = dyn_cast<triton::PointerType>(descriptor.base.getType());
  if (!baseType) {
    setFailureReason(failureReason, "descriptor base must be pointer");
    return failure();
  }

  if (descriptor.rank <= 0 ||
      descriptor.rank != static_cast<int64_t>(descriptor.dims.size())) {
    setFailureReason(failureReason, "descriptor rank mismatch");
    return failure();
  }

  if (descriptor.elementType != baseType.getPointeeType() ||
      descriptor.addressSpace != baseType.getAddressSpace()) {
    setFailureReason(failureReason, "descriptor pointer type mismatch");
    return failure();
  }

  if (descriptor.layoutKind == LayoutKind::Block) {
    if (!descriptor.blockLayout.has_value()) {
      setFailureReason(failureReason, "descriptor block layout missing");
      return failure();
    }
    if (descriptor.blockLayout->parentShape.size() != descriptor.dims.size() ||
        descriptor.blockLayout->order.size() != descriptor.dims.size()) {
      setFailureReason(failureReason, "descriptor block layout rank mismatch");
      return failure();
    }
  }

  return success();
}

void AnalysisAddress::dumpDescriptor(const AddressDescriptor &descriptor,
                                     StringRef stage) {
  llvm::dbgs() << "[analysis-address] " << stage
               << " layout=" << layoutKindToString(descriptor.layoutKind)
               << " rank=" << descriptor.rank << "\n";
  llvm::dbgs() << "  base: ";
  descriptor.base.print(llvm::dbgs());
  llvm::dbgs() << "\n";
  llvm::dbgs() << "  element_type: " << descriptor.elementType
               << " address_space=" << descriptor.addressSpace << "\n";

  for (auto [index, dim] : llvm::enumerate(descriptor.dims)) {
    llvm::dbgs() << "  dim[" << index << "]: size=";
    printOpFoldResult(llvm::dbgs(), dim.size);
    llvm::dbgs() << " stride=";
    printOpFoldResult(llvm::dbgs(), dim.stride);
    llvm::dbgs() << " offset=";
    printOpFoldResult(llvm::dbgs(), dim.offset);
    llvm::dbgs() << " wrap=";
    if (dim.wrapBoundary.has_value()) {
      printOpFoldResult(llvm::dbgs(), dim.wrapBoundary->boundary);
    } else {
      llvm::dbgs() << "<none>";
    }
    llvm::dbgs() << " indirect=";
    if (!dim.indirect.has_value()) {
      llvm::dbgs() << "<none>";
    } else {
      llvm::dbgs() << "{index=";
      dim.indirect->indexTensor.print(llvm::dbgs());
      llvm::dbgs() << ", mask=";
      if (dim.indirect->maskTensor) {
        dim.indirect->maskTensor.print(llvm::dbgs());
      } else {
        llvm::dbgs() << "<none>";
      }
      llvm::dbgs() << "}";
    }
    llvm::dbgs() << "\n";
  }

  if (descriptor.blockLayout.has_value()) {
    llvm::dbgs() << "  block.parent_shape=[";
    for (auto [index, ofr] :
         llvm::enumerate(descriptor.blockLayout->parentShape)) {
      if (index != 0) {
        llvm::dbgs() << ", ";
      }
      printOpFoldResult(llvm::dbgs(), ofr);
    }
    llvm::dbgs() << "]\n";
    llvm::dbgs() << "  block.order=[";
    for (auto [index, order] : llvm::enumerate(descriptor.blockLayout->order)) {
      if (index != 0) {
        llvm::dbgs() << ", ";
      }
      llvm::dbgs() << order;
    }
    llvm::dbgs() << "]\n";
  }
}

FailureOr<Value>
TTAEmitter::emitAddress(const AddressDescriptor &descriptor, Location loc,
                        OpBuilder &builder,
                        std::optional<StringRef> *failureReason) {
  AddressDescriptor baseDescriptor = descriptor;
  struct IndirectEmission {
    int64_t dim;
    Value index;
    Value mask;
  };
  SmallVector<IndirectEmission> indirects;

  for (auto [dim, rule] : llvm::enumerate(baseDescriptor.dims)) {
    if (!rule.indirect.has_value()) {
      continue;
    }
    indirects.push_back(IndirectEmission{static_cast<int64_t>(dim),
                                         rule.indirect->indexTensor,
                                         rule.indirect->maskTensor});
    rule.indirect.reset();
  }

  auto maybeBase = emitMakeAddr(baseDescriptor, loc, builder, failureReason);
  if (failed(maybeBase)) {
    setFailureReason(failureReason, "emit make_addr failed");
    return failure();
  }
  if (indirects.empty()) {
    return *maybeBase;
  }

  Value current = *maybeBase;
  for (const IndirectEmission &indirect : indirects) {
    auto normalizedIndex = normalizeIndirectIndexTensorForTTA(
        indirect.index, loc, builder, failureReason);
    if (failed(normalizedIndex)) {
      return failure();
    }
    if (indirect.mask) {
      current = tta::IndirectReindexOp::create(builder, loc, current,
                                               *normalizedIndex, indirect.mask,
                                               indirect.dim)
                    .getResult();
      continue;
    }
    current = tta::IndirectReindexOp::create(builder, loc, current,
                                             *normalizedIndex, indirect.dim)
                  .getResult();
  }
  return current;
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
