#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <optional>

namespace mlir {
namespace tta {

namespace {
struct ImportedAddressInfo {
  int64_t rank;
  Type elementType;
  int addressSpace;
};

struct LoadedAddressInfo {
  int64_t rank;
  Type elementType;
  SmallVector<int64_t> shape;
};

std::optional<ImportedAddressInfo> getImportedAddressInfo(Type type) {
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementPtrType =
        dyn_cast<triton::PointerType>(ptrTensorType.getElementType());
    if (!elementPtrType) {
      return std::nullopt;
    }

    ImportedAddressInfo info{ptrTensorType.getRank(),
                             elementPtrType.getPointeeType(),
                             elementPtrType.getAddressSpace()};
    return info;
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    Type pointeeType = ptrType.getPointeeType();
    Type elementType = pointeeType;
    int64_t rank = 1;
    if (auto pointeeTensorType = dyn_cast<RankedTensorType>(pointeeType)) {
      rank = pointeeTensorType.getRank();
      elementType = pointeeTensorType.getElementType();
    }

    ImportedAddressInfo info{rank, elementType, ptrType.getAddressSpace()};
    return info;
  }

  return std::nullopt;
}

std::optional<int64_t> getAddressRank(Type type) {
  if (auto addrType = dyn_cast<AddrType>(type)) {
    return addrType.getRank();
  }

  return std::nullopt;
}

std::optional<LoadedAddressInfo> getLoadedAddressInfo(Type type) {
  if (auto addrType = dyn_cast<AddrType>(type)) {
    SmallVector<int64_t> shape(addrType.getRank(), ShapedType::kDynamic);
    return LoadedAddressInfo{addrType.getRank(), addrType.getElementType(),
                             std::move(shape)};
  }

  return std::nullopt;
}

std::optional<Type> getAtomicElementType(Type type) {
  if (auto addrType = dyn_cast<AddrType>(type)) {
    return addrType.getElementType();
  }

  return std::nullopt;
}

bool isValidAtomicKind(StringRef kind) {
  static constexpr std::array<StringLiteral, 8> kKinds = {
      "add", "and", "or", "xor", "max", "min", "xchg", "fadd"};
  return llvm::is_contained(kKinds, kind);
}

bool areAllZero(ArrayRef<OpFoldResult> values) {
  return llvm::all_of(
      values, [](OpFoldResult value) { return isConstantIntValue(value, 0); });
}

Value materializeIndexValue(OpFoldResult ofr, Location loc,
                            PatternRewriter &rewriter) {
  if (auto value = dyn_cast<Value>(ofr)) {
    if (value.getType().isIndex()) {
      return value;
    }
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                      value)
        .getResult();
  }

  auto intValue = cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
  return arith::ConstantOp::create(rewriter, loc,
                                   rewriter.getIndexAttr(intValue))
      .getResult();
}

SmallVector<OpFoldResult> composePairwiseAdd(ArrayRef<OpFoldResult> lhs,
                                             ArrayRef<OpFoldResult> rhs,
                                             Location loc,
                                             PatternRewriter &rewriter) {
  SmallVector<OpFoldResult> composed;
  composed.reserve(lhs.size());

  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    auto lhsConstant = getConstantIntValue(lhsValue);
    auto rhsConstant = getConstantIntValue(rhsValue);
    if (lhsConstant && rhsConstant) {
      composed.push_back(rewriter.getIndexAttr(*lhsConstant + *rhsConstant));
      continue;
    }

    if (lhsConstant && *lhsConstant == 0) {
      composed.push_back(rhsValue);
      continue;
    }

    if (rhsConstant && *rhsConstant == 0) {
      composed.push_back(lhsValue);
      continue;
    }

    Value lhsIndex = materializeIndexValue(lhsValue, loc, rewriter);
    Value rhsIndex = materializeIndexValue(rhsValue, loc, rewriter);
    composed.push_back(
        arith::AddIOp::create(rewriter, loc, lhsIndex, rhsIndex).getResult());
  }

  return composed;
}

ReindexOp createReindexWithSameIndirection(PatternRewriter &rewriter,
                                           Location loc, Value address,
                                           Value indirectIndex, Value mask,
                                           IntegerAttr indirectDimAttr,
                                           ArrayRef<OpFoldResult> offsets) {
  if (!indirectIndex) {
    return ReindexOp::create(rewriter, loc, address, offsets);
  }

  int64_t indirectDim = indirectDimAttr.getInt();
  if (mask) {
    return ReindexOp::create(rewriter, loc, address, indirectIndex, mask,
                             indirectDim, offsets);
  }

  return ReindexOp::create(rewriter, loc, address, indirectIndex, indirectDim,
                           offsets);
}

struct ComposeReindexOfReindexPattern : public OpRewritePattern<ReindexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReindexOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getAddress().getDefiningOp<ReindexOp>();
    if (!producer || op.getIndirectIndex() || op.getMask() ||
        producer.getIndirectIndex() || producer.getMask()) {
      return failure();
    }

    auto outerOffsets = op.getMixedOffsets();
    auto innerOffsets = producer.getMixedOffsets();
    if (outerOffsets.size() != innerOffsets.size()) {
      return failure();
    }

    auto mergedOffsets =
        composePairwiseAdd(innerOffsets, outerOffsets, op.getLoc(), rewriter);

    auto replacement = ReindexOp::create(rewriter, op.getLoc(),
                                         producer.getAddress(), mergedOffsets);
    rewriter.replaceOp(op, replacement.getResult());
    return success();
  }
};

struct ComposeAdvanceOfAdvancePattern : public OpRewritePattern<AdvanceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AdvanceOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getAddress().getDefiningOp<AdvanceOp>();
    if (!producer) {
      return failure();
    }

    auto outerDeltas = op.getMixedDeltas();
    auto innerDeltas = producer.getMixedDeltas();
    if (outerDeltas.size() != innerDeltas.size()) {
      return failure();
    }

    auto mergedDeltas =
        composePairwiseAdd(innerDeltas, outerDeltas, op.getLoc(), rewriter);

    auto replacement = AdvanceOp::create(rewriter, op.getLoc(),
                                         producer.getAddress(), mergedDeltas);
    rewriter.replaceOp(op, replacement.getResult());
    return success();
  }
};

struct ComposeAdvanceOfReindexPattern : public OpRewritePattern<AdvanceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AdvanceOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getAddress().getDefiningOp<ReindexOp>();
    if (!producer) {
      return failure();
    }

    auto outerDeltas = op.getMixedDeltas();
    auto innerOffsets = producer.getMixedOffsets();
    if (outerDeltas.size() != innerOffsets.size()) {
      return failure();
    }

    auto mergedOffsets =
        composePairwiseAdd(innerOffsets, outerDeltas, op.getLoc(), rewriter);
    auto replacement = createReindexWithSameIndirection(
        rewriter, op.getLoc(), producer.getAddress(),
        producer.getIndirectIndex(), producer.getMask(),
        producer.getIndirectDimAttr(), mergedOffsets);
    rewriter.replaceOp(op, replacement.getResult());
    return success();
  }
};

struct ComposeReindexOfAdvancePattern : public OpRewritePattern<ReindexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReindexOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getAddress().getDefiningOp<AdvanceOp>();
    if (!producer) {
      return failure();
    }

    auto outerOffsets = op.getMixedOffsets();
    auto innerDeltas = producer.getMixedDeltas();
    if (outerOffsets.size() != innerDeltas.size()) {
      return failure();
    }

    auto mergedOffsets =
        composePairwiseAdd(innerDeltas, outerOffsets, op.getLoc(), rewriter);
    auto replacement = createReindexWithSameIndirection(
        rewriter, op.getLoc(), producer.getAddress(), op.getIndirectIndex(),
        op.getMask(), op.getIndirectDimAttr(), mergedOffsets);
    rewriter.replaceOp(op, replacement.getResult());
    return success();
  }
};

struct HoistFromTTPtrThroughReindexPattern
    : public OpRewritePattern<FromTTPtrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FromTTPtrOp op,
                                PatternRewriter &rewriter) const override {
    auto reindex = op.getSource().getDefiningOp<ReindexOp>();
    if (!reindex) {
      return failure();
    }

    auto importedAddress =
        FromTTPtrOp::create(rewriter, op.getLoc(), reindex.getAddress())
            .getResult();
    auto replacement = createReindexWithSameIndirection(
        rewriter, op.getLoc(), importedAddress, reindex.getIndirectIndex(),
        reindex.getMask(), reindex.getIndirectDimAttr(),
        reindex.getMixedOffsets());
    rewriter.replaceOp(op, replacement.getResult());
    return success();
  }
};

struct HoistFromTTPtrThroughAdvancePattern
    : public OpRewritePattern<FromTTPtrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FromTTPtrOp op,
                                PatternRewriter &rewriter) const override {
    auto advance = op.getSource().getDefiningOp<AdvanceOp>();
    if (!advance) {
      return failure();
    }

    auto importedAddress =
        FromTTPtrOp::create(rewriter, op.getLoc(), advance.getAddress())
            .getResult();
    auto replacement = AdvanceOp::create(rewriter, op.getLoc(), importedAddress,
                                         advance.getMixedDeltas());
    rewriter.replaceOp(op, replacement.getResult());
    return success();
  }
};
} // namespace

void FromTTPtrOp::build(OpBuilder &b, OperationState &state, Value source) {
  auto importedInfo = getImportedAddressInfo(source.getType());
  assert(importedInfo && "tta.from_tt_ptr source must be tt.ptr-like");

  auto resultType = AddrType::get(importedInfo->elementType, importedInfo->rank,
                                  importedInfo->addressSpace);
  build(b, state, resultType, source);
}

LogicalResult FromTTPtrOp::verify() {
  auto importedInfo = getImportedAddressInfo(getSource().getType());
  if (!importedInfo) {
    return emitOpError("source must be !tt.ptr<...>, tensor<...x!tt.ptr<...>>, "
                       "or !tt.ptr<tensor<...>>");
  }

  if (isa<triton::PointerType>(importedInfo->elementType) ||
      isa<TensorType>(importedInfo->elementType)) {
    return emitOpError("source pointee element type must be scalar "
                       "(non-pointer, non-tensor)");
  }

  auto resultType = dyn_cast<AddrType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !tta.addr<elem, rank, space>");
  }
  if (resultType.getElementType() != importedInfo->elementType) {
    return emitOpError("result element type must match source pointee "
                       "element type");
  }
  if (resultType.getRank() != importedInfo->rank) {
    return emitOpError("result rank must match imported pointer rank");
  }
  if (resultType.getAddressSpace() != importedInfo->addressSpace) {
    return emitOpError("result address space must match source address "
                       "space");
  }

  return success();
}

void FromTTPtrOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<HoistFromTTPtrThroughReindexPattern,
              HoistFromTTPtrThroughAdvancePattern>(context);
}

LogicalResult MakeAddrOp::verify() {
  int64_t rank = getSizes().size();

  if (static_cast<int64_t>(getMixedStrides().size()) != rank) {
    return emitOpError("strides must match sizes rank");
  }
  if (static_cast<int64_t>(getMixedOffsets().size()) != rank) {
    return emitOpError("offsets must match sizes rank");
  }
  if (static_cast<int64_t>(getMixedShape().size()) != rank) {
    return emitOpError("shape must match sizes rank");
  }

  auto baseType = dyn_cast<triton::PointerType>(getBase().getType());
  if (!baseType) {
    return emitOpError("base must be !tt.ptr<...>");
  }

  auto resultType = dyn_cast<AddrType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be !tta.addr<...>");
  }

  if (resultType.getRank() != rank) {
    return emitOpError("result rank must match sizes rank");
  }
  if (resultType.getElementType() != baseType.getPointeeType()) {
    return emitOpError("result element type must match base pointee type");
  }
  if (resultType.getAddressSpace() != baseType.getAddressSpace()) {
    return emitOpError("result address space must match base address space");
  }

  if (!getOrder().empty() && static_cast<int64_t>(getOrder().size()) != rank) {
    return emitOpError(
        "order length must match sizes rank for block pointer result");
  }

  if (!getOrder().empty()) {
    SmallVector<int64_t> order;
    order.reserve(getOrder().size());
    for (int32_t value : getOrder()) {
      order.push_back(value);
    }

    if (!isPermutationVector(order)) {
      return emitOpError("order must be a permutation of [0, rank)");
    }
  }

  return success();
}

void MakeAddrOp::build(OpBuilder &b, OperationState &state, Value base,
                       ArrayRef<int64_t> sizes, ArrayRef<OpFoldResult> strides,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> shape, ArrayRef<int32_t> order) {
  SmallVector<int64_t> staticStrides, staticOffsets, staticShape;
  SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicShape;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);

  auto basePtr = cast<triton::PointerType>(base.getType());
  Type resultType = AddrType::get(basePtr.getPointeeType(), sizes.size(),
                                  basePtr.getAddressSpace());

  build(b, state, resultType, base, sizes, dynamicStrides, dynamicOffsets,
        dynamicShape, b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticShape), order);
}

void ReindexOp::build(OpBuilder &b, OperationState &state, Value address,
                      ArrayRef<OpFoldResult> offsets) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(b, state, address.getType(), address, Value(), IntegerAttr(),
        dynamicOffsets, b.getDenseI64ArrayAttr(staticOffsets), Value());
}

void ReindexOp::build(OpBuilder &b, OperationState &state, Value address,
                      Value indirectIndex, int indirectDim,
                      ArrayRef<OpFoldResult> offsets) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(b, state, address.getType(), address, indirectIndex,
        b.getI32IntegerAttr(indirectDim), dynamicOffsets,
        b.getDenseI64ArrayAttr(staticOffsets), Value());
}

void ReindexOp::build(OpBuilder &b, OperationState &state, Value address,
                      Value indirectIndex, Value mask, int indirectDim,
                      ArrayRef<OpFoldResult> offsets) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(b, state, address.getType(), address, indirectIndex,
        b.getI32IntegerAttr(indirectDim), dynamicOffsets,
        b.getDenseI64ArrayAttr(staticOffsets), mask);
}

LogicalResult ReindexOp::verify() {
  auto rank = getAddressRank(getAddress().getType());
  if (!rank.has_value()) {
    return emitOpError("address must be !tta.addr<...>");
  }

  if (static_cast<int64_t>(getMixedOffsets().size()) != *rank) {
    return emitOpError("offsets must match address rank");
  }

  if (getResult().getType() != getAddress().getType()) {
    return emitOpError("result type must match address type");
  }

  auto indirectDimAttr = getIndirectDimAttr();
  if (getIndirectIndex()) {
    auto indexType = dyn_cast<RankedTensorType>(getIndirectIndex().getType());
    if (!indexType || indexType.getRank() != 1 ||
        !indexType.getElementType().isIntOrIndex()) {
      return emitOpError("indirect_index must be a 1D tensor of int or index "
                         "type");
    }

    if (!indirectDimAttr) {
      return emitOpError("indirect_dim is required when indirect_index is "
                         "present");
    }

    int64_t dim = indirectDimAttr.getInt();
    if (dim < 0 || dim >= *rank) {
      return emitOpError("indirect_dim is out of bounds");
    }

    if (getMask()) {
      auto maskType = dyn_cast<RankedTensorType>(getMask().getType());
      if (!maskType || maskType.getRank() != 1 ||
          !maskType.getElementType().isInteger(1)) {
        return emitOpError("mask must be a 1D tensor of i1");
      }
      if (maskType.getShape()[0] != indexType.getShape()[0]) {
        return emitOpError("mask size must match indirect_index size");
      }
    }
  } else {
    if (indirectDimAttr) {
      return emitOpError("indirect_dim requires indirect_index");
    }
    if (getMask()) {
      return emitOpError("mask requires indirect_index");
    }
  }

  return success();
}

OpFoldResult ReindexOp::fold(FoldAdaptor adaptor) {
  if (!getIndirectIndex() && !getMask() && areAllZero(getMixedOffsets())) {
    return getAddress();
  }
  return {};
}

void ReindexOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ComposeReindexOfReindexPattern, ComposeReindexOfAdvancePattern>(
      context);
}

void AdvanceOp::build(OpBuilder &b, OperationState &state, Value address,
                      ArrayRef<OpFoldResult> deltas) {
  SmallVector<int64_t> staticDeltas;
  SmallVector<Value> dynamicDeltas;
  dispatchIndexOpFoldResults(deltas, dynamicDeltas, staticDeltas);

  build(b, state, address.getType(), address, dynamicDeltas,
        b.getDenseI64ArrayAttr(staticDeltas));
}

LogicalResult AdvanceOp::verify() {
  auto rank = getAddressRank(getAddress().getType());
  if (!rank.has_value()) {
    return emitOpError("address must be !tta.addr<...>");
  }

  if (static_cast<int64_t>(getMixedDeltas().size()) != *rank) {
    return emitOpError("deltas must match address rank");
  }

  if (getResult().getType() != getAddress().getType()) {
    return emitOpError("result type must match address type");
  }

  return success();
}

OpFoldResult AdvanceOp::fold(FoldAdaptor adaptor) {
  if (areAllZero(getMixedDeltas())) {
    return getAddress();
  }
  return {};
}

void AdvanceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ComposeAdvanceOfAdvancePattern, ComposeAdvanceOfReindexPattern>(
      context);
}

void AtomicOp::build(OpBuilder &b, OperationState &state, StringRef kind,
                     Value ptr, Value offset, Value value) {
  build(b, state, value.getType(), b.getStringAttr(kind), ptr, offset, value,
        Value());
}

void AtomicOp::build(OpBuilder &b, OperationState &state, StringRef kind,
                     Value ptr, Value offset, Value value, Value mask) {
  build(b, state, value.getType(), b.getStringAttr(kind), ptr, offset, value,
        mask);
}

LogicalResult AtomicOp::verify() {
  if (!isValidAtomicKind(getKindAttr().getValue())) {
    return emitOpError("unsupported atomic kind: ") << getKindAttr().getValue();
  }

  Type valueType = getValue().getType();
  Type offsetType = getOffset().getType();

  auto valueTensorType = dyn_cast<RankedTensorType>(valueType);
  auto offsetTensorType = dyn_cast<RankedTensorType>(offsetType);
  if (static_cast<bool>(valueTensorType) !=
      static_cast<bool>(offsetTensorType)) {
    return emitOpError(
        "offset and value must both be scalars or both be tensors");
  }
  if (valueTensorType && offsetTensorType &&
      !llvm::equal(valueTensorType.getShape(), offsetTensorType.getShape())) {
    return emitOpError("offset and value tensor shapes must match");
  }

  auto ptrElementType = getAtomicElementType(getPtr().getType());
  if (!ptrElementType) {
    return emitOpError("ptr must be !tta.addr<...>");
  }

  Type valueElemType = mlir::getElementTypeOrSelf(valueType);
  if (*ptrElementType != valueElemType) {
    return emitOpError("value element type must match address element type");
  }

  StringRef kind = getKindAttr().getValue();
  if (kind == "fadd" && !isa<FloatType>(valueElemType)) {
    return emitOpError("fadd requires floating-point value type");
  }
  if ((kind == "and" || kind == "or" || kind == "xor") &&
      !valueElemType.isIntOrIndex()) {
    return emitOpError(kind) << " requires integer value type";
  }
  if ((kind == "add" || kind == "max" || kind == "min") &&
      !valueElemType.isIntOrFloat()) {
    return emitOpError(kind)
           << " requires integer or floating-point value type";
  }

  return success();
}

void AtomicCASOp::build(OpBuilder &b, OperationState &state, Value ptr,
                        Value offset, Value compare, Value value) {
  build(b, state, value.getType(), ptr, offset, compare, value);
}

LogicalResult AtomicCASOp::verify() {
  Type valueType = getValue().getType();
  Type compareType = getCompare().getType();
  Type offsetType = getOffset().getType();

  if (valueType != compareType) {
    return emitOpError("compare and value types must match");
  }

  auto valueTensorType = dyn_cast<RankedTensorType>(valueType);
  auto offsetTensorType = dyn_cast<RankedTensorType>(offsetType);
  if (static_cast<bool>(valueTensorType) !=
      static_cast<bool>(offsetTensorType)) {
    return emitOpError(
        "offset and value must both be scalars or both be tensors");
  }
  if (valueTensorType && offsetTensorType &&
      !llvm::equal(valueTensorType.getShape(), offsetTensorType.getShape())) {
    return emitOpError("offset and value tensor shapes must match");
  }

  auto ptrElementType = getAtomicElementType(getPtr().getType());
  if (!ptrElementType) {
    return emitOpError("ptr must be !tta.addr<...>");
  }

  Type valueElemType = mlir::getElementTypeOrSelf(valueType);
  if (*ptrElementType != valueElemType) {
    return emitOpError("value element type must match address element type");
  }

  if (!valueElemType.isIntOrFloat()) {
    return emitOpError(
        "atomic_cas requires integer or floating-point value type");
  }

  return success();
}

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   ArrayRef<OpFoldResult> maskDims, Value other) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  dispatchIndexOpFoldResults(maskDims, dynamicDims, staticDims);

  auto addrType = cast<AddrType>(ptr.getType());
  SmallVector<int64_t> shape(addrType.getRank(), ShapedType::kDynamic);
  Type resultType = RankedTensorType::get(shape, addrType.getElementType());

  build(b, state, resultType, ptr, dynamicDims,
        b.getDenseI64ArrayAttr(staticDims), other);
}

LogicalResult LoadOp::verify() {
  auto loadedInfo = getLoadedAddressInfo(getPtr().getType());
  if (!loadedInfo) {
    return emitOpError("ptr must be !tta.addr<...>");
  }

  int64_t maskRank = getMixedMaskDims().size();
  if (maskRank != 0 && maskRank != loadedInfo->rank) {
    return emitOpError("mask_dims must be empty or match pointer rank");
  }

  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be a ranked tensor");
  }
  if (loadedInfo->shape.size() != static_cast<size_t>(loadedInfo->rank)) {
    return emitOpError("internal error: loaded shape rank mismatch");
  }

  if (llvm::all_of(loadedInfo->shape,
                   [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
    if (resultType.getRank() != loadedInfo->rank) {
      return emitOpError("result rank must match address rank");
    }
  } else if (!llvm::equal(resultType.getShape(), loadedInfo->shape)) {
    return emitOpError("result shape must match loaded pointer shape");
  }
  if (resultType.getElementType() != loadedInfo->elementType) {
    return emitOpError("result element type must match address element type");
  }

  if (Value other = getOther()) {
    if (other.getType() != loadedInfo->elementType) {
      return emitOpError("other type must match address element type");
    }
  }

  return success();
}

void StoreOp::build(OpBuilder &b, OperationState &state, Value ptr, Value value,
                    ArrayRef<OpFoldResult> dims) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);

  build(b, state, ptr, value, dynamicDims, b.getDenseI64ArrayAttr(staticDims));
}

LogicalResult StoreOp::verify() {
  auto loadedInfo = getLoadedAddressInfo(getPtr().getType());
  if (!loadedInfo) {
    return emitOpError("ptr must be !tta.addr<...>");
  }

  int64_t maskRank = getMixedMaskDims().size();
  if (maskRank != 0 && maskRank != loadedInfo->rank) {
    return emitOpError("mask_dims must be empty or match pointer rank");
  }

  auto valueType = dyn_cast<RankedTensorType>(getValue().getType());
  if (!valueType) {
    return emitOpError("value must be a ranked tensor");
  }
  if (llvm::all_of(loadedInfo->shape,
                   [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
    if (valueType.getRank() != loadedInfo->rank) {
      return emitOpError("value rank must match address rank");
    }
  } else if (!llvm::equal(valueType.getShape(), loadedInfo->shape)) {
    return emitOpError("value shape must match stored pointer shape");
  }
  if (valueType.getElementType() != loadedInfo->elementType) {
    return emitOpError("value element type must match address element type");
  }

  return success();
}

} // namespace tta
} // namespace mlir
