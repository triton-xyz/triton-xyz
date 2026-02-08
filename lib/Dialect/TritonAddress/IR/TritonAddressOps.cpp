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
struct LoadedAddressInfo {
  int64_t rank;
  Type elementType;
  SmallVector<int64_t> shape;
};

std::optional<int64_t> getAddressRank(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (isa<triton::PointerType>(tensorType.getElementType())) {
      return tensorType.getRank();
    }
    return std::nullopt;
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      return tensorType.getRank();
    }
    return 1;
  }

  return std::nullopt;
}

std::optional<LoadedAddressInfo> getLoadedAddressInfo(Type type) {
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementPtrType =
        dyn_cast<triton::PointerType>(ptrTensorType.getElementType());
    if (!elementPtrType) {
      return std::nullopt;
    }

    LoadedAddressInfo info{ptrTensorType.getRank(),
                           elementPtrType.getPointeeType(),
                           SmallVector<int64_t>(ptrTensorType.getShape())};
    return info;
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (!pointeeTensorType) {
      return std::nullopt;
    }

    LoadedAddressInfo info{pointeeTensorType.getRank(),
                           pointeeTensorType.getElementType(),
                           SmallVector<int64_t>(pointeeTensorType.getShape())};
    return info;
  }

  return std::nullopt;
}

bool isValidAtomicKind(StringRef kind) {
  static constexpr std::array<StringLiteral, 9> kKinds = {
      "add", "and", "or", "xor", "max", "min", "xchg", "cmpxchg", "fadd"};
  return llvm::is_contained(kKinds, kind);
}
} // namespace

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

  auto resultType = getResult().getType();
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(resultType)) {
    if (!isa<triton::PointerType>(ptrTensorType.getElementType())) {
      return emitOpError("tensor result must have pointer element type");
    }
    if (ptrTensorType.getRank() != rank) {
      return emitOpError("tensor result rank must match sizes rank");
    }
    if (!llvm::equal(ptrTensorType.getShape(), getSizes())) {
      return emitOpError("tensor result shape must match sizes");
    }
    if (!getOrder().empty()) {
      return emitOpError("order must be empty for tensor-of-ptr result");
    }
    return success();
  }

  auto ptrType = dyn_cast<triton::PointerType>(resultType);
  if (!ptrType) {
    return emitOpError("result must be tensor-of-ptr or ptr<tensor<...>>");
  }

  auto pointeeTensorType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
  if (!pointeeTensorType) {
    return emitOpError("pointer result must be ptr<tensor<...>>");
  }

  if (pointeeTensorType.getRank() != rank) {
    return emitOpError("pointer pointee rank must match sizes rank");
  }
  if (!llvm::equal(pointeeTensorType.getShape(), getSizes())) {
    return emitOpError("pointer pointee shape must match sizes");
  }

  if (static_cast<int64_t>(getOrder().size()) != rank) {
    return emitOpError(
        "order length must match sizes rank for block pointer result");
  }

  SmallVector<int64_t> order;
  order.reserve(getOrder().size());
  for (int32_t value : getOrder()) {
    order.push_back(value);
  }

  if (!isPermutationVector(order)) {
    return emitOpError("order must be a permutation of [0, rank)");
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

  Type resultType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();
  if (order.empty()) {
    resultType = RankedTensorType::get(sizes, basePtr);
  } else {
    resultType = triton::PointerType::get(
        RankedTensorType::get(sizes, elemType), basePtr.getAddressSpace());
  }

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
    return emitOpError("address must be pointer-like type");
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
    return emitOpError("address must be pointer-like type");
  }

  if (static_cast<int64_t>(getMixedDeltas().size()) != *rank) {
    return emitOpError("deltas must match address rank");
  }

  if (getResult().getType() != getAddress().getType()) {
    return emitOpError("result type must match address type");
  }

  return success();
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

  auto ptrType = cast<triton::PointerType>(getPtr().getType());
  Type valueElemType = mlir::getElementTypeOrSelf(valueType);
  if (ptrType.getPointeeType() != valueElemType) {
    return emitOpError("value element type must match pointer pointee type");
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

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   ArrayRef<OpFoldResult> maskDims, Value other) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  dispatchIndexOpFoldResults(maskDims, dynamicDims, staticDims);

  auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType());
  auto tensorPtrType = dyn_cast<triton::PointerType>(ptr.getType());

  Type resultType;
  if (ptrTensorType) {
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    resultType = RankedTensorType::get(ptrTensorType.getShape(), elemType);
  } else if (tensorPtrType) {
    auto tensorType = dyn_cast<ShapedType>(tensorPtrType.getPointeeType());
    assert(tensorType &&
           "tta.load requires ptr<tensor<...>> for non-tensor pointer input");
    resultType = RankedTensorType::get(tensorType.getShape(),
                                       tensorType.getElementType());
  }

  build(b, state, resultType, ptr, dynamicDims,
        b.getDenseI64ArrayAttr(staticDims), other);
}

LogicalResult LoadOp::verify() {
  auto loadedInfo = getLoadedAddressInfo(getPtr().getType());
  if (!loadedInfo) {
    return emitOpError(
        "ptr must be tensor<...x!tt.ptr<...>> or !tt.ptr<tensor<...>>");
  }

  int64_t maskRank = getMixedMaskDims().size();
  if (maskRank != 0 && maskRank != loadedInfo->rank) {
    return emitOpError("mask_dims must be empty or match pointer rank");
  }

  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType) {
    return emitOpError("result must be a ranked tensor");
  }
  if (!llvm::equal(resultType.getShape(), loadedInfo->shape)) {
    return emitOpError("result shape must match loaded pointer shape");
  }
  if (resultType.getElementType() != loadedInfo->elementType) {
    return emitOpError("result element type must match pointer pointee type");
  }

  if (Value other = getOther()) {
    if (other.getType() != loadedInfo->elementType) {
      return emitOpError("other type must match pointer pointee element type");
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
    return emitOpError(
        "ptr must be tensor<...x!tt.ptr<...>> or !tt.ptr<tensor<...>>");
  }

  int64_t maskRank = getMixedMaskDims().size();
  if (maskRank != 0 && maskRank != loadedInfo->rank) {
    return emitOpError("mask_dims must be empty or match pointer rank");
  }

  auto valueType = dyn_cast<RankedTensorType>(getValue().getType());
  if (!valueType) {
    return emitOpError("value must be a ranked tensor");
  }
  if (!llvm::equal(valueType.getShape(), loadedInfo->shape)) {
    return emitOpError("value shape must match stored pointer shape");
  }
  if (valueType.getElementType() != loadedInfo->elementType) {
    return emitOpError("value element type must match pointer pointee type");
  }

  return success();
}

} // namespace tta
} // namespace mlir
