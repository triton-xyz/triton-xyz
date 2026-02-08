#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
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

bool isValidAtomicKind(StringRef kind) {
  static constexpr std::array<StringLiteral, 9> kKinds = {
      "add", "and", "or", "xor", "max", "min", "xchg", "cmpxchg", "fadd"};
  return llvm::is_contained(kKinds, kind);
}
} // namespace

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

void StoreOp::build(OpBuilder &b, OperationState &state, Value ptr, Value value,
                    ArrayRef<OpFoldResult> dims) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);

  build(b, state, ptr, value, dynamicDims, b.getDenseI64ArrayAttr(staticDims));
}

} // namespace tta
} // namespace mlir
