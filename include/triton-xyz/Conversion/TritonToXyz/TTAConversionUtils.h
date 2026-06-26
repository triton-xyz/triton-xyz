#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton-xyz/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/Support/ErrorHandling.h"

#include <cassert>

namespace mlir::triton::tta_conversion {

inline bool isTensorOfTritonPointers(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

inline bool isScalarTritonPointer(Type type) {
  auto ptrType = dyn_cast<triton::PointerType>(type);
  return ptrType && !isa<RankedTensorType>(ptrType.getPointeeType());
}

inline bool isTritonPointerLikeForMemoryAccess(Type type) {
  return isTensorOfTritonPointers(type) || isScalarTritonPointer(type);
}

inline Type getPointerOffsetType(Type ptrLikeType, unsigned bitWidth) {
  MLIRContext *context = ptrLikeType.getContext();
  Type offsetElementType = IntegerType::get(context, bitWidth);

  if (auto tensorType = dyn_cast<RankedTensorType>(ptrLikeType)) {
    assert(isa<triton::PointerType>(tensorType.getElementType()) &&
           "expected a tensor of Triton pointers");
    return RankedTensorType::get(tensorType.getShape(), offsetElementType);
  }

  assert(isa<triton::PointerType>(ptrLikeType) &&
         "expected a Triton pointer-like type");
  return offsetElementType;
}

inline unsigned getIntegerLikeBitWidth(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    type = tensorType.getElementType();
  }

  if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }

  llvm_unreachable("expected integer or tensor-of-integer type");
}

inline Operation *getTTAAddressRootOp(Value address) {
  while (address) {
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
    return address.getDefiningOp();
  }
  return nullptr;
}

inline bool isTTAMakeAddrRootedChain(Value address) {
  return isa_and_nonnull<tta::MakeAddrOp>(getTTAAddressRootOp(address));
}

inline bool hasLoweredTTAAddressRoot(Value value) {
  if (!value) {
    return false;
  }

  return value.getDefiningOp<tta::MakeAddrOp>() ||
         value.getDefiningOp<tta::ReindexOp>() ||
         value.getDefiningOp<tta::IndirectReindexOp>() ||
         value.getDefiningOp<tta::AdvanceOp>();
}

} // namespace mlir::triton::tta_conversion
