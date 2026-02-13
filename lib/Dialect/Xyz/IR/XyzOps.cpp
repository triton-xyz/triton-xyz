#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/Xyz/IR/XyzDialect.h"

using namespace mlir;
using namespace mlir::xyz;

namespace {

static LogicalResult verifyIndicesLike(Operation *op, Value indices, Value mask,
                                       int64_t gatherDim, int64_t rank) {
  auto indicesTy = dyn_cast<RankedTensorType>(indices.getType());
  if (!indicesTy || indicesTy.getRank() != 1 ||
      !indicesTy.getElementType().isIntOrIndex()) {
    return op->emitOpError("indices must be a 1D tensor of int/index");
  }

  if (gatherDim < 0 || gatherDim >= rank) {
    return op->emitOpError("gather_dim is out of bounds");
  }

  if (!mask) {
    return success();
  }

  auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskTy || maskTy.getRank() != 1 ||
      !maskTy.getElementType().isInteger(1)) {
    return op->emitOpError("mask must be a 1D tensor of i1");
  }

  int64_t maskSize = maskTy.getShape()[0];
  int64_t indexSize = indicesTy.getShape()[0];
  if (maskSize != ShapedType::kDynamic && indexSize != ShapedType::kDynamic &&
      maskSize != indexSize) {
    return op->emitOpError("mask size must match indices size");
  }

  return success();
}

} // namespace

LogicalResult GatherOp::verify() {
  auto addrTy = dyn_cast<tta::AddrType>(getAddress().getType());
  if (!addrTy) {
    return emitOpError("address must be !tta.addr");
  }

  int64_t gatherDim = getGatherDim();
  if (failed(verifyIndicesLike(*this, getIndices(), getMask(), gatherDim,
                               addrTy.getRank()))) {
    return failure();
  }

  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultTy) {
    return emitOpError("result must be a ranked tensor");
  }

  if (resultTy.getRank() != addrTy.getRank()) {
    return emitOpError("result rank must match address rank");
  }

  if (resultTy.getElementType() != addrTy.getElementType()) {
    return emitOpError("result element type must match address element type");
  }

  auto indicesTy = cast<RankedTensorType>(getIndices().getType());
  int64_t idxSize = indicesTy.getShape()[0];
  int64_t dimSize = resultTy.getShape()[gatherDim];
  if (idxSize != ShapedType::kDynamic && dimSize != ShapedType::kDynamic &&
      idxSize != dimSize) {
    return emitOpError("result gather dimension size must match indices size");
  }

  if (Value other = getOther()) {
    if (getElementTypeOrSelf(other.getType()) != addrTy.getElementType()) {
      return emitOpError("other element type must match address element type");
    }
  }

  return success();
}

LogicalResult ScatterOp::verify() {
  auto addrTy = dyn_cast<tta::AddrType>(getAddress().getType());
  if (!addrTy) {
    return emitOpError("address must be !tta.addr");
  }

  int64_t gatherDim = getGatherDim();
  if (failed(verifyIndicesLike(*this, getIndices(), getMask(), gatherDim,
                               addrTy.getRank()))) {
    return failure();
  }

  auto valueTy = dyn_cast<RankedTensorType>(getValue().getType());
  if (!valueTy) {
    return emitOpError("value must be a ranked tensor");
  }

  if (valueTy.getRank() != addrTy.getRank()) {
    return emitOpError("value rank must match address rank");
  }

  if (valueTy.getElementType() != addrTy.getElementType()) {
    return emitOpError("value element type must match address element type");
  }

  auto indicesTy = cast<RankedTensorType>(getIndices().getType());
  int64_t idxSize = indicesTy.getShape()[0];
  int64_t dimSize = valueTy.getShape()[gatherDim];
  if (idxSize != ShapedType::kDynamic && dimSize != ShapedType::kDynamic &&
      idxSize != dimSize) {
    return emitOpError("value gather dimension size must match indices size");
  }

  return success();
}
