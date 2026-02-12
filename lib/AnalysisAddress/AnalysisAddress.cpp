#include "triton-shared/AnalysisAddress/AnalysisAddress.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace triton {
namespace address {

namespace {

static FailureOr<AnalyzedAddress>
toAnalyzedAddress(const ptrexpr::PtrState &state) {
  if (state.isEmpty() || !state.source || state.getRank() <= 0) {
    return failure();
  }

  if (!state.isStructured()) {
    return failure();
  }

  if (!isa<triton::PointerType>(state.source.getType())) {
    return failure();
  }

  AnalyzedAddress result;
  result.base = state.source;
  result.strides = SmallVector<OpFoldResult>(state.strides);
  result.offsets = SmallVector<OpFoldResult>(state.offsets);
  result.shape = SmallVector<OpFoldResult>(state.shape);
  result.order = SmallVector<int32_t>(state.order);

  result.sizes.reserve(state.sizes.size());
  for (OpFoldResult size : state.sizes) {
    auto sizeAttr = getIntAttr(size);
    if (!sizeAttr.has_value()) {
      return failure();
    }
    result.sizes.push_back(sizeAttr.value());
  }

  return result;
}

} // namespace

FailureOr<AnalyzedAddress> AnalysisAddress::analyze(Value ptrLike, Location loc,
                                                    OpBuilder &builder) {
  if (auto makeAddr = ptrLike.getDefiningOp<tta::MakeAddrOp>()) {
    AnalyzedAddress result;
    result.base = makeAddr.getBase();
    result.sizes = llvm::to_vector(makeAddr.getSizes());
    result.strides = makeAddr.getMixedStrides();
    result.offsets = makeAddr.getMixedOffsets();
    result.shape = makeAddr.getMixedShape();
    result.order = llvm::to_vector(makeAddr.getOrder());
    return result;
  }

  if (auto advanceOp = ptrLike.getDefiningOp<triton::AdvanceOp>()) {
    auto maybeAddress = analyze(advanceOp.getPtr(), loc, builder);
    if (failed(maybeAddress)) {
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
        return failure();
      }
      Value cast = arith::IndexCastOp::create(builder, loc,
                                              builder.getIndexType(), delta)
                       .getResult();
      deltas.push_back(cast);
    }

    if (deltas.size() != maybeAddress->offsets.size() ||
        deltas.size() != maybeAddress->strides.size()) {
      return failure();
    }

    for (auto [index, delta] : llvm::enumerate(deltas)) {
      OpFoldResult scaledDelta =
          mulOFRs(delta, maybeAddress->strides[index], loc, builder);
      maybeAddress->offsets[index] =
          addOFRs(maybeAddress->offsets[index], scaledDelta, loc, builder);
    }

    return maybeAddress;
  }

  ptrexpr::PtrState state;
  if (auto makeTensorPtr = ptrLike.getDefiningOp<triton::MakeTensorPtrOp>()) {
    if (failed(ptrAnalysis.visitOperandMakeTensorPtr(makeTensorPtr, state, loc,
                                                     builder))) {
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
    return failure();
  }

  return toAnalyzedAddress(state);
}

FailureOr<Value> TTAEmitter::emitMakeAddr(const AnalyzedAddress &address,
                                          Location loc, OpBuilder &builder) {
  if (!isa<triton::PointerType>(address.base.getType())) {
    return failure();
  }

  auto makeAddr = tta::MakeAddrOp::create(
      builder, loc, address.base, address.sizes, address.strides,
      address.offsets, address.shape, address.order);
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
