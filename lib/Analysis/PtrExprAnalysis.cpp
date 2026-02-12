#include "triton-shared/Analysis/PtrExprAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstddef>

#define DEBUG_TYPE "triton-ptr-expr-analysis"

using namespace mlir;

namespace mlir {
namespace triton {
namespace ptrexpr {

int32_t PtrState::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         shape.size() == offsets.size());
  return offsets.size();
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

bool PtrState::hasModulo() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimHasModulo(i)) {
      return true;
    }
  }
  return false;
}

bool PtrState::dimHasModulo(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check modulo if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(shape[dim]);
  if (!intAttr.has_value()) {
    return true;
  }

  return intAttr.value() != 0;
}

bool isNotStructured(OpFoldResult offset) {
  auto value = dyn_cast<Value>(offset);
  return value && isa<ShapedType>(value.getType());
}

bool PtrState::dimIsStructured(uint32_t dim) const {
  assert(dim < getRank());

  return !isNotStructured(offsets[dim]);
}

int32_t PtrState::getNonStructuredDim() const {
  SmallVector<int32_t> dims;
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimIsStructured(i))
      continue;
    dims.emplace_back(i);
  }
  assert(dims.size() == 1 && "must have single non-continuous dimension");
  return dims.front();
}

bool PtrState::noStructuredDimExists() const {
  return getRank() > 0 && llvm::all_of(offsets, [](OpFoldResult offset) {
           return isNotStructured(offset);
         });
}

bool PtrState::isStructured() const {
  return llvm::all_of(
      offsets, [](OpFoldResult offset) { return !isNotStructured(offset); });
}

bool PtrState::isBlockPtr() const { return !order.empty(); }

bool isNotSingleDim(Value v) {
  auto shapedTy = dyn_cast<ShapedType>(v.getType());
  if (!shapedTy)
    return false;
  auto valShape = shapedTy.getShape();

  // Make sure there are more than 1 dimensions with size > 1.
  return llvm::find_singleton<int64_t>(
             valShape,
             [](int64_t size, bool) {
               return size > 1 ? (int64_t *)size : nullptr;
             },
             false) == nullptr;
}

LogicalResult PtrState::rebuildAsUnsupportedOp(Value operand) {
  if (isNotSingleDim(operand))
    return failure();

  if (!isEmpty())
    return failure();

  // Scalar has been take care early.
  // Assume here must be shape type.
  auto opType = cast<ShapedType>(operand.getType());
  // Skip support for pointer types which could be source of PtrState.
  // This check avoids creating a PtrState with non-structured source.
  if (isa<triton::PointerType>(opType.getElementType()))
    return failure();

  auto opShape = opType.getShape();

  // Setup state for unsupported operation.
  auto indexTy = IndexType::get(operand.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));
  for (auto size : opShape) {
    if (size == 1) {
      offsets.push_back(index0);
      strides.push_back(index0);
    } else {
      offsets.push_back(operand);
      strides.push_back(index1);
    }
    sizes.push_back(IntegerAttr::get(indexTy, APInt(64, size)));
    shape.push_back(index0);
  }
  return success();
}

LogicalResult PtrState::rebuildAsGatherScatter(Value op, int nonContinuousDim) {
  if (isNotSingleDim(op))
    return failure();
  if (nonContinuousDim >= getRank())
    return failure();

  // Scalar has been take care early.
  // Assume here must be shape type.
  auto opShape = cast<ShapedType>(op.getType()).getShape();
  // Make sure the op only contribute to nonContinuousDim by check
  // nonContinuousDim is the dimension with size > 1.
  if (opShape[nonContinuousDim] <= 1)
    return failure();

  // Setup state for nonContinuousDim.
  auto indexTy = IndexType::get(op.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));

  offsets[nonContinuousDim] = op;
  strides[nonContinuousDim] = index1;
  shape[nonContinuousDim] = index0;
  return success();
}

LogicalResult PtrState::addState(const PtrState &lhsState,
                                 const PtrState &rhsState,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support adding two pointer states that both "
        "have base pointers"));
    return failure();
  }

  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        arith::AddIOp::create(builder, loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) { // both lhs and rhs are scalars
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  if (!lhsState.isStructured() && !rhsState.isStructured()) {
    if (lhsState.getNonStructuredDim() != rhsState.getNonStructuredDim()) {
      LLVM_DEBUG(op->emitRemark(
          "PtrAnalysis: do not support adding two pointer states "
          "that have different non-continuous dimension"));
      return failure();
    }
  }

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    if (lhsState.dimIsStructured(i) && rhsState.dimIsStructured(i)) {
      auto newOffset =
          addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, builder);
      offsets.push_back(newOffset);
      auto newStride =
          addOFRs(lhsState.strides[i], rhsState.strides[i], loc, builder);
      strides.push_back(newStride);
    } else {
      if (isAnalysisingUnstructured) {
        assert(!lhsState.hasModulo() && !rhsState.hasModulo() &&
               "should not have dimension with modulo when analysing "
               "unstructured");
        if (hasConstZero(lhsState.strides[i]) &&
            hasConstZero(lhsState.offsets[i])) {
          // If lhs is not for dim i, we can just use rhs's stride and offset.
          offsets.push_back(rhsState.offsets[i]);
          strides.push_back(rhsState.strides[i]);
        } else if (hasConstZero(rhsState.strides[i]) &&
                   hasConstZero(rhsState.offsets[i])) {
          // If rhs is not for dim i, we can just use lhs's stride and offset.
          offsets.push_back(lhsState.offsets[i]);
          strides.push_back(lhsState.strides[i]);
        } else {
          OpFoldResult lhsOffset = lhsState.offsets[i];
          OpFoldResult rhsOffset = rhsState.offsets[i];
          OpFoldResult lhsStride = lhsState.strides[i];
          OpFoldResult rhsStride = rhsState.strides[i];
          // If stride is 0 which will happen after
          // visitOperandExpandDims/visitOperandSplat, we set the stride to 1 to
          // mul it with offset.
          if (hasConstZero(lhsStride)) {
            assert(lhsState.dimIsStructured(i) &&
                   !rhsState.dimIsStructured(i) &&
                   "If lhs stride is zero, it must be structured and rhs "
                   "stride is unstructured");
            lhsStride = builder.getIndexAttr(1);
          }
          if (hasConstZero(rhsStride)) {
            assert(rhsState.dimIsStructured(i) &&
                   !lhsState.dimIsStructured(i) &&
                   "If rhs stride is zero, it must be structured and lhs "
                   "stride is unstructured");
            rhsStride = builder.getIndexAttr(1);
          }

          // If both offset and stride not equal, we merge 2 PtrStates by change
          // offset * stride into (offset * stride) * 1 where new offset is
          // offset * stride and new stride is set to 1.
          // Then we'll have strides equal as 1, and merge them as PtrState with
          // same strides.
          if (lhsOffset != rhsOffset && lhsStride != rhsStride) {
            // Expand offset since unstructured offset has tensor type.
            OpFoldResult stride =
                expandOFRIndex(lhsStride, lhsOffset, loc, builder);
            // new offset = offset * stride
            lhsOffset = mulOFRs(lhsOffset, stride, loc, builder);
            // Expand offset since unstructured offset has tensor type.
            stride = expandOFRIndex(rhsStride, rhsOffset, loc, builder);
            // new offset = offset * stride
            rhsOffset = mulOFRs(rhsOffset, stride, loc, builder);
            // Set both strides to 1.
            lhsStride = builder.getIndexAttr(1);
            rhsStride = builder.getIndexAttr(1);
          }

          if (lhsStride == rhsStride) {
            // For case like lhs_offset * stride + rhs_offset * stride, it is
            // same as (lhs_offset + rhs_offset) * stride. We can just add the
            // offsets and reuse the stride like this:
            //   offsets[i] = lhsOffset + rhsOffset
            //   strides[i] = lhsStride
            // Expand structured offset since unstructured offset has tensor
            // type.
            if (!lhsState.dimIsStructured(i)) {
              rhsOffset = expandOFRIndex(rhsOffset, lhsOffset, loc, builder);
            } else {
              lhsOffset = expandOFRIndex(lhsOffset, rhsOffset, loc, builder);
            }
            // Add offsets.
            offsets.push_back(addOFRs(lhsOffset, rhsOffset, loc, builder));
            // Reuse stride.
            strides.push_back(lhsStride);
          } else {
            // Assert that offsets are equal if strides are not equal.
            // This is because we are already forcing the strides to be
            // equal to 1 earlier for case both offsets and strides not equal.
            assert(lhsOffset == rhsOffset &&
                   "If strides are not equal, offsets must be equal");
            // For case like offset * lhs_stride + offset * rhs_stride, it is
            // same as offset * (lhs_stride + rhs_stride). We can just add the
            // strides and reuse the offset like this:
            //   offsets[i] = lhsOffset
            //   strides[i] = lhsStride + rhsStride

            // Reuse offsets.
            offsets.push_back(lhsOffset);
            // Add strides.
            strides.push_back(addOFRs(lhsStride, rhsStride, loc, builder));
          }
        }
      } else {
        // Set stride to 1 when not continuous.
        strides.push_back(builder.getIndexAttr(1));
        // New offset is offset * stride.
        auto newLhsOffset = lhsState.offsets[i];
        auto newRhsOffset = rhsState.offsets[i];
        // Just propagate the unstructured offset to the result to track the
        // unstructured dimension. The real address calculation will be done
        // later in the PtrExprAnalysis::visitOperandAddptr.
        auto newOffset =
            lhsState.dimIsStructured(i) ? newRhsOffset : newLhsOffset;
        offsets.push_back(newOffset);
      }
    }

    sizes.push_back(lhsState.sizes[i]);
  }

  // AddPtr where both lhs and rhs containing modulo operators not supported
  if (lhsState.hasModulo() && rhsState.hasModulo()) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: do not support adding two pointer states "
                       "that both have modulo"));
    return failure();
  }

  if (lhsState.hasModulo() || rhsState.hasModulo()) {
    // visitOperandSplat and visitOperandExpandDims should enforce below
    assert(lhsState.getRank() <= 2);
  }

  // dealing with modulo:
  // - If lhs has no modulo, skip
  // - If rhs has zero offset on dim i, we can just use lhs's modulo
  // - If i == 0 and rhs is the result of a splat, we will allow the add. This
  // is because the user may be trying to express adding a constant offset to
  // increment dim1, but pointer analysis cannot differentiate dim1 vs dim0 in
  // this case.
  // - Else, the analysis fails

  // An example for the 3rd condition above can look like:
  // %0 = tt.splat %scalar
  // %1 = tt.splat %ptr
  // %2 = tt.arange
  // %3 = arith.remsi %2, %size
  // %4 = tt.addptr %1, %3
  // %5 = tt.addptr %4, %0
  // %5 may also occur in a loop to increment %4 every iteration.

  // Note that this is not bullet-proof. E.g., broken IR can actually increment
  // dim0 while dim0 already has modulo, since Triton offsets are element-wise
  // and not in unit of lower dimensions. However, this is highly unlikely but
  // the analysis will provide wrong result. Hence we provide a warning in this
  // case.
  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (rhs->hasModulo()) {
    std::swap(lhs, rhs);
  }

  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhs->getRank(); i++) {
    if (!lhs->dimIsStructured(i) || !rhs->dimIsStructured(i)) {
      // Shape is always 0 for non-structured dimension.
      shape.push_back(index0);
      continue;
    }

    if (!lhs->dimHasModulo(i)) {
      shape.push_back(lhs->shape[i]);
    } else if (hasConstZero(rhs->offsets[i])) {
      shape.push_back(lhs->shape[i]);
    } else if (i == 0 && lhs->getRank() == 2 && rhs->scalar) {
      shape.push_back(lhs->shape[1]);
      shape.push_back(lhs->shape[0]);
      LLVM_DEBUG(op->emitWarning(
          "PtrAnalysis: allowing adding pointer state with modulo in dim 0 to "
          "another pointer state with offset in dim 0.\nPlease verify the "
          "operand that contains a scalar is meant to increment pointers in "
          "dim1. If that is not the case it WILL LEAD TO WRONG COMPILATION "
          "RESULTS.\n\nTo avoid this warning, use expand_dims (instead of "
          "splat) to explicitly specify which dimension contains the scalar."));
      break;
    } else {
      LLVM_DEBUG(op->emitRemark(
          "PtrAnalysis: do not support adding to operand with modulo"));
      return failure();
    }
  }

  return success();
}

void PtrState::dump() const {
  llvm::dbgs() << "PtrState: ";
  if (source) {
    llvm::dbgs() << "source: " << source << "\n";
  }
  if (scalar) {
    llvm::dbgs() << "scalar: " << scalar << "\n";
  }

  llvm::dbgs() << "offsets:\n";
  llvm::interleave(offsets, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nstrides:\n";
  llvm::interleave(strides, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nsizes:\n";
  llvm::interleave(sizes, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nshape:\n";
  llvm::interleave(shape, llvm::dbgs(), "\n");
  llvm::dbgs() << "\norder:\n";
  llvm::interleave(order, llvm::dbgs(), "\n");
  if (isStructured()) {
    llvm::dbgs() << "structured\n";
  } else {
    for (int i = 0; i < getRank(); i++) {
      llvm::dbgs() << "dim " << i;
      if (dimIsStructured(i))
        llvm::dbgs() << " structured\n";
      else
        llvm::dbgs() << " not strucuted\n";
    }
  }

  llvm::dbgs() << "\n";
}

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  auto loc = op->getLoc();

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  if (lhsState.source && rhsState.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support multiplying base pointers"));
    return failure();
  }

  // currently do not support both tensors are effectively non-scalar
  if (!lhsState.scalar && !rhsState.scalar) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: only support multiplying pointer states when one of "
        "them represent a scalar"));
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  if (lhsState.scalar && rhsState.scalar) {
    scalar =
        arith::MulIOp::create(builder, loc, lhsState.scalar, rhsState.scalar);
  }

  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhs->sizes.size(); i++) {
    if (lhs->dimIsStructured(i)) {
      OpFoldResult newOffset =
          mulOFRs(lhs->offsets[i], rhs->scalar, loc, builder);
      offsets.push_back(newOffset);
      OpFoldResult newStride =
          mulOFRs(lhs->strides[i], rhs->scalar, loc, builder);
      strides.push_back(newStride);
      OpFoldResult newShape = mulOFRs(lhs->shape[i], rhs->scalar, loc, builder);
      shape.push_back(newShape);
    } else {
      assert(!lhs->dimHasModulo(i) &&
             "should not have non-structured dimension with modulo");
      if (isAnalysisingUnstructured) {
        assert(!lhs->hasModulo() &&
               "should not have non-structured dimension with modulo");
        // Keep offsets as is for unstructured dimension.
        // The address calculation will be done later in structured to
        // memref pass.
        offsets.push_back(lhs->offsets[i]);
        // Mul the scalar to stride.
        OpFoldResult newStride =
            mulOFRs(lhs->strides[i], rhs->scalar, loc, builder);
        strides.push_back(newStride);
      } else {
        // Just propagate the unstructured offset to the result to track the
        // unstructured dimension. The real address calculation will be done
        // later in the PtrExprAnalysis::visitOperandAddptr.
        OpFoldResult newOffset = lhs->offsets[i];
        offsets.push_back(newOffset);
        // Mul the scalar to stride.
        OpFoldResult newStride = lhs->strides[i];
        strides.push_back(newStride);
      }
      // Shape is always 0 for non-structured dimension.
      shape.push_back(index0);
    }
    sizes.push_back(lhs->sizes[i]);
  }

  if (rhs->hasModulo()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support multiplying pointer states that has "
        "modulos"));
    return failure();
  }

  return success();
}

LogicalResult PtrState::mergeUnstructuredState(const PtrState &other,
                                               Operation *op) {
  if (isStructured() || other.isStructured()) {
    LLVM_DEBUG(op->emitRemark("Expect merging pointer states both of which are "
                              "unstructured, but got structured state"));
    return failure();
  }
  if (other.getNonStructuredDim() != getNonStructuredDim()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support merging pointer states with "
        "different non-structured dimensions"));
    return failure();
  }
  if (getRank() != other.getRank()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support merging pointer states with "
        "different ranks"));
    return failure();
  }
  int gatherDim = other.getNonStructuredDim();

  // Merge gatherDim data from other.
  offsets[gatherDim] = other.offsets[gatherDim];
  strides[gatherDim] = other.strides[gatherDim];
  shape[gatherDim] = other.shape[gatherDim];

  return success();
}

LogicalResult PtrExprAnalysis::visitOperandAdd(arith::AddIOp addOp,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  // Checking for higher dimension is done in addState below
  if ((lhsState.getRank() == 1 && lhsState.hasModulo()) ||
      (rhsState.getRank() == 1 && rhsState.hasModulo())) {
    LLVM_DEBUG(addOp->emitRemark(
        "PtrAnalysis: do not support this pattern: a + arange(0, K) % M"));
    return failure();
  }

  // When one state hasModulo while other state is not structured.
  // Need to clear the modulo and use the operand as offset directly.
  if (!lhsState.isStructured() && rhsState.hasModulo()) {
    // TODO: support modulo in this case.
    if (!enableMakeGatherScatterTensorPtr ||
        rhsState
            .rebuildAsGatherScatter(addOp.getRhs(),
                                    lhsState.getNonStructuredDim())
            .failed())
      return failure();
  } else if (lhsState.hasModulo() && !rhsState.isStructured()) {
    if (!enableMakeGatherScatterTensorPtr ||
        lhsState
            .rebuildAsGatherScatter(addOp.getLhs(),
                                    rhsState.getNonStructuredDim())
            .failed())
      return failure();
  }
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.addState(lhsState, rhsState, isAnalysisingUnstructured, addOp,
                        builder);
}

LogicalResult PtrExprAnalysis::visitOperandMul(arith::MulIOp mulOp,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  // When one state hasModulo while other state is not structured.
  // Need to clear the modulo and use the operand as offset directly.
  if (!lhsState.isStructured() && rhsState.hasModulo()) {
    // TODO: support modulo in this case.
    if (!enableMakeGatherScatterTensorPtr ||
        rhsState
            .rebuildAsGatherScatter(mulOp.getRhs(),
                                    lhsState.getNonStructuredDim())
            .failed())
      return failure();
  } else if (lhsState.hasModulo() && !rhsState.isStructured()) {
    if (!enableMakeGatherScatterTensorPtr ||
        lhsState
            .rebuildAsGatherScatter(mulOp.getLhs(),
                                    rhsState.getNonStructuredDim())
            .failed())
      return failure();
  }

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.mulState(lhsState, rhsState, isAnalysisingUnstructured, mulOp,
                        builder);
}

LogicalResult PtrExprAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  assert(state.isEmpty());
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "PtrAnalysis: isAnalysisingUnstructured should only be true "
           "when enableMakeGatherScatterTensorPtr is true");
    // If we are analyzing unstructured state, just build state from current op.
    return state.rebuildAsUnsupportedOp(remOp.getResult());
  }

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    LLVM_DEBUG(remOp->emitRemark(
        "PtrAnalysis: only support cases when rhs of remainder "
        "contains scalar"));
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  // When lhs already not structured, just build state from current op.
  if (!state.isStructured()) {
    return state.rebuildAsGatherScatter(remOp.getResult(),
                                        state.getNonStructuredDim());
  }

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  if (state.hasModulo()) {
    LLVM_DEBUG(remOp->emitRemark(
        "PtrAnalysis: do not support multiple modulo within an expression"));
    // Multiple modulo ops on an expression is not supported.
    // But when the state has only one dimension, we can make it as
    // gather/scatter tensor ptr.
    if (state.getRank() == 1 && enableMakeGatherScatterTensorPtr)
      // Build the state from the current operation as an unstructured state,
      // but only when there is a single dimension involved.
      return state.rebuildAsGatherScatter(remOp.getResult(), 0);
    else
      return failure();
  }

  if (state.getRank() == 1) {
    // Apply the modulo before expanding shape, the common pattern is
    // offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    // a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
    // stride_ak)
    state.shape.back() = rhsState.scalar;
  } else if (state.getRank() == 2) {
    // torch inductor expands the tensor shape before applying the modulo.
    //
    // We only support either:
    // - (tl.arange(0, end)[:, None] % mod), or
    // - (tl.arange(0, end)[None, :] % mod)
    //
    // In both cases, we apply the modulo to the non-singleton dimension.
    auto shape = cast<TensorType>(remOp.getResult().getType()).getShape();
    if (shape[0] == 1) {
      state.shape[1] = rhsState.scalar;
    } else if (shape[1] == 1) {
      state.shape[0] = rhsState.scalar;
    } else {
      LLVM_DEBUG(remOp->emitRemark(
          "PtrAnalysis: taking modulo on a 2D tensor with no singleton "
          "dimension not supported"));
      return failure();
    }
  } else {
    LLVM_DEBUG(remOp->emitRemark("PtrAnalysis: unsupported modulo pattern"));
    return failure();
  }
  return success();
}

LogicalResult PtrExprAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                                 PtrState &state,
                                                 const Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());
  return visitOperand(extOp.getIn(), state, loc, builder);
}

LogicalResult
PtrExprAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                       PtrState &state, Location loc,
                                       OpBuilder &builder) {
  assert(state.isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  state.offsets.push_back(builder.getIndexAttr(start));
  state.sizes.push_back(builder.getIndexAttr(shape[0]));
  state.strides.push_back(builder.getIndexAttr(stride));
  state.shape.push_back(builder.getIndexAttr(0));
  return success();
}

LogicalResult
PtrExprAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                        PtrState &state, const Location loc,
                                        OpBuilder &builder) {
  assert(state.isEmpty());

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  state.offsets.insert(state.offsets.begin() + axis, builder.getIndexAttr(0));
  state.sizes.insert(state.sizes.begin() + axis, builder.getIndexAttr(1));
  state.strides.insert(state.strides.begin() + axis, builder.getIndexAttr(0));
  state.shape.insert(state.shape.begin() + axis, builder.getIndexAttr(0));

  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG(expandDimsOp->emitRemark(
        "PtrAnalysis: unsupported scenario where expand_dims result "
        "has modulo and rank > 2"));
    return failure();
  }

  return success();
}

LogicalResult
PtrExprAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    LLVM_DEBUG(broadcastOp->emitRemark(
        "PtrAnalysis: Unsupported broadcast source type"));
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  for (size_t i = 0; i < dstShape.size(); i++) {
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i]) {
      state.sizes[i] = builder.getIndexAttr(dstShape[i]);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

LogicalResult PtrExprAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                                 PtrState &state,
                                                 const Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
    for (auto s : dstShape) {
      state.offsets.push_back(builder.getIndexAttr(0));
      state.sizes.push_back(builder.getIndexAttr(s));
      state.strides.push_back(builder.getIndexAttr(0));
      state.shape.push_back(builder.getIndexAttr(0));
    }
  } else {
    LLVM_DEBUG(splatOp->emitRemark("PtrAnalysis: unsupported splat pattern"));
    return failure();
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension
  if (state.scalar)
    state.offsets[0] = state.scalar;

  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG(splatOp->emitRemark(
        "PtrAnalysis: unsupported scenario where splat result "
        "has modulo and rank > 2"));
    return failure();
  }

  return success();
}

LogicalResult PtrExprAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    return failure();
  } else if (!ptrState.source) {
    LLVM_DEBUG(llvm::dbgs()
               << "No src ptr state when processing " << addptrOp << "\n");
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  assert(ptrState.source && "ptr field should provide source / base pointer");

  assert(ptrState.getRank() == offsetState.getRank() &&
         "ptr and offset field should have the same rank");

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.addState(ptrState, offsetState, isAnalysisingUnstructured,
                        addptrOp, builder);
}

LogicalResult PtrExprAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                      PtrState &state,
                                                      const Location loc,
                                                      OpBuilder &builder) {
  assert(state.isEmpty());
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(builder, constAttr,
                                                builder.getIndexType(), loc);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    if (i == 0) {
      state.offsets.push_back(constOp.getResult());
    } else {
      state.offsets.push_back(builder.getIndexAttr(0));
    }

    state.sizes.push_back(builder.getIndexAttr(resultType.getShape()[i]));
    state.strides.push_back(builder.getIndexAttr(0));
    state.shape.push_back(builder.getIndexAttr(0));
  }

  return success();
}

LogicalResult
PtrExprAnalysis::visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();

  if (makeTPtrOp.getOrder().empty()) {
    LLVM_DEBUG(makeTPtrOp->emitRemark(
        "PtrAnalysis: expect tt.make_tensor_ptr to have order field set"));
    return failure();
  }

  auto resType = cast<triton::PointerType>(makeTPtrOp.getResult().getType());
  auto pointeeType = cast<ShapedType>(resType.getPointeeType());
  auto shape = pointeeType.getShape();

  for (int64_t i = 0; i < pointeeType.getRank(); i++) {
    state.sizes.push_back(builder.getIndexAttr(shape[i]));

    auto strideCst = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), makeTPtrOp.getStrides()[i]);
    state.strides.push_back(strideCst.getResult());

    auto offsetCst = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), makeTPtrOp.getOffsets()[i]);

    auto scaledOffset = arith::MulIOp::create(
        builder, loc, offsetCst.getResult(), strideCst.getResult());
    state.offsets.push_back(scaledOffset.getResult());

    auto shapeCst = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), makeTPtrOp.getShape()[i]);
    state.shape.push_back(shapeCst.getResult());
  }
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());
  assert(state.isBlockPtr() &&
         "tt.make_tensor_ptr pointer state should describe a block pointer");

  return success();
}

LogicalResult PtrExprAnalysis::visitOperandForOp(scf::ForOp forOp,
                                                 Value operand, PtrState &state,
                                                 const Location loc,
                                                 OpBuilder &builder) {

  if (!loopResultResolver) {
    return failure();
  }

  auto newState = loopResultResolver(forOp, operand);
  if (failed(newState)) {
    LLVM_DEBUG(forOp.emitWarning(
        "PtrExprAnalysis: failed to resolve PtrState returned by "
        "the loop."));
    return failure();
  }

  state = newState.value();
  return success();
}

LogicalResult PtrExprAnalysis::visitOperandIntToPtr(triton::IntToPtrOp op,
                                                    PtrState &state,
                                                    const Location loc,
                                                    OpBuilder &builder) {
  state.source = op.getResult();
  return success();
}

LogicalResult PtrExprAnalysis::visitOperandBitcast(triton::BitcastOp op,
                                                   PtrState &state,
                                                   const Location loc,
                                                   OpBuilder &builder) {
  auto resType = op.getResult().getType();
  if (isa<ShapedType>(resType)) {
    return visitOperand(op.getSrc(), state, loc, builder);
  }
  state.source = op.getResult();
  return success();
}

LogicalResult PtrExprAnalysis::visitOperand(Value operand, PtrState &state,
                                            const Location loc,
                                            OpBuilder &builder) {
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  // Not using knownPtrs when isAnalysisingUnstructured is true.
  // This is because we are analyzing unstructured state, the data in knownPtrs
  // is not valid for unstructured state.
  if (!isAnalysisingUnstructured &&
      knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = arith::IndexCastOp::create(builder, loc,
                                             builder.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return success();
  } else if (isa<IndexType>(operand.getType())) {
    state.scalar = operand;
    return success();
  }

  if (isa<triton::PointerType>(operand.getType())) {
    // A scalar pointer can either be produced by AddPtrOp or a block
    // argument
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                  builder);
      } else if (auto castOp = dyn_cast<triton::BitcastOp>(op)) {
        return visitOperandBitcast(castOp, state, loc, builder);
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        return visitOperandIntToPtr(intToPtrOp, state, loc, builder);
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        llvm_unreachable("Unexpected operand defining operation tts.make_tptr");
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        state.source = operand;
        return success();
      } else {
        LLVM_DEBUG(op->emitRemark(
            "Unexpected defining op for triton pointer operand"));
        return failure();
      }
    } else {
      state.source = operand;
      return success();
    }
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::ForOp>()) {
    return visitOperandForOp(op, operand, state, loc, builder);
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      return failure();
    }

    // This operand must be an iter-arg of an inner-loop in a multiple-level
    // nested loop, which means its PtrState must have already been populated
    // during rewriteForOp of the parent loop.
    state = knownPtrs[operand];
    return success();
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "PtrAnalysis: encountered addptr operand produced by an "
                  "unsupported operation: "
               << operand);

    if (!enableMakeGatherScatterTensorPtr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "PtrAnalysis: failed to rebuild as unsupported op\n");
      return failure();
    }
    return state.rebuildAsUnsupportedOp(operand);
  }
}

} // namespace ptrexpr
} // namespace triton
} // namespace mlir
