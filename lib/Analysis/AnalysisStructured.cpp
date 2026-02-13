#include "triton-shared/Analysis/AnalysisStructured.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstddef>
#include <optional>
#include <queue>
#include <string>

#define DEBUG_TYPE "triton-ptr-analysis"

using namespace mlir;

// Try to apply unstructured mask on the ptr.
static Value applyUnstructuredMask(Operation *op, Value ptr,
                                   triton::MaskState &mstate, Location loc,
                                   OpBuilder builder) {
  SmallVector<std::pair<unsigned, Value>> masks = mstate.getUnstructuredMasks();
  if (masks.empty()) {
    return ptr;
  }
  if (masks.size() > 1) {
    LLVM_DEBUG(op->emitRemark(
        "MaskAnalysis failed for more than one unstructured masks"));
    return nullptr;
  }

  auto [dim, unstructuredMask] = masks[0];
  if (auto gatherScatterPtr =
          ptr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
    if (dim != gatherScatterPtr.getGatherScatterDim()) {
      LLVM_DEBUG(op->emitRemark(
          "MaskAnalysis failed for unstructured mask dim not equal "
          "gather scatter dim"));
      return nullptr;
    }

    ptr = tts::MakeGatherScatterTensorPtrOp::create(
              builder, loc, gatherScatterPtr.getBase(),
              gatherScatterPtr.getGatherScatterOffset(), unstructuredMask,
              gatherScatterPtr.getGatherScatterDim(),
              gatherScatterPtr.getSizes(), gatherScatterPtr.getMixedStrides(),
              gatherScatterPtr.getMixedOffsets())
              .getResult();
  } else if (auto structuredPtr = ptr.getDefiningOp<tts::MakeTensorPtrOp>()) {
    auto ofrToI32Value = [&](OpFoldResult ofr) {
      Value v = dyn_cast<Value>(ofr);
      if (!v) {
        v = arith::ConstantOp::create(builder, loc,
                                      cast<TypedAttr>(cast<Attribute>(ofr)))
                .getResult();
      }
      if (isa<IndexType>(v.getType())) {
        v = arith::IndexCastOp::create(builder, loc, builder.getI32Type(), v)
                .getResult();
      } else if (v.getType().isInteger(64)) {
        v = arith::TruncIOp::create(builder, loc, builder.getI32Type(), v)
                .getResult();
      }

      return v;
    };
    OpFoldResult offsetFold = structuredPtr.getMixedOffsets()[dim];
    Value offset = ofrToI32Value(offsetFold);
    auto offsetRowType = RankedTensorType::get({structuredPtr.getSizes()[dim]},
                                               offset.getType());
    OpFoldResult strideFold = structuredPtr.getMixedStrides()[dim];
    Value stride = ofrToI32Value(strideFold);
    // Divide stride since offset of tts::MakeTensorPtrOp already include the
    // stride, but gatherScatterOffset of tts::MakeGatherScatterTensorPtrOp
    // should not include stride.
    offset = arith::DivUIOp::create(builder, loc, offset, stride).getResult();

    Value gatherScatterOffset =
        tensor::SplatOp::create(builder, loc, offsetRowType, offset)
            .getResult();
    Value range = triton::MakeRangeOp::create(builder, loc, offsetRowType, 0,
                                              structuredPtr.getSizes()[dim])
                      .getResult();
    gatherScatterOffset =
        arith::AddIOp::create(builder, loc, gatherScatterOffset, range)
            .getResult();
    ptr = tts::MakeGatherScatterTensorPtrOp::create(
              builder, loc, structuredPtr.getBase(), gatherScatterOffset,
              unstructuredMask, dim, structuredPtr.getSizes(),
              structuredPtr.getMixedStrides(), structuredPtr.getMixedOffsets())
              .getResult();
  } else {
    return nullptr;
  }
  // Clear the mask size for gather/scatter dim.
  mstate.dims[dim] = OpFoldResult(builder.getI32IntegerAttr(0));
  return ptr;
}

namespace mlir {

namespace tts {

PtrAnalysis::PtrAnalysis(bool enableMakeGatherScatterTensorPtr)
    : PtrExprAnalysis(enableMakeGatherScatterTensorPtr) {
  setLoopResultResolver(
      [this](scf::ForOp forOp,
             Value operand) -> FailureOr<mlir::triton::ptrexpr::PtrState> {
        auto it = llvm::find(forOp->getResults(), operand);
        auto index = std::distance(forOp->getResults().begin(), it);
        auto state = getLoopResultPtrState(forOp, index);
        if (failed(state)) {
          return failure();
        }
        return mlir::triton::ptrexpr::PtrState(*state);
      });
}

tts::MakeTensorPtrOp PtrState::createTTSMakeTensorPtrOp(OpBuilder &builder,
                                                        Location loc) {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(sizes[i]);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }

  auto op = mlir::tts::MakeTensorPtrOp::create(
      builder, loc, source, staticSizes, strides, offsets, shape, order);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

tts::MakeGatherScatterTensorPtrOp
PtrState::createTTSMakeGatherScatterTensorPtrOp(OpBuilder &builder,
                                                Location loc) {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(sizes[i]);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }

  int nonContinuousDim = getNonStructuredDim();

  Value nonContinuousOffset = cast<Value>(offsets[nonContinuousDim]);

  // Collapse nonContinuousOffset to 1D.
  auto offsetTy = cast<ShapedType>(nonContinuousOffset.getType());
  if (offsetTy.getRank() > 1) {
    SmallVector<ReassociationExprs, 4> reassociationMap(1);
    for (int i = 0; i < offsetTy.getRank(); ++i)
      reassociationMap[0].push_back(builder.getAffineDimExpr(i));

    int offsetSize = 1;
    for (int size : offsetTy.getShape())
      offsetSize *= size;

    auto collapseTy =
        RankedTensorType::get({offsetSize}, offsetTy.getElementType());
    nonContinuousOffset =
        tensor::CollapseShapeOp::create(builder, loc, collapseTy,
                                        nonContinuousOffset, reassociationMap)
            .getResult();
    offsets[nonContinuousDim] = nonContinuousOffset;
  }
  // Generate tts::make_gather_scatter_tensor_ptr.
  auto op = mlir::tts::MakeGatherScatterTensorPtrOp::create(
      builder, loc, source, nonContinuousOffset, nonContinuousDim, staticSizes,
      strides, offsets);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_gather_scatter_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandAdd(addOp, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandMul(mulOp, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandRem(remOp, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandExtSI(extOp, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandMakeRange(rangeOp, state, loc, builder);
}

LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandExpandDims(expandDimsOp, state, loc,
                                                 builder);
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandBroadcast(broadcastOp, state, loc,
                                                builder);
}

LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandSplat(splatOp, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandAddptr(addptrOp, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandConstSplat(op, state, loc, builder);
}

LogicalResult
PtrAnalysis::visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandMakeTensorPtr(makeTPtrOp, state, loc,
                                                    builder);
}

LogicalResult PtrAnalysis::visitOperandForOp(scf::ForOp forOp, Value operand,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandForOp(forOp, operand, state, loc,
                                            builder);
}

LogicalResult PtrAnalysis::visitOperandIntToPtr(triton::IntToPtrOp op,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandIntToPtr(op, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandBitcast(triton::BitcastOp op,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  return PtrExprAnalysis::visitOperandBitcast(op, state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {
  return PtrExprAnalysis::visitOperand(operand, state, loc, builder);
}

LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  PtrExprAnalysis::knownPtrs[op.getResult()] = state;

  if (isa<RankedTensorType>(op.getPtr().getType())) {
    if (state.isStructured()) {
      auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else if (enableMakeGatherScatterTensorPtr) {
      PtrState unstructuredState;
      // Switch to unstructured state analysis to create offsets and strides
      // for the non-structured dimension.
      // NOTE: this is the only place where we switch to unstructured state
      // analysis.
      isAnalysisingUnstructured = true;
      // Visit the operand again to calculate the offsets and strides for the
      // unstructured state.
      LogicalResult result =
          visitOperandAddptr(op, unstructuredState, op.getLoc(), builder);
      // Switch back to structured state analysis.
      isAnalysisingUnstructured = false;
      if (result.failed()) {
        LLVM_DEBUG(op->emitRemark(
            "PtrAnalysis: Failed to analyze ptr of tt.addptr for "
            "unstructured state"));
        return failure();
      }
      if (state.mergeUnstructuredState(unstructuredState, op).failed()) {
        LLVM_DEBUG(op->emitRemark(
            "PtrAnalysis: Failed to merge unstructured state for tt.addptr"));
        return failure();
      }
      auto maketptrOp =
          state.createTTSMakeGatherScatterTensorPtrOp(builder, op.getLoc());
      // Update knownPtrs to merged state.
      PtrExprAnalysis::knownPtrs[op.getResult()] = state;
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else {
      return failure();
    }
  } else {
    // record the ptr as we have visited and built up the state for this scalar
    // pointer, which may be used by rewriteForOp later.
    ptrMap.map(op.getResult(), op.getResult());
  }
  return success();
}

LogicalResult PtrAnalysis::rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandMakeTensorPtr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
  PtrExprAnalysis::knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), maketptrOp.getResult());
  return success();
}

LogicalResult PtrAnalysis::rewriteAdvanceOp(triton::AdvanceOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperand(op->getOperand(0), state, loc, builder).failed()) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: Failed to analyze ptr of tt.advance"));
    return failure();
  }
  assert(state.isBlockPtr() &&
         "tt.advance pointer state should describe a block pointer");

  auto incrementOffsets = op.getOffsets();

  SmallVector<OpFoldResult> newOffsets;
  for (auto [increment, offset, stride] :
       llvm::zip(incrementOffsets, state.offsets, state.strides)) {
    Value offsetValue;
    if (auto offsetIntAttr = getIntAttr(offset)) {
      auto constOp = arith::ConstantOp::create(
          builder, loc, builder.getIndexAttr(offsetIntAttr.value()));
      offsetValue = constOp.getResult();
    } else {
      offsetValue = cast<Value>(offset);
    }
    auto castOp = arith::IndexCastOp::create(builder, loc,
                                             builder.getIndexType(), increment);
    auto mulOp = arith::MulIOp::create(builder, loc, castOp.getResult(),
                                       cast<Value>(stride));
    auto addOp =
        arith::AddIOp::create(builder, loc, mulOp.getResult(), offsetValue);
    newOffsets.push_back(addOp.getResult());
  }

  state.offsets = SmallVector<OpFoldResult>(newOffsets);

  auto newOp = state.createTTSMakeTensorPtrOp(builder, loc);
  PtrExprAnalysis::knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), newOp.getResult());
  return success();
}

static bool isPointerType(Type t) {
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensor.getElementType());
  }
  return isa<triton::PointerType>(t);
}

FailureOr<PtrState> PtrAnalysis::getLoopInitArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto ptr = forOp.getInitArgs()[index];

  // If the pointer into the scf.for was defined by tts.get_structured_state,
  // we can get the pointer state from the original pointer (the op's input):
  //
  // %ptr, %offset_1, %offset_2,..., %stride_1, %stride_2,... =
  // tts.get_structured_state %original
  // scf.for ... (%ptr) {...}
  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (PtrExprAnalysis::knownPtrs.count(originalPtr)) {
      return PtrState(PtrExprAnalysis::knownPtrs[originalPtr]);
    }
  }

  // For nested loops scenarios, a pointer in init-args can be returned from
  // another loop of the same level:
  // e.g.:
  // clang-format off
  //  %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //    %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
  //    }
  //    %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
  //      %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      ...
  //    }
  //    ...
  //  }
  // clang-format on
  // Notice %arg8 = %23 comes from the return value of the first loop.
  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  // If the pointer isn't defined by tts.get_structured_state nor another loop,
  // it means the current pointer is an iterarg of the outer loop.
  // In such cases, the outer loops would have already set up the PtrState for
  // us already.
  //
  // scf.for iterargs(%ptr = %init_arg) {
  //    scf.for iterargs(%ptr1 = %ptr) {  <--- we're dealing with `%ptr1` here.
  //          ...
  //    }
  // }
  if (PtrExprAnalysis::knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return PtrState(PtrExprAnalysis::knownPtrs[ptr]);
  }

  return failure();
}

PtrState PtrAnalysis::reconcileLoopPtrState(
    scf::ForOp forOp, size_t iterArgIndex, const PtrState &state,
    llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = iterArgIndex + 1;
  if (newState.getRank() == 0) {
    assert(newState.scalar);
    // for scalar pointers, the scalar contains the offset and is the only
    // relevant newState that could be updated by the loop.
    newState.scalar = getReplacementVal(forOp, cnt);
  } else {
    for (auto &offset : newState.offsets) {
      offset = getReplacementVal(forOp, cnt++);
    }

    for (auto &stride : newState.strides) {
      stride = getReplacementVal(forOp, cnt++);
    }
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getLoopIterArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    // Skip if the loop init arg is not structured.
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op.getRegionIterArg(index); });
}

FailureOr<PtrState> PtrAnalysis::getLoopResultPtrState(scf::ForOp forOp,
                                                       size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    // Skip if the loop init arg is not structured.
    return failure();
  }
  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op->getResult(index); });
}

LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {
  for (auto [i, arg] : llvm::enumerate(op.getRegionIterArgs())) {
    if (!maybeStructuredArgs.contains(arg)) {
      continue;
    }

    auto state = getLoopIterArgPtrState(op, i);
    if (failed(state)) {
      // Because the maybeStructuredArgs may contain values that are not
      // considered structured by PtrAnalysis, failing to retrieve the PtrState
      // should not fail the rewrite process.
      // We emit an error for diagnostics and debugging purposes.
      LLVM_DEBUG(op->emitWarning(
          "Rewrite for-op failed. Could not find PtrState for iter-arg index " +
          std::to_string(i)));
      continue;
    }
    // Skip when no structured dimension exists
    if (state->noStructuredDimExists())
      continue;

    // Save the current init arg's PtrState
    PtrExprAnalysis::knownPtrs[arg] = state.value();

    // For tensors of pointers, create a tts.make_tptr at the beginning of the
    // loop body that correspond to this region iter arg. In case it is used
    // by tt.load/tt.store in the loop body before pointer updates, this will
    // make sure rewriteLoadOp/rewriteStoreOp can use the analysis result.
    // E.g., given the following input (%tensor_of_ptr is a block arg):
    // scf.for (%tensor_of_ptr) {
    //   %data = tt.load %tensor_of_ptr
    //   // more operations to update %tensor_of_ptr
    // }
    // We may produce the following output:
    // scf.for (%base_ptr, %stride, %offset) {
    //   %tensor_of_ptr = tts.make_tptr(%base_ptr, %stride, %offset)
    //   %data = tts.load %tensor_of_ptr
    //   // more operations to update %offset
    // }
    // If %tensor_of_ptr is not used (i.e., %tensor_of_ptr is updated before
    // used in the original IR), it will simply be removed by
    // canonicalization.

    // For scalar pointers, there is no need to create a tts.addptr at the
    // beginning of the loop body. We don't lower tt.load and tt.store on
    // scalars in this pass; pointer arithmetics can also just use the
    // original pointer.
    // Note that there can be tensor of indices in iter-arg, so we only create
    // the make_tensor_ptr op when the arg is of pointer type.
    if (isPointerType(arg.getType())) {
      if (state->getRank() != 0) {
        OpBuilder builder(op.getRegion());
        auto maketptrOp = state->createTTSMakeTensorPtrOp(builder, op.getLoc());
        ptrMap.map(arg, maketptrOp.getResult());
      }
    }
  }

  // Recursively rewrite the inner ops
  if (rewriteOp(op).failed()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting for op"));
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::rewriteGetStructuredStateOp(tts::GetStructuredStateOp op) {
  auto tritonValue = op->getOperand(0);

  // If this triton value isn't known, it means PtrAnalysis has failed to
  // analyze this pointer. In such cases, simply remap all uses of the
  // structured value back to its original triton value.
  if (!PtrExprAnalysis::knownPtrs.contains(tritonValue)) {
    LLVM_DEBUG(op.emitRemark(
        "Rewrite GetStructuredStateOp failed. Could not find PtrState."));
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }

  tts::PtrState state = PtrExprAnalysis::knownPtrs[tritonValue];
  if (!state.isStructured()) {
    LLVM_DEBUG(op.emitRemark(
        "Rewrite GetStructuredStateOp failed. PtrState is not structured."));
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }
  Value remappedValue =
      ptrMap.contains(tritonValue) ? ptrMap.lookup(tritonValue) : tritonValue;

  SmallVector<Value> replacements{remappedValue};
  OpBuilder builder(op);

  if (state.getRank() == 0) {
    // For scalar pointers, the scalar contains the offset and is the only
    // relevant state that could be updated by the loop.
    if (state.scalar) {
      replacements.push_back(state.scalar);
    } else {
      // This operand is a pointer directly from the kernel arguments.
      // Use offset 0.
      assert(!tritonValue.getDefiningOp());
      replacements.push_back(arith::ConstantOp::create(
          builder, op.getLoc(), builder.getIndexAttr(0)));
    }
  } else {
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = arith::ConstantOp::create(
            builder, op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = arith::ConstantOp::create(
            builder, op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }
  }

  op->replaceAllUsesWith(replacements);
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op,
                                         bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "loadOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: scalar loadOp will not be rewritten"));
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);
  Value scalarOther;

  OpBuilder builder(op);
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      LLVM_DEBUG(op->emitRemark("MaskAnalysis failed"));
      return failure();
    }
    ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
    if (!ptr) {
      return failure();
    }
    dims = mstate.dims;
  }

  if (other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = utils::getScalarValue(other, loc, builder);
    if (!scalarOther) {
      LLVM_DEBUG(op->emitRemark("other value used in masked load produced by "
                                "unsupported instruction"));
      return failure();
    }
  }

  auto loadOp = tts::LoadOp::create(builder, loc, ptr, dims, scalarOther);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::load:\n";
    loadOp->dump();
  });

  op.replaceAllUsesWith(loadOp.getResult());
  op->erase();
  return success();
}

// Structured values from the TritonStructuredDialect have offsets and strides
// that might change in each loop iteration and hence will appear in an scf.for
// iter-args like so:
//
// %structured, %offsets, %strides  = tts.get_structured_state
// scf.for (%arg0 = %structured, %arg1 = %offsets, %arg2 = %strides) {
//   %a = %arg0 + 1
//   %b = %b + 2
//   scf.for (%arg1 = %b) {
//      ...
//   }
// }
//
// In `rewriteForOp`, we have to recognize such structured values in order to
// rewrite their PtrState accordingly. Previously, only values of Pointer-like
// type (e.g.: tensor<tt.ptr<>> or tt.ptr<tensor<>>), so detecting these values
// is as easy as checking the type.
//
// Now, tensor of indices could also appear in a loop's iter-arg. To reliably
// detect all such cases, we perform a BFS-like traversal of the IR where the
// sources are the results of `tts.get_structured_state`. All values that
// originate from the results of `tts.get_structured_state` are consider
// "maybeStructured". If a loop's iter-arg is considered "maybeStructured", we
// must set up their PtrState during `rewriteForOp`.
void PtrAnalysis::initializeMaybeStructuredArgs(Operation *op) {
  std::queue<Value> q;
  DenseSet<Value> visited;

  op->walk([&q, &visited](tts::GetStructuredStateOp getStateOp) {
    Value value = getStateOp->getResult(0);
    visited.insert(value);
    q.push(value);
  });

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto user : v.getUsers()) {
      // scf.for is a special case. We have 2 set of values to consider:
      // - iter-args
      // - loop results
      // for every init arg that originates from a `tts.get_structured_state`
      // op, its corresponding iter-arg and loop result will also be considered
      // "maybeStructured".
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [argIndex, arg] :
             llvm::zip(llvm::index_range(0, forOp.getInitArgs().size()),
                       forOp.getInitArgs())) {
          if (arg != v) {
            continue;
          }
          auto iterArg = forOp.getRegionIterArg(argIndex);
          auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);
          SmallVector<Value> neighbors{iterArg, tiedLoopRes};
          for (auto neighbor : neighbors) {
            maybeStructuredArgs.insert(neighbor);
            if (!visited.contains(neighbor)) {
              visited.insert(neighbor);
              q.push(neighbor);
            }
          }
        }
      } else {
        for (auto res : user->getResults()) {
          if (res.getType() != v.getType()) {
            continue;
          }
          maybeStructuredArgs.insert(res);
          if (!visited.contains(res)) {
            visited.insert(res);
            q.push(res);
          }
        }
      }
    }
  }
}

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op,
                                          bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getValue();
  auto mask = op.getMask();
  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "storeOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: scalar storeOp will not be rewritten"));
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);

  OpBuilder builder(op);

  // Analyze the mask operand to determine at runtime the size of the data
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      LLVM_DEBUG(op->emitRemark("MaskAnalysis failed"));
      return failure();
    }
    ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
    if (!ptr) {
      return failure();
    }
    dims = mstate.dims;
  }

  auto storeOp = tts::StoreOp::create(builder, loc, ptr, val, dims);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::store:\n";
    storeOp->dump();
  });

  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp, bool useUnsafeMask) {
  LLVM_DEBUG({
    llvm::dbgs() << "rewriting rootOp\n";
    rootOp->dump();
  });

  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == rootOp) {
      return WalkResult::advance();
    }
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<triton::AddPtrOp>([&](auto addptr) {
          if (rewriteAddptrOp(addptr).failed()) {
            LLVM_DEBUG(
                addptr->emitRemark("PtrAnalysis: Failed to rewrite AddPtrOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::MakeTensorPtrOp>([&](auto maketptr) {
          if (rewriteMakeTensorPtrOp(maketptr).failed()) {
            LLVM_DEBUG(maketptr->emitRemark(
                "PtrAnalysis: Failed to rewrite MakeTensorPtrOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::AdvanceOp>([&](auto advance) {
          if (rewriteAdvanceOp(advance).failed()) {
            LLVM_DEBUG(advance->emitRemark(
                "PtrAnalysis: Failed to rewrite AdvanceOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::LoadOp>([&](auto load) {
          if (rewriteLoadOp(load, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::StoreOp>([&](auto store) {
          if (rewriteStoreOp(store, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<scf::ForOp>([&](auto forOp) {
          // `rewriteForOp` recursively visits its children, so regardless
          // whether the rewrite succeeds or not, we need to return "skip" so
          // that the the walk does not visit the for-op's child operations
          // the second time.
          if (rewriteForOp(forOp).failed()) {
            LLVM_DEBUG(
                forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp"));
          }
          return WalkResult::skip();
        })
        .Case<tts::GetStructuredStateOp>(
            [&](tts::GetStructuredStateOp getStateOp) {
              // For tensor of indices potentially being used in pointer
              // arithmetic sequence, we need to manually populate the state of
              // none already exists.
              // This process is necessary because unlike triton pointers in a
              // loop which always have a `tt.addptr` that triggers the rewrite
              // process which includes generating the ops for updating offsets
              // and strides, tensor of indices only have a simple `arith.addi`
              // (or other arith ops).
              // Without visiting these ops manually, the ops to update the
              // offsets and strides would not be generated.
              auto tritonValue = getStateOp->getOperand(0);
              if (!PtrExprAnalysis::knownPtrs.contains(tritonValue)) {
                PtrState state;
                OpBuilder b(getStateOp);
                if (succeeded(visitOperand(tritonValue, state,
                                           getStateOp->getLoc(), b)) &&
                    state.isStructured()) {
                  PtrExprAnalysis::knownPtrs[tritonValue] = state;
                } else {
                  LLVM_DEBUG(getStateOp->emitRemark(
                      "PtrAnalysis: Failed to populate ptr "
                      "state for tensor of indices"));
                }
              }

              return WalkResult::skip();
            })
        .Default([&](auto) { return WalkResult::advance(); });
  });

  return success();
}

} // namespace tts
} // namespace mlir
