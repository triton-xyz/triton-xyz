#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"
#include "triton-shared/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <optional>
#include <queue>

#define DEBUG_TYPE "triton-to-tta-unstructured"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTOTTAUNSTRUCTURED
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

// Given a type, return the offset type corresponding to that type with the
// specified width.
// If the type is a tensor, return a tensor of offsets of the same shape. If the
// type is a pointer, return a single offset type.
static Type getPtrOffsetType(Type type, unsigned int bitWidth) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto ptrType =
            dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth));
    }
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    return IntegerType::get(type.getContext(), bitWidth);
  }

  llvm_unreachable("unexpected type");
  return nullptr;
}

static unsigned int getBitWidth(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto integerType = dyn_cast<IntegerType>(tensorType.getElementType())) {
      return integerType.getWidth();
    }
  } else if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }

  llvm_unreachable("unexpected type");
  return 0;
}

static bool isTensorOfPointers(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return false;
  }
  return isa<triton::PointerType>(tensorType.getElementType());
}

static bool shouldHandleForFallback(triton::LoadOp op) {
  if (!isTensorOfPointers(op.getPtr().getType())) {
    return false;
  }

  if (op.getPtr().getDefiningOp<tta::MakeAddrOp>() ||
      op.getPtr().getDefiningOp<tta::ReindexOp>() ||
      op.getPtr().getDefiningOp<tta::AdvanceOp>()) {
    return false;
  }

  if (op.getPtr().getDefiningOp<triton::MakeTensorPtrOp>() &&
      op.getPtr().getDefiningOp<triton::MakeTensorPtrOp>()->hasAttr(
          "tta.fallback")) {
    return true;
  }

  return true;
}

static bool shouldHandleForFallback(triton::StoreOp op) {
  if (!isTensorOfPointers(op.getPtr().getType())) {
    return false;
  }

  if (op.getPtr().getDefiningOp<tta::MakeAddrOp>() ||
      op.getPtr().getDefiningOp<tta::ReindexOp>() ||
      op.getPtr().getDefiningOp<tta::AdvanceOp>()) {
    return false;
  }

  if (op.getPtr().getDefiningOp<triton::MakeTensorPtrOp>() &&
      op.getPtr().getDefiningOp<triton::MakeTensorPtrOp>()->hasAttr(
          "tta.fallback")) {
    return true;
  }

  return true;
}

static SmallVector<ReassociationIndices>
getCollapseTo1DReassociation(unsigned rank) {
  SmallVector<ReassociationIndices> reassociation(1);
  reassociation.front().reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    reassociation.front().push_back(i);
  }
  return reassociation;
}

static FailureOr<Value> flattenTensorTo1D(OpBuilder &builder, Location loc,
                                          Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || tensorType.getRank() == 0 ||
      !tensorType.hasStaticShape()) {
    return failure();
  }

  if (tensorType.getRank() == 1) {
    return value;
  }

  auto flatType = RankedTensorType::get({tensorType.getNumElements()},
                                        tensorType.getElementType());
  auto reassociation = getCollapseTo1DReassociation(tensorType.getRank());
  auto collapse = tensor::CollapseShapeOp::create(builder, loc, flatType, value,
                                                  reassociation);
  return collapse.getResult();
}

static Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *operation) {
                  Type resultType = operation->getResult(0).getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
                    resultType = shapedType.getElementType();
                  }
                  return arith::SIToFPOp::create(builder, loc, resultType, src)
                      .getResult();
                })
                .Case<arith::TruncFOp>([&](Operation *operation) {
                  Type resultType = operation->getResult(0).getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
                    resultType = shapedType.getElementType();
                  }
                  return arith::TruncFOp::create(builder, loc, resultType, src)
                      .getResult();
                })
                .Default([](Operation *) -> Value {
                  llvm_unreachable("unsupported scalar cast op");
                });
    }
    return src;
  };

  while (true) {
    if (!isa<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    }

    if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (!attr.isSplat()) {
          return Value();
        }
        auto elementAttr = attr.getSplatValue<Attribute>();
        auto scalar = arith::ConstantOp::materialize(
            builder, elementAttr, attr.getElementType(), constOp.getLoc());
        return reconstructScalarValue(scalar.getResult());
      }
      return Value();
    }

    if (auto splat = operand.getDefiningOp<triton::SplatOp>()) {
      operand = splat.getSrc();
      continue;
    }

    if (auto sitofp = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(sitofp.getOperation());
      operand = sitofp.getIn();
      continue;
    }

    if (auto truncf = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(truncf.getOperation());
      operand = truncf.getIn();
      continue;
    }

    return Value();
  }
}

static FailureOr<Value> castOffsetToSupportedType(Value offset, Location loc,
                                                  OpBuilder &builder) {
  Type offsetType = offset.getType();

  if (offsetType.isIndex()) {
    return arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                      offset)
        .getResult();
  }

  if (auto intType = dyn_cast<IntegerType>(offsetType)) {
    if (intType.getWidth() == 32 || intType.getWidth() == 64) {
      return offset;
    }
    if (intType.getWidth() < 64) {
      return arith::ExtSIOp::create(builder, loc, builder.getI64Type(), offset)
          .getResult();
    }
    return arith::TruncIOp::create(builder, loc, builder.getI64Type(), offset)
        .getResult();
  }

  if (auto tensorType = dyn_cast<RankedTensorType>(offsetType)) {
    Type elementType = tensorType.getElementType();
    auto targetTensorType =
        RankedTensorType::get(tensorType.getShape(), builder.getI64Type());

    if (elementType.isIndex()) {
      return arith::IndexCastOp::create(builder, loc, targetTensorType, offset)
          .getResult();
    }

    auto intElementType = dyn_cast<IntegerType>(elementType);
    if (!intElementType) {
      return failure();
    }
    if (intElementType.getWidth() == 32 || intElementType.getWidth() == 64) {
      return offset;
    }

    if (intElementType.getWidth() < 64) {
      return arith::ExtSIOp::create(builder, loc, targetTensorType, offset)
          .getResult();
    }
    return arith::TruncIOp::create(builder, loc, targetTensorType, offset)
        .getResult();
  }

  return failure();
}

static FailureOr<Value> normalizeOffsetTo1DTensor(Value offset, Location loc,
                                                  OpBuilder &builder) {
  if (auto tensorType = dyn_cast<RankedTensorType>(offset.getType())) {
    if (tensorType.getRank() == 0) {
      return failure();
    }

    Value normalizedOffset = offset;
    if (tensorType.getRank() > 1) {
      auto maybeFlatOffset = flattenTensorTo1D(builder, loc, offset);
      if (failed(maybeFlatOffset)) {
        return failure();
      }
      normalizedOffset = *maybeFlatOffset;
    }

    return castOffsetToSupportedType(normalizedOffset, loc, builder);
  }

  if (!offset.getType().isIntOrIndex()) {
    return failure();
  }

  auto maybeScalarOffset = castOffsetToSupportedType(offset, loc, builder);
  if (failed(maybeScalarOffset)) {
    return failure();
  }

  auto tensorOffset = tensor::FromElementsOp::create(
                          builder, loc, ValueRange{*maybeScalarOffset})
                          .getResult();
  return tensorOffset;
}

static FailureOr<Value> normalizeMaskTo1DTensor(Value mask, Location loc,
                                                OpBuilder &builder) {
  if (auto tensorType = dyn_cast<RankedTensorType>(mask.getType())) {
    if (!tensorType.getElementType().isInteger(1) ||
        tensorType.getRank() == 0) {
      return failure();
    }
    if (tensorType.getRank() == 1) {
      return mask;
    }
    return flattenTensorTo1D(builder, loc, mask);
  }

  if (!mask.getType().isInteger(1)) {
    return failure();
  }

  auto tensorMask =
      tensor::FromElementsOp::create(builder, loc, ValueRange{mask})
          .getResult();
  return tensorMask;
}

static FailureOr<Value> rebuildLoadResultFrom1DTensor(Value loaded,
                                                      Type targetType,
                                                      Location loc,
                                                      OpBuilder &builder) {
  if (!isa<ShapedType>(targetType)) {
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    return tensor::ExtractOp::create(builder, loc, loaded, ValueRange{zero})
        .getResult();
  }

  auto targetTensorType = dyn_cast<RankedTensorType>(targetType);
  auto loadedType = dyn_cast<RankedTensorType>(loaded.getType());
  if (!targetTensorType || !loadedType || loadedType.getRank() != 1) {
    return failure();
  }

  if (targetTensorType.getRank() == 1) {
    return loaded;
  }

  if (!targetTensorType.hasStaticShape()) {
    return failure();
  }

  auto reassociation = getCollapseTo1DReassociation(targetTensorType.getRank());
  auto expanded = tensor::ExpandShapeOp::create(builder, loc, targetTensorType,
                                                loaded, reassociation);
  return expanded.getResult();
}

static FailureOr<Value> normalizeStoreValueTo1DTensor(Value value, Location loc,
                                                      OpBuilder &builder) {
  if (!isa<RankedTensorType>(value.getType())) {
    return tensor::FromElementsOp::create(builder, loc, ValueRange{value})
        .getResult();
  }

  auto tensorType = cast<RankedTensorType>(value.getType());
  if (tensorType.getRank() == 1) {
    return value;
  }

  return flattenTensorTo1D(builder, loc, value);
}

static FailureOr<Value> getScalarOther(triton::LoadOp load, Location loc,
                                       OpBuilder &builder) {
  if (Value other = load.getOther()) {
    auto scalarOther = getScalarValue(other, loc, builder);
    if (!scalarOther) {
      return failure();
    }
    return scalarOther;
  }

  Type elementType = getElementTypeOrSelf(load.getType());
  auto zeroAttr = builder.getZeroAttr(elementType);
  if (!zeroAttr) {
    return failure();
  }

  return arith::ConstantOp::create(builder, loc, zeroAttr).getResult();
}

static std::optional<int64_t> getStaticOffsetSize(Type offsetType) {
  auto tensorType = dyn_cast<RankedTensorType>(offsetType);
  if (!tensorType || tensorType.getRank() != 1 || tensorType.isDynamicDim(0)) {
    return std::nullopt;
  }
  if (!tensorType.getElementType().isInteger(32) &&
      !tensorType.getElementType().isInteger(64)) {
    return std::nullopt;
  }
  return tensorType.getShape()[0];
}

static FailureOr<tta::MakeAddrOp>
buildLinearMakeAddr(OpBuilder &builder, Location loc, Value ptr, Value offset) {
  if (!isa<triton::PointerType>(ptr.getType())) {
    return failure();
  }

  auto offsetSize = getStaticOffsetSize(offset.getType());
  if (!offsetSize) {
    return failure();
  }

  SmallVector<int64_t> sizes{*offsetSize};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
  SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> shape{builder.getIndexAttr(0)};

  return tta::MakeAddrOp::create(builder, loc, ptr, sizes, strides, offsets,
                                 shape, ArrayRef<int32_t>{});
}
class TritonToTTAUnstructuredPass
    : public mlir::triton::impl::TritonToTTAUnstructuredBase<
          TritonToTTAUnstructuredPass> {
  using Base = mlir::triton::impl::TritonToTTAUnstructuredBase<
      TritonToTTAUnstructuredPass>;
  using Base::Base;

public:
  struct PtrOffset {
    // the source pointer which comes from the kernel argument
    Value ptr;
    // the pointer type that corresponds to this offset; used when
    // creating tts.make_unstructured_tptr
    Type ptrType;
    // bitwidth that is used for this offset, used to track if sign-extension is
    // necessary
    unsigned int bitWidth;
    // the offset value
    Value offset;
  };

  LogicalResult processUnstructuredPtrs(unsigned int defaultBitWidth = 32) {
    llvm::SmallDenseSet<Value> ptrArgs;
    llvm::DenseMap<Value, PtrOffset> offsetMap;
    std::queue<Value> workList;

    llvm::SmallDenseSet<Value> fallbackPtrs;
    llvm::SmallDenseSet<Operation *> fallbackLoadOps;
    llvm::SmallDenseSet<Operation *> fallbackStoreOps;
    getOperation().walk([&](Operation *op) {
      if (auto load = dyn_cast<triton::LoadOp>(op)) {
        if (shouldHandleForFallback(load)) {
          fallbackPtrs.insert(load.getPtr());
          fallbackLoadOps.insert(load.getOperation());
        }
      } else if (auto store = dyn_cast<triton::StoreOp>(op)) {
        if (shouldHandleForFallback(store)) {
          fallbackPtrs.insert(store.getPtr());
          fallbackStoreOps.insert(store.getOperation());
        }
      }
    });

    auto hasFallbackPathUse = [&](Value root) {
      llvm::SmallVector<Value> queue{root};
      llvm::SmallDenseSet<Value> visited;
      while (!queue.empty()) {
        Value current = queue.pop_back_val();
        if (!visited.insert(current).second) {
          continue;
        }
        if (fallbackPtrs.contains(current)) {
          return true;
        }
        for (Operation *user : current.getUsers()) {
          if (auto forOp = dyn_cast<scf::ForOp>(user)) {
            for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
              if (initArg != current) {
                continue;
              }
              queue.push_back(forOp.getRegionIterArg(index));
              queue.push_back(forOp.getResult(index));
            }
            continue;
          }

          for (Value result : user->getResults()) {
            if (triton::isPtrTypeLike(result.getType())) {
              queue.push_back(result);
            }
          }
        }
      }
      return false;
    };

    getOperation().walk([&](FunctionOpInterface func) {
      for (auto arg : func.getArguments()) {
        if (!triton::isPtrTypeLike(arg.getType())) {
          continue;
        }
        if (isTensorOfPointers(arg.getType())) {
          // Unstructured lowering requires a single scalar base pointer.
          // Leave tensor-of-ptr args to the fallback pass.
          continue;
        }

        if (!hasFallbackPathUse(arg)) {
          continue;
        }

        OpBuilder b(func->getRegion(0));
        Value zero =
            arith::ConstantOp::create(
                b, arg.getLoc(),
                b.getIntegerAttr(
                    IntegerType::get(&getContext(), defaultBitWidth), 0))
                .getResult();

        ptrArgs.insert(arg);
        offsetMap.insert({arg, {arg, arg.getType(), defaultBitWidth, zero}});
        workList.push(arg);
      }
    });

    getOperation().walk([&](triton::IntToPtrOp op) {
      // We only want to handle single source pointer,
      // skip if this op produces tensor of pointers
      if (isTensorOfPointers(op.getType())) {
        return;
      }

      if (!hasFallbackPathUse(op.getResult())) {
        return;
      }
      auto res = op.getResult();
      OpBuilder b(op);
      Value zero = arith::ConstantOp::create(
                       b, op.getLoc(),
                       b.getIntegerAttr(
                           IntegerType::get(&getContext(), defaultBitWidth), 0))
                       .getResult();

      offsetMap.insert({res, {res, res.getType(), defaultBitWidth, zero}});
      workList.push(res);
    });

    llvm::SmallVector<Operation *> toDelete;
    llvm::SmallVector<Operation *> ptrUsers;

    while (!workList.empty()) {
      auto val = workList.front();
      workList.pop();

      for (auto &use : val.getUses()) {
        auto user = use.getOwner();

        auto res =
            llvm::TypeSwitch<Operation *, LogicalResult>(user)

                .Case<triton::PtrToIntOp>([&](triton::PtrToIntOp op) {
                  if (isa<RankedTensorType>(op.getType())) {
                    return failure();
                  }

                  auto offsetInfo = offsetMap.at(op.getSrc());

                  OpBuilder b{op};
                  // We are converting a pointer to an integer here,
                  // materialized the pointer using the accumulated offset
                  // that we have stored so far.
                  auto materializedAddPtr =
                      triton::AddPtrOp::create(
                          b, op->getLoc(), offsetInfo.ptrType, offsetInfo.ptr,
                          offsetInfo.offset)
                          .getResult();

                  // Change the op to use the "simplified" pointer above.
                  // This should not affect the traversal of uses, but hacky.
                  // We will need to revisit how we process the IRs in this pass
                  // later.
                  op->setOperand(0, materializedAddPtr);

                  return success();
                })
                .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptr) {
                  // Bail when we have an addptr in an scf.if as we  do not know
                  // if the pointer returning from both branches will have the
                  // same source
                  if (addptr->getParentOfType<scf::IfOp>()) {
                    return failure();
                  }

                  OpBuilder b{addptr};
                  auto loc = addptr->getLoc();

                  auto offsetInfo = offsetMap.at(addptr.getPtr());

                  auto prevOff = offsetInfo.offset;
                  auto off = addptr.getOffset();

                  auto lhsWidth = offsetInfo.bitWidth;
                  auto rhsWidth = getBitWidth(off.getType());
                  auto resWidth = std::max(lhsWidth, rhsWidth);

                  if (lhsWidth < resWidth) {
                    prevOff =
                        arith::ExtSIOp::create(
                            b, loc,
                            getPtrOffsetType(offsetInfo.ptrType, resWidth),
                            prevOff)
                            .getResult();
                  }

                  if (rhsWidth < resWidth) {
                    off =
                        arith::ExtSIOp::create(
                            b, loc,
                            getPtrOffsetType(offsetInfo.ptrType, resWidth), off)
                            .getResult();
                  }

                  auto accumulatedOff =
                      arith::AddIOp::create(
                          b, loc, getPtrOffsetType(addptr.getType(), resWidth),
                          prevOff, off)
                          .getResult();

                  PtrOffset newOffsetInfo{offsetInfo.ptr, addptr.getType(),
                                          resWidth, accumulatedOff};

                  offsetMap.insert({addptr, newOffsetInfo});
                  workList.push(addptr);
                  toDelete.push_back(addptr);

                  return success();
                })
                .Case<triton::SplatOp, triton::BroadcastOp,
                      triton::ExpandDimsOp>([&](Operation *op) {
                  auto res = op->getResult(0);
                  auto resType = res.getType();

                  if (!triton::isPtrTypeLike(resType)) {
                    return success();
                  }

                  auto ptr = op->getOperand(0);
                  auto offsetInfo = offsetMap.at(ptr);

                  OpBuilder b{op};
                  auto clone =
                      b.create(op->getLoc(), op->getName().getIdentifier(),
                               ValueRange{offsetInfo.offset},
                               TypeRange{getPtrOffsetType(
                                   resType, offsetInfo.bitWidth)});

                  PtrOffset newOffsetInfo{offsetInfo.ptr, resType,
                                          offsetInfo.bitWidth,
                                          clone->getResult(0)};

                  offsetMap.insert({
                      res,
                      newOffsetInfo,
                  });
                  workList.push(res);
                  toDelete.push_back(op);

                  return success();
                })
                .Case<triton::BitcastOp>([&](triton::BitcastOp bitcast) {
                  auto res = bitcast.getResult();
                  auto resType = res.getType();

                  if (!triton::isPtrTypeLike(resType)) {
                    return success();
                  }

                  auto offsetInfo = offsetMap.at(bitcast.getSrc());
                  Value basePtr = offsetInfo.ptr;

                  Type newBasePtrType = resType;
                  if (auto tensorType = dyn_cast<RankedTensorType>(resType)) {
                    newBasePtrType = tensorType.getElementType();
                  }

                  if (basePtr.getType() != newBasePtrType) {
                    OpBuilder b{bitcast};
                    basePtr = triton::BitcastOp::create(b, bitcast.getLoc(),
                                                        newBasePtrType, basePtr)
                                  .getResult();
                  }

                  PtrOffset newOffsetInfo{basePtr, resType, offsetInfo.bitWidth,
                                          offsetInfo.offset};

                  offsetMap.insert({res, newOffsetInfo});
                  workList.push(res);
                  toDelete.push_back(bitcast);
                  return success();
                })
                .Case<triton::LoadOp, triton::StoreOp, triton::MakeTensorPtrOp,
                      triton::AtomicRMWOp, triton::AtomicCASOp>(
                    [&](Operation *op) {
                      ptrUsers.push_back(op);
                      return success();
                    })
                .Case<scf::ForOp>([&](scf::ForOp forOp) {
                  // Index of the init-arg corresponding to this use, note that
                  // we have to subtract by 3 from the operand number because
                  // scf.for ops always have 3 leading operands for start, end,
                  // and step.
                  auto argIndex = use.getOperandNumber() - 3;
                  auto init = forOp.getInitArgs()[argIndex];

                  auto offsetInfo = offsetMap.at(init);

                  auto offsetType =
                      getPtrOffsetType(offsetInfo.ptrType, offsetInfo.bitWidth);

                  // We're setting both the types of the iter-arg and the
                  // corresponding result directly to the offset type.
                  // At this point, the IR is in an invalid state because the
                  // init-args still have tt.ptr. But at the end, we will
                  // replace all uses of the tt.ptr to offset values.
                  auto iterArg = forOp.getRegionIterArg(argIndex);
                  iterArg.setType(offsetType);

                  auto res = forOp.getResult(argIndex);
                  res.setType(offsetType);

                  // For other ops, we only need to push the result into the
                  // worklist. But for scf.for, the iter-arg corresponding to
                  // the init-arg is used in the op's body instead, we have to
                  // process uses of the iter-arg.
                  PtrOffset iterArgOffset{offsetInfo.ptr, offsetInfo.ptrType,
                                          offsetInfo.bitWidth, iterArg};
                  offsetMap.insert({
                      iterArg,
                      iterArgOffset,
                  });

                  PtrOffset resOffset{offsetInfo.ptr, offsetInfo.ptrType,
                                      offsetInfo.bitWidth, res};
                  offsetMap.insert({
                      res,
                      resOffset,
                  });
                  workList.push(iterArg);
                  workList.push(res);

                  return success();
                })
                .Case<scf::YieldOp>([](auto) { return success(); })
                .Case<triton::CatOp>([](triton::CatOp op) {
                  op->emitRemark("Do not support gather / scatter with "
                                 "multiple bases yet");
                  return failure();
                })
                .Default([&](Operation *op) {
                  LLVM_DEBUG(op->emitRemark("unexpected op in ptr sequence"));
                  return failure();
                });

        if (failed(res)) {
          return failure();
        }
      }
    }

    for (auto op : ptrUsers) {
      OpBuilder b{op};
      auto loc = op->getLoc();
      auto res =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<triton::LoadOp>([&](triton::LoadOp load) {
                if (!fallbackLoadOps.contains(load.getOperation())) {
                  return success();
                }
                auto offsetInfo = offsetMap.at(load.getPtr());

                auto maybeFlatOffset =
                    normalizeOffsetTo1DTensor(offsetInfo.offset, loc, b);
                if (failed(maybeFlatOffset)) {
                  load->emitRemark("cannot normalize offset to 1D tensor");
                  return failure();
                }

                auto maybeMakeAddr = buildLinearMakeAddr(b, loc, offsetInfo.ptr,
                                                         *maybeFlatOffset);
                if (failed(maybeMakeAddr)) {
                  load->emitRemark("cannot build linear tta.make_addr");
                  return failure();
                }

                SmallVector<OpFoldResult> zeroOffsets{b.getIndexAttr(0)};
                tta::ReindexOp reindex;
                if (Value mask = load.getMask()) {
                  auto maybeFlatMask = normalizeMaskTo1DTensor(mask, loc, b);
                  if (failed(maybeFlatMask)) {
                    load->emitRemark("cannot normalize mask to 1D tensor");
                    return failure();
                  }

                  reindex = tta::ReindexOp::create(
                      b, loc, maybeMakeAddr->getResult(), *maybeFlatOffset,
                      *maybeFlatMask, /*indirectDim=*/0, zeroOffsets);
                } else {
                  reindex = tta::ReindexOp::create(
                      b, loc, maybeMakeAddr->getResult(), *maybeFlatOffset,
                      /*indirectDim=*/0, zeroOffsets);
                }

                auto scalarOther = getScalarOther(load, loc, b);
                if (failed(scalarOther)) {
                  load->emitRemark("cannot parse scalar `other` value");
                  return failure();
                }

                auto ttaLoad =
                    tta::LoadOp::create(b, loc, reindex.getResult(),
                                        ArrayRef<OpFoldResult>{}, *scalarOther);

                auto maybeResult = rebuildLoadResultFrom1DTensor(
                    ttaLoad.getResult(), load.getType(), loc, b);
                if (failed(maybeResult)) {
                  load->emitRemark("cannot rebuild load result shape");
                  return failure();
                }

                load.getResult().replaceAllUsesWith(*maybeResult);
                load->erase();
                return success();
              })
              .Case<triton::StoreOp>([&](triton::StoreOp store) {
                if (!fallbackStoreOps.contains(store.getOperation())) {
                  return success();
                }
                auto offsetInfo = offsetMap.at(store.getPtr());

                auto maybeFlatOffset =
                    normalizeOffsetTo1DTensor(offsetInfo.offset, loc, b);
                if (failed(maybeFlatOffset)) {
                  store->emitRemark("cannot normalize offset to 1D tensor");
                  return failure();
                }

                auto maybeMakeAddr = buildLinearMakeAddr(b, loc, offsetInfo.ptr,
                                                         *maybeFlatOffset);
                if (failed(maybeMakeAddr)) {
                  store->emitRemark("cannot build linear tta.make_addr");
                  return failure();
                }

                SmallVector<OpFoldResult> zeroOffsets{b.getIndexAttr(0)};
                tta::ReindexOp reindex;
                if (Value mask = store.getMask()) {
                  auto maybeFlatMask = normalizeMaskTo1DTensor(mask, loc, b);
                  if (failed(maybeFlatMask)) {
                    store->emitRemark("cannot normalize mask to 1D tensor");
                    return failure();
                  }

                  reindex = tta::ReindexOp::create(
                      b, loc, maybeMakeAddr->getResult(), *maybeFlatOffset,
                      *maybeFlatMask, /*indirectDim=*/0, zeroOffsets);
                } else {
                  reindex = tta::ReindexOp::create(
                      b, loc, maybeMakeAddr->getResult(), *maybeFlatOffset,
                      /*indirectDim=*/0, zeroOffsets);
                }

                auto maybeFlatValue =
                    normalizeStoreValueTo1DTensor(store.getValue(), loc, b);
                if (failed(maybeFlatValue)) {
                  store->emitRemark(
                      "cannot normalize store value to 1D tensor");
                  return failure();
                }

                tta::StoreOp::create(b, loc, reindex.getResult(),
                                     *maybeFlatValue, ArrayRef<OpFoldResult>{});
                store->erase();
                return success();
              })
              .Case<triton::MakeTensorPtrOp>([&](triton::MakeTensorPtrOp
                                                     makeTensorPtr) {
                // For block pointers, the base could come from a sequence of
                // `tt.addptr`. Accumulate the target offset with the offset
                // we have saved.
                auto offsetInfo = offsetMap.at(makeTensorPtr.getBase());
                auto baseOffset = offsetInfo.offset;

                makeTensorPtr.getBaseMutable().set(offsetInfo.ptr);

                // Add the existing offset from the base to the offset
                // operand in the ops.
                auto &offsetOpnd = makeTensorPtr.getOffsetsMutable()[0];
                auto currOffset = offsetOpnd.get();

                auto baseOffType = baseOffset.getType();
                auto currOffType = currOffset.getType();

                if (baseOffType != currOffType) {
                  if (currOffType.isIndex()) {
                    baseOffset = arith::IndexCastOp::create(
                                     b, loc, b.getIndexType(), baseOffset)
                                     .getResult();
                  } else if (currOffType.isInteger()) {
                    if (baseOffType.getIntOrFloatBitWidth() <
                        currOffType.getIntOrFloatBitWidth()) {
                      baseOffset = arith::ExtSIOp::create(b, loc, currOffType,
                                                          baseOffset)
                                       .getResult();
                    } else {
                      // MakeTensorPtrOp only takes i32 offsets, so we need
                      // to truncate if the offsets were already in i64
                      makeTensorPtr.emitWarning(
                          "truncating offsets which may result in data loss");
                      baseOffset = arith::TruncIOp::create(b, loc, currOffType,
                                                           baseOffset)
                                       .getResult();
                    }
                  }
                }

                auto accumulatedOffset =
                    arith::AddIOp::create(b, loc, currOffset.getType(),
                                          baseOffset, currOffset)
                        .getResult();

                offsetOpnd.set(accumulatedOffset);

                return success();
              })

              .Case<triton::AtomicRMWOp>([&](triton::AtomicRMWOp atomic) {
                auto offsetInfo = offsetMap.at(atomic.getPtr());
                OpBuilder b{atomic};
                auto loc = atomic.getLoc();
                Value basePtr = offsetInfo.ptr;
                if (auto tensorType =
                        dyn_cast<RankedTensorType>(offsetInfo.ptrType)) {
                  basePtr = triton::SplatOp::create(b, loc, tensorType, basePtr)
                                .getResult();
                }
                auto materializedAddPtr =
                    triton::AddPtrOp::create(b, loc, offsetInfo.ptrType,
                                             basePtr, offsetInfo.offset)
                        .getResult();
                atomic.getPtrMutable().set(materializedAddPtr);
                return success();
              })
              .Case<triton::AtomicCASOp>([&](triton::AtomicCASOp atomic) {
                auto offsetInfo = offsetMap.at(atomic.getPtr());
                OpBuilder b{atomic};
                auto loc = atomic.getLoc();
                Value basePtr = offsetInfo.ptr;
                if (auto tensorType =
                        dyn_cast<RankedTensorType>(offsetInfo.ptrType)) {
                  basePtr = triton::SplatOp::create(b, loc, tensorType, basePtr)
                                .getResult();
                }
                auto materializedAddPtr =
                    triton::AddPtrOp::create(b, loc, offsetInfo.ptrType,
                                             basePtr, offsetInfo.offset)
                        .getResult();
                atomic.getPtrMutable().set(materializedAddPtr);
                return success();
              })
              .Default([&](Operation *op) {
                LLVM_DEBUG(op->emitRemark("unexpected op in ptr sequence"));
                return failure();
              });

      if (failed(res)) {
        return failure();
      }
    }

    for (auto op : toDelete) {
      auto ptrInfo = offsetMap.at(op->getResult(0));
      op->replaceAllUsesWith(ValueRange{ptrInfo.offset});
      op->erase();
    }

    return success();
  }

  void runOnOperation() override {
    if (failed(processUnstructuredPtrs(offsetBitWidth))) {
      LLVM_DEBUG(getOperation()->emitRemark(
          "Cannot transform tensor of pointers into a "
          "single base pointer with tensor of offsets"));
      return;
    }
  }
};
} // namespace
