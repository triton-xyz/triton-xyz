#include "TTAFallbackUtils.h"
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

using mlir::triton::tta_conversion::hasLoweredTTAAddressRoot;
using mlir::triton::tta_conversion::markFallback;

// Given a type, return the offset type corresponding to that type with the
// specified width.
// If the type is a tensor, return a tensor of offsets of the same shape. If the
// type is a pointer, return a single offset type.
static Type getPtrOffsetType(Type type, unsigned int bitWidth) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (isa<triton::PointerType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth));
    }
  }

  if (isa<triton::PointerType>(type)) {
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

static bool isScalarPointer(Type type) {
  auto ptrType = dyn_cast<triton::PointerType>(type);
  if (!ptrType) {
    return false;
  }

  return !isa<RankedTensorType>(ptrType.getPointeeType());
}

static bool isValueDefinedInsideOp(Operation *scope, Value value) {
  if (!scope || !value) {
    return false;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    return owner && scope->isAncestor(owner);
  }

  if (Operation *def = value.getDefiningOp()) {
    return scope->isAncestor(def);
  }

  return false;
}

template <typename LoadStoreLikeOp>
static bool shouldHandleForFallback(LoadStoreLikeOp op) {
  Type ptrType = op.getPtr().getType();
  if (!isTensorOfPointers(ptrType) && !isScalarPointer(ptrType)) {
    return false;
  }

  return !hasLoweredTTAAddressRoot(op.getPtr());
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

static std::optional<StringRef> getTTAAtomicKind(triton::RMWOp rmwOp) {
  switch (rmwOp) {
  case triton::RMWOp::ADD:
    return StringRef("add");
  case triton::RMWOp::AND:
    return StringRef("and");
  case triton::RMWOp::OR:
    return StringRef("or");
  case triton::RMWOp::XOR:
    return StringRef("xor");
  case triton::RMWOp::MAX:
    return StringRef("max");
  case triton::RMWOp::MIN:
    return StringRef("min");
  case triton::RMWOp::XCHG:
    return StringRef("xchg");
  case triton::RMWOp::FADD:
    return StringRef("fadd");
  case triton::RMWOp::UMAX:
  case triton::RMWOp::UMIN:
    return std::nullopt;
  }
  return std::nullopt;
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

static FailureOr<Value> buildLinearImportedAddr(OpBuilder &builder,
                                                Location loc, Value ptr,
                                                Value offset) {
  auto maybeMakeAddr = buildLinearMakeAddr(builder, loc, ptr, offset);
  if (failed(maybeMakeAddr)) {
    return failure();
  }
  return maybeMakeAddr->getResult();
}

static Value createZeroOffset(OpBuilder &builder, Location loc,
                              unsigned int bitWidth) {
  return arith::ConstantOp::create(
             builder, loc,
             builder.getIntegerAttr(builder.getIntegerType(bitWidth), 0))
      .getResult();
}

static Value buildLinearIndirectReindex(OpBuilder &builder, Location loc,
                                        Value importedAddr, Value flatOffset,
                                        Value flatMask = Value()) {
  if (flatMask) {
    return tta::IndirectReindexOp::create(builder, loc, importedAddr,
                                          flatOffset, flatMask,
                                          /*indirectDim=*/0)
        .getResult();
  }
  return tta::IndirectReindexOp::create(builder, loc, importedAddr, flatOffset,
                                        /*indirectDim=*/0)
      .getResult();
}

static Value materializeAddPtrFromOffset(OpBuilder &builder, Location loc,
                                         Value basePtr, Type ptrType,
                                         Value offset) {
  if (auto tensorPtrType = dyn_cast<RankedTensorType>(ptrType)) {
    if (basePtr.getType() != tensorPtrType) {
      basePtr = triton::SplatOp::create(builder, loc, tensorPtrType, basePtr)
                    .getResult();
    }
  }

  return triton::AddPtrOp::create(builder, loc, ptrType, basePtr, offset)
      .getResult();
}

static FailureOr<Value> buildAtomicImportedAddr(OpBuilder &builder,
                                                Location loc, Value ptr) {
  if (!isa<triton::PointerType>(ptr.getType())) {
    return failure();
  }

  SmallVector<int64_t> sizes{1};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1)};
  SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> shape{builder.getIndexAttr(0)};
  return tta::MakeAddrOp::create(builder, loc, ptr, sizes, strides, offsets,
                                 shape, ArrayRef<int32_t>{})
      .getResult();
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

  bool shouldSkipLowering(Value root) const {
    if (!root) {
      return true;
    }
    return hasLoweredTTAAddressRoot(root);
  }

  template <typename T>
  bool tryGetOffsetInfo(Value ptr, llvm::DenseMap<Value, T> &offsetMap,
                        T &out) const {
    auto it = offsetMap.find(ptr);
    if (it == offsetMap.end()) {
      return false;
    }
    out = it->second;
    return true;
  }

  LogicalResult processUnstructuredPtrs(unsigned int defaultBitWidth = 32) {
    llvm::DenseMap<Value, PtrOffset> offsetMap;
    std::queue<Value> workList;

    llvm::SmallDenseSet<Value> fallbackPtrs;
    llvm::SmallDenseSet<Value> atomicPtrs;
    llvm::SmallDenseSet<Operation *> fallbackLoadOps;
    llvm::SmallDenseSet<Operation *> fallbackStoreOps;
    llvm::SmallDenseSet<Value> preservedPtrValues;

    auto preservePtrChain = [&](Value root) {
      if (!root || !triton::isPtrTypeLike(root.getType())) {
        return;
      }

      llvm::SmallVector<Value> queue{root};
      llvm::SmallDenseSet<Value> visited;
      while (!queue.empty()) {
        Value current = queue.pop_back_val();
        if (!current || !triton::isPtrTypeLike(current.getType()) ||
            !visited.insert(current).second) {
          continue;
        }

        preservedPtrValues.insert(current);

        if (Operation *def = current.getDefiningOp()) {
          for (Value operand : def->getOperands()) {
            if (triton::isPtrTypeLike(operand.getType())) {
              queue.push_back(operand);
            }
          }
          continue;
        }

        auto blockArg = dyn_cast<BlockArgument>(current);
        if (!blockArg) {
          continue;
        }

        Block *owner = blockArg.getOwner();
        auto forOp = dyn_cast_or_null<scf::ForOp>(owner->getParentOp());
        if (!forOp || owner != &forOp.getRegion().front()) {
          continue;
        }

        unsigned argNumber = blockArg.getArgNumber();
        if (argNumber == 0) {
          continue;
        }

        unsigned iterIndex = argNumber - 1;
        if (iterIndex >= forOp.getInitArgs().size()) {
          continue;
        }

        Value initArg = forOp.getInitArgs()[iterIndex];
        if (triton::isPtrTypeLike(initArg.getType())) {
          queue.push_back(initArg);
        }
        Value result = forOp.getResult(iterIndex);
        if (triton::isPtrTypeLike(result.getType())) {
          queue.push_back(result);
        }
      }
    };

    auto markFallbackAndPreserve = [&](Operation *op, StringRef reason) {
      markFallback(op, reason);
      for (Value operand : op->getOperands()) {
        preservePtrChain(operand);
      }
      for (Value result : op->getResults()) {
        preservePtrChain(result);
      }
    };

    auto tryGetOffsetInfoOrFallback =
        [&](Operation *op, Value ptr, PtrOffset &out,
            StringRef reason = "offset_info_missing") {
          if (tryGetOffsetInfo(ptr, offsetMap, out)) {
            return true;
          }
          markFallbackAndPreserve(op, reason);
          return false;
        };

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
      } else if (auto atomic = dyn_cast<triton::AtomicRMWOp>(op)) {
        atomicPtrs.insert(atomic.getPtr());
      } else if (auto atomic = dyn_cast<triton::AtomicCASOp>(op)) {
        atomicPtrs.insert(atomic.getPtr());
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
        if (fallbackPtrs.contains(current) || atomicPtrs.contains(current)) {
          return true;
        }
        for (OpOperand &use : current.getUses()) {
          Operation *user = use.getOwner();

          if (auto forOp = dyn_cast<scf::ForOp>(user)) {
            unsigned operandNumber = use.getOperandNumber();
            if (operandNumber < 3) {
              continue;
            }
            unsigned index = operandNumber - 3;
            if (index >= forOp.getInitArgs().size()) {
              continue;
            }
            queue.push_back(forOp.getRegionIterArg(index));
            queue.push_back(forOp.getResult(index));
            continue;
          }

          if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
            auto ifOp = dyn_cast_or_null<scf::IfOp>(yieldOp->getParentOp());
            if (!ifOp) {
              continue;
            }
            unsigned resultIndex = use.getOperandNumber();
            if (resultIndex >= ifOp.getNumResults()) {
              continue;
            }
            Value ifResult = ifOp.getResult(resultIndex);
            if (triton::isPtrTypeLike(ifResult.getType())) {
              queue.push_back(ifResult);
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
        Value zero = createZeroOffset(b, arg.getLoc(), defaultBitWidth);

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
      Value zero = createZeroOffset(b, op.getLoc(), defaultBitWidth);

      offsetMap.insert({res, {res, res.getType(), defaultBitWidth, zero}});
      workList.push(res);
    });

    llvm::SmallVector<Operation *> toDelete;
    llvm::SmallVector<Operation *> ptrUsers;

    while (!workList.empty()) {
      auto val = workList.front();
      workList.pop();

      if (preservedPtrValues.contains(val)) {
        continue;
      }

      for (auto &use : val.getUses()) {
        auto user = use.getOwner();

        auto res =
            llvm::TypeSwitch<Operation *, LogicalResult>(user)

                .Case<triton::PtrToIntOp>([&](triton::PtrToIntOp op) {
                  if (isa<RankedTensorType>(op.getType())) {
                    markFallbackAndPreserve(
                        op, "ptr_to_int_tensor_result_unsupported");
                    return success();
                  }

                  PtrOffset offsetInfo;
                  if (!tryGetOffsetInfoOrFallback(op, op.getSrc(),
                                                  offsetInfo)) {
                    return success();
                  }

                  OpBuilder b{op};
                  // We are converting a pointer to an integer here,
                  // materialized the pointer using the accumulated offset
                  // that we have stored so far.
                  auto materializedAddPtr = materializeAddPtrFromOffset(
                      b, op->getLoc(), offsetInfo.ptr, offsetInfo.ptrType,
                      offsetInfo.offset);

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
                    markFallbackAndPreserve(addptr,
                                            "addptr_in_scf_if_unsupported");
                    return success();
                  }

                  OpBuilder b{addptr};
                  auto loc = addptr->getLoc();

                  PtrOffset offsetInfo;
                  if (!tryGetOffsetInfoOrFallback(addptr, addptr.getPtr(),
                                                  offsetInfo)) {
                    return success();
                  }

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
                  PtrOffset offsetInfo;
                  if (!tryGetOffsetInfoOrFallback(op, ptr, offsetInfo)) {
                    return success();
                  }

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

                  PtrOffset offsetInfo;
                  if (!tryGetOffsetInfoOrFallback(bitcast, bitcast.getSrc(),
                                                  offsetInfo)) {
                    return success();
                  }
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
                .Case<arith::SelectOp>([&](arith::SelectOp select) {
                  auto res = select.getResult();
                  auto resType = res.getType();

                  if (!triton::isPtrTypeLike(resType)) {
                    return success();
                  }

                  PtrOffset trueInfo;
                  bool hasTrue = tryGetOffsetInfo(select.getTrueValue(),
                                                  offsetMap, trueInfo);

                  PtrOffset falseInfo;
                  bool hasFalse = tryGetOffsetInfo(select.getFalseValue(),
                                                   offsetMap, falseInfo);

                  if (!hasTrue || !hasFalse) {
                    return success();
                  }

                  if (trueInfo.ptr != falseInfo.ptr) {
                    markFallbackAndPreserve(select,
                                            "select_multi_base_unsupported");
                    return success();
                  }

                  OpBuilder b{select};
                  auto loc = select.getLoc();
                  auto resWidth =
                      std::max(trueInfo.bitWidth, falseInfo.bitWidth);
                  auto resOffsetType = getPtrOffsetType(resType, resWidth);

                  Value trueOffset = trueInfo.offset;
                  if (trueInfo.bitWidth < resWidth) {
                    trueOffset = arith::ExtSIOp::create(b, loc, resOffsetType,
                                                        trueOffset)
                                     .getResult();
                  }

                  Value falseOffset = falseInfo.offset;
                  if (falseInfo.bitWidth < resWidth) {
                    falseOffset = arith::ExtSIOp::create(b, loc, resOffsetType,
                                                         falseOffset)
                                      .getResult();
                  }

                  auto selectedOffset =
                      arith::SelectOp::create(b, loc, select.getCondition(),
                                              trueOffset, falseOffset)
                          .getResult();

                  PtrOffset newOffsetInfo{trueInfo.ptr, resType, resWidth,
                                          selectedOffset};
                  offsetMap.insert({res, newOffsetInfo});
                  workList.push(res);
                  toDelete.push_back(select);
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

                  PtrOffset offsetInfo;
                  if (!tryGetOffsetInfoOrFallback(
                          forOp, init, offsetInfo,
                          "iter_arg_offset_info_missing")) {
                    return success();
                  }

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
                .Case<scf::YieldOp>([&](scf::YieldOp yieldOp) {
                  auto ifOp =
                      dyn_cast_or_null<scf::IfOp>(yieldOp->getParentOp());
                  if (!ifOp) {
                    return success();
                  }

                  unsigned resultIndex = use.getOperandNumber();
                  if (resultIndex >= ifOp.getNumResults()) {
                    return success();
                  }

                  Value ifResult = ifOp.getResult(resultIndex);
                  Type ifResultType = ifResult.getType();
                  if (!triton::isPtrTypeLike(ifResultType)) {
                    return success();
                  }

                  PtrOffset existing;
                  if (tryGetOffsetInfo(ifResult, offsetMap, existing)) {
                    return success();
                  }

                  auto thenYield = dyn_cast<scf::YieldOp>(
                      ifOp.getThenRegion().front().getTerminator());
                  auto elseYield = dyn_cast_or_null<scf::YieldOp>(
                      ifOp.elseBlock() ? ifOp.elseBlock()->getTerminator()
                                       : nullptr);
                  if (!thenYield || !elseYield ||
                      resultIndex >= thenYield.getNumOperands() ||
                      resultIndex >= elseYield.getNumOperands()) {
                    markFallback(ifOp, "if_yield_mismatch");
                    preservePtrChain(ifResult);
                    return success();
                  }

                  Value thenPtr = thenYield.getOperand(resultIndex);
                  Value elsePtr = elseYield.getOperand(resultIndex);

                  PtrOffset thenInfo;
                  bool hasThen = tryGetOffsetInfo(thenPtr, offsetMap, thenInfo);
                  PtrOffset elseInfo;
                  bool hasElse = tryGetOffsetInfo(elsePtr, offsetMap, elseInfo);
                  if (!hasThen || !hasElse) {
                    return success();
                  }

                  OpBuilder b{ifOp};
                  auto loc = ifOp.getLoc();

                  Value selectedBasePtr = thenInfo.ptr;
                  if (thenInfo.ptr != elseInfo.ptr) {
                    if (thenInfo.ptr.getType() != elseInfo.ptr.getType()) {
                      markFallback(ifOp, "if_multi_base_type_mismatch");
                      preservePtrChain(thenPtr);
                      preservePtrChain(elsePtr);
                      preservePtrChain(ifResult);
                      return success();
                    }
                    if (!isa<triton::PointerType>(thenInfo.ptr.getType())) {
                      markFallback(ifOp, "if_multi_base_non_scalar_ptr");
                      preservePtrChain(thenPtr);
                      preservePtrChain(elsePtr);
                      preservePtrChain(ifResult);
                      return success();
                    }

                    selectedBasePtr =
                        arith::SelectOp::create(b, loc, ifOp.getCondition(),
                                                thenInfo.ptr, elseInfo.ptr)
                            .getResult();
                  }

                  if (isValueDefinedInsideOp(ifOp, thenInfo.offset) ||
                      isValueDefinedInsideOp(ifOp, elseInfo.offset)) {
                    markFallback(ifOp, "if_internal_offset_scope_unsupported");
                    preservePtrChain(thenPtr);
                    preservePtrChain(elsePtr);
                    preservePtrChain(ifResult);
                    return success();
                  }

                  auto resWidth =
                      std::max(thenInfo.bitWidth, elseInfo.bitWidth);
                  auto resOffsetType = getPtrOffsetType(ifResultType, resWidth);

                  Value thenOffset = thenInfo.offset;
                  if (thenInfo.bitWidth < resWidth) {
                    thenOffset = arith::ExtSIOp::create(b, loc, resOffsetType,
                                                        thenOffset)
                                     .getResult();
                  }

                  Value elseOffset = elseInfo.offset;
                  if (elseInfo.bitWidth < resWidth) {
                    elseOffset = arith::ExtSIOp::create(b, loc, resOffsetType,
                                                        elseOffset)
                                     .getResult();
                  }

                  auto selectedOffset =
                      arith::SelectOp::create(b, loc, ifOp.getCondition(),
                                              thenOffset, elseOffset)
                          .getResult();

                  PtrOffset newOffsetInfo{selectedBasePtr, ifResultType,
                                          resWidth, selectedOffset};
                  offsetMap.insert({ifResult, newOffsetInfo});
                  workList.push(ifResult);

                  // Keep the original ptr-producing chains until users of the
                  // scf.if result are fully rewritten.
                  preservePtrChain(thenPtr);
                  preservePtrChain(elsePtr);

                  return success();
                })
                .Case<triton::CatOp>([&](triton::CatOp op) {
                  markFallbackAndPreserve(op, "multi_base_cat_unsupported");
                  return success();
                })
                .Default([&](Operation *op) {
                  markFallbackAndPreserve(op, "unexpected_ptr_sequence_op");
                  return success();
                });

        if (failed(res)) {
          continue;
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

                if (shouldSkipLowering(load.getPtr())) {
                  markFallbackAndPreserve(load, "skip_due_to_fallback_root");
                  return success();
                }

                PtrOffset offsetInfo;
                if (!tryGetOffsetInfoOrFallback(load, load.getPtr(),
                                                offsetInfo)) {
                  return success();
                }

                auto maybeFlatOffset =
                    normalizeOffsetTo1DTensor(offsetInfo.offset, loc, b);
                if (failed(maybeFlatOffset)) {
                  markFallbackAndPreserve(load, "offset_not_1d_normalizable");
                  return success();
                }

                auto maybeAddr = buildLinearImportedAddr(b, loc, offsetInfo.ptr,
                                                         *maybeFlatOffset);
                if (failed(maybeAddr)) {
                  markFallbackAndPreserve(load, "make_addr_build_failed");
                  return success();
                }

                Value flatMask;
                if (Value mask = load.getMask()) {
                  auto maybeFlatMask = normalizeMaskTo1DTensor(mask, loc, b);
                  if (failed(maybeFlatMask)) {
                    markFallbackAndPreserve(load, "mask_not_1d_normalizable");
                    return success();
                  }
                  flatMask = *maybeFlatMask;
                }

                Value reindex = buildLinearIndirectReindex(
                    b, loc, *maybeAddr, *maybeFlatOffset, flatMask);

                auto scalarOther = getScalarOther(load, loc, b);
                if (failed(scalarOther)) {
                  markFallbackAndPreserve(load, "other_not_scalar_splat");
                  return success();
                }

                auto flatOffsetType =
                    dyn_cast<RankedTensorType>((*maybeFlatOffset).getType());
                if (!flatOffsetType) {
                  load->emitRemark("flat offset must be ranked tensor");
                  return failure();
                }

                auto flatLoadType =
                    RankedTensorType::get(flatOffsetType.getShape(),
                                          getElementTypeOrSelf(load.getType()));

                auto ttaLoad = tta::LoadOp::create(
                    b, loc, flatLoadType, reindex, ArrayRef<Value>{},
                    b.getDenseI64ArrayAttr({}), *scalarOther);

                auto maybeResult = rebuildLoadResultFrom1DTensor(
                    ttaLoad.getResult(), load.getType(), loc, b);
                if (failed(maybeResult)) {
                  markFallbackAndPreserve(load,
                                          "load_result_shape_rebuild_failed");
                  return success();
                }

                load.getResult().replaceAllUsesWith(*maybeResult);
                load->erase();
                return success();
              })
              .Case<triton::StoreOp>([&](triton::StoreOp store) {
                if (!fallbackStoreOps.contains(store.getOperation())) {
                  return success();
                }

                if (shouldSkipLowering(store.getPtr())) {
                  markFallbackAndPreserve(store, "skip_due_to_fallback_root");
                  return success();
                }

                PtrOffset offsetInfo;
                if (!tryGetOffsetInfoOrFallback(store, store.getPtr(),
                                                offsetInfo)) {
                  return success();
                }

                auto maybeFlatOffset =
                    normalizeOffsetTo1DTensor(offsetInfo.offset, loc, b);
                if (failed(maybeFlatOffset)) {
                  markFallbackAndPreserve(store, "offset_not_1d_normalizable");
                  return success();
                }

                auto maybeAddr = buildLinearImportedAddr(b, loc, offsetInfo.ptr,
                                                         *maybeFlatOffset);
                if (failed(maybeAddr)) {
                  markFallbackAndPreserve(store, "make_addr_build_failed");
                  return success();
                }

                Value flatMask;
                if (Value mask = store.getMask()) {
                  auto maybeFlatMask = normalizeMaskTo1DTensor(mask, loc, b);
                  if (failed(maybeFlatMask)) {
                    markFallbackAndPreserve(store, "mask_not_1d_normalizable");
                    return success();
                  }
                  flatMask = *maybeFlatMask;
                }

                Value reindex = buildLinearIndirectReindex(
                    b, loc, *maybeAddr, *maybeFlatOffset, flatMask);

                auto maybeFlatValue =
                    normalizeStoreValueTo1DTensor(store.getValue(), loc, b);
                if (failed(maybeFlatValue)) {
                  markFallbackAndPreserve(store,
                                          "store_value_not_1d_normalizable");
                  return success();
                }

                tta::StoreOp::create(b, loc, reindex, *maybeFlatValue,
                                     ArrayRef<OpFoldResult>{});
                store->erase();
                return success();
              })
              .Case<triton::MakeTensorPtrOp>([&](triton::MakeTensorPtrOp
                                                     makeTensorPtr) {
                if (shouldSkipLowering(makeTensorPtr.getBase())) {
                  markFallbackAndPreserve(makeTensorPtr,
                                          "skip_due_to_fallback_root");
                  return success();
                }

                // For block pointers, the base could come from a sequence of
                // `tt.addptr`. Accumulate the target offset with the offset
                // we have saved.
                PtrOffset offsetInfo;
                if (!tryGetOffsetInfoOrFallback(
                        makeTensorPtr, makeTensorPtr.getBase(), offsetInfo)) {
                  return success();
                }
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
                if (shouldSkipLowering(atomic.getPtr())) {
                  markFallbackAndPreserve(atomic, "skip_due_to_fallback_root");
                  return success();
                }

                PtrOffset offsetInfo;
                if (!tryGetOffsetInfoOrFallback(atomic, atomic.getPtr(),
                                                offsetInfo)) {
                  return success();
                }
                OpBuilder b{atomic};
                auto loc = atomic.getLoc();

                auto materializeAtomicPtr = [&]() {
                  auto materializedAddPtr = materializeAddPtrFromOffset(
                      b, loc, offsetInfo.ptr, offsetInfo.ptrType,
                      offsetInfo.offset);
                  atomic.getPtrMutable().set(materializedAddPtr);
                };

                if (isa<ShapedType>(atomic.getType())) {
                  materializeAtomicPtr();
                  return success();
                }

                auto maybeAtomicKind =
                    getTTAAtomicKind(atomic.getAtomicRmwOp());
                if (!maybeAtomicKind.has_value()) {
                  markFallbackAndPreserve(atomic, "atomic_kind_unsupported");
                  materializeAtomicPtr();
                  return success();
                }

                if (!isa<triton::PointerType>(offsetInfo.ptr.getType())) {
                  markFallbackAndPreserve(atomic, "atomic_base_ptr_not_scalar");
                  materializeAtomicPtr();
                  return success();
                }

                auto maybeAtomicOffset =
                    castOffsetToSupportedType(offsetInfo.offset, loc, b);
                if (failed(maybeAtomicOffset)) {
                  markFallbackAndPreserve(atomic, "atomic_offset_cast_failed");
                  materializeAtomicPtr();
                  return success();
                }

                auto maybeImportedPtr =
                    buildAtomicImportedAddr(b, loc, offsetInfo.ptr);
                if (failed(maybeImportedPtr)) {
                  markFallbackAndPreserve(atomic, "make_addr_build_failed");
                  materializeAtomicPtr();
                  return success();
                }

                tta::AtomicOp ttaAtomic;
                if (Value mask = atomic.getMask()) {
                  ttaAtomic = tta::AtomicOp::create(
                      b, loc, *maybeAtomicKind, *maybeImportedPtr,
                      *maybeAtomicOffset, atomic.getVal(), mask);
                } else {
                  ttaAtomic = tta::AtomicOp::create(
                      b, loc, *maybeAtomicKind, *maybeImportedPtr,
                      *maybeAtomicOffset, atomic.getVal());
                }
                atomic.replaceAllUsesWith(ttaAtomic.getResult());
                atomic->erase();
                return success();
              })
              .Case<triton::AtomicCASOp>([&](triton::AtomicCASOp atomic) {
                if (shouldSkipLowering(atomic.getPtr())) {
                  markFallbackAndPreserve(atomic, "skip_due_to_fallback_root");
                  return success();
                }

                PtrOffset offsetInfo;
                if (!tryGetOffsetInfoOrFallback(atomic, atomic.getPtr(),
                                                offsetInfo)) {
                  return success();
                }
                OpBuilder b{atomic};
                auto loc = atomic.getLoc();

                auto materializeAtomicPtr = [&]() {
                  auto materializedAddPtr = materializeAddPtrFromOffset(
                      b, loc, offsetInfo.ptr, offsetInfo.ptrType,
                      offsetInfo.offset);
                  atomic.getPtrMutable().set(materializedAddPtr);
                };

                if (isa<ShapedType>(atomic.getType())) {
                  materializeAtomicPtr();
                  return success();
                }

                if (!isa<triton::PointerType>(offsetInfo.ptr.getType())) {
                  markFallbackAndPreserve(atomic, "atomic_base_ptr_not_scalar");
                  materializeAtomicPtr();
                  return success();
                }

                auto maybeAtomicOffset =
                    castOffsetToSupportedType(offsetInfo.offset, loc, b);
                if (failed(maybeAtomicOffset)) {
                  markFallbackAndPreserve(atomic, "atomic_offset_cast_failed");
                  materializeAtomicPtr();
                  return success();
                }

                auto maybeImportedPtr =
                    buildAtomicImportedAddr(b, loc, offsetInfo.ptr);
                if (failed(maybeImportedPtr)) {
                  markFallbackAndPreserve(atomic, "make_addr_build_failed");
                  materializeAtomicPtr();
                  return success();
                }

                auto ttaAtomic = tta::AtomicCASOp::create(
                    b, loc, *maybeImportedPtr, *maybeAtomicOffset,
                    atomic.getCmp(), atomic.getVal());
                atomic.replaceAllUsesWith(ttaAtomic.getResult());
                atomic->erase();
                return success();
              })
              .Default([&](Operation *op) {
                markFallbackAndPreserve(op, "unexpected_ptr_user_op");
                return success();
              });

      if (failed(res)) {
        continue;
      }
    }

    for (auto op : toDelete) {
      bool shouldPreserve = false;
      for (Value result : op->getResults()) {
        if (preservedPtrValues.contains(result)) {
          shouldPreserve = true;
          break;
        }
      }
      if (shouldPreserve) {
        continue;
      }

      PtrOffset ptrInfo;
      if (!tryGetOffsetInfo(op->getResult(0), offsetMap, ptrInfo)) {
        continue;
      }
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
