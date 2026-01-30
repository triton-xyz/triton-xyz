#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_NORMALIZETENSORPTRORDER
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct PermInfo {
  SmallVector<unsigned> reorderMap; // new_dim -> old_dim
  SmallVector<int64_t> permLoad;    // transpose after load
  SmallVector<int64_t> permStore;   // transpose before store
};

static bool isDescendingOrder(ArrayRef<int32_t> order) {
  return llvm::is_sorted(order, std::greater<>());
}

static bool isIdentityPerm(ArrayRef<int64_t> perm) {
  for (auto [i, v] : llvm::enumerate(perm)) {
    if (static_cast<int64_t>(i) != v)
      return false;
  }
  return true;
}

static PermInfo computePermInfo(ArrayRef<int32_t> order) {
  PermInfo info;
  int64_t rank = static_cast<int64_t>(order.size());

  info.reorderMap.resize(rank);
  for (int64_t i = 0; i < rank; ++i) {
    info.reorderMap[i] = static_cast<unsigned>(order[rank - 1 - i]);
  }

  info.permLoad.resize(rank);
  for (int64_t i = 0; i < rank; ++i) {
    info.permLoad[i] = rank - 1 - order[i];
  }

  info.permStore.resize(rank);
  for (int64_t i = 0; i < rank; ++i) {
    info.permStore[i] = order[rank - 1 - i];
  }

  return info;
}

template <typename T>
static SmallVector<T> permuteByMap(ArrayRef<T> input, ArrayRef<unsigned> map) {
  SmallVector<T> result;
  result.reserve(map.size());
  for (auto idx : map) {
    result.push_back(input[idx]);
  }
  return result;
}

static SmallVector<int32_t> getContiguousOrder(int64_t rank) {
  SmallVector<int32_t> order;
  order.reserve(rank);
  for (int64_t i = rank - 1; i >= 0; --i) {
    order.push_back(static_cast<int32_t>(i));
  }
  return order;
}

class OrderNormalizer {
public:
  OrderNormalizer() = default;

  LogicalResult normalize(tts::MakeTensorPtrOp op) {
    auto order = op.getOrder();
    if (order.empty() || isDescendingOrder(order))
      return success();

    auto rank = static_cast<int64_t>(order.size());
    if (rank == 0) {
      return success();
    }
    SmallVector<int32_t> expected(static_cast<size_t>(rank));
    std::iota(expected.begin(), expected.end(), 0);
    if (!std::is_permutation(order.begin(), order.end(), expected.begin(),
                             expected.end())) {
      op->emitOpError("order is not a permutation when normalizing");
      return failure();
    }

    PermInfo info = computePermInfo(order);
    SmallVector<int64_t> newSizes =
        permuteByMap<int64_t>(op.getSizes(), info.reorderMap);
    SmallVector<OpFoldResult> newStrides =
        permuteByMap<OpFoldResult>(op.getMixedStrides(), info.reorderMap);
    SmallVector<OpFoldResult> newOffsets =
        permuteByMap<OpFoldResult>(op.getMixedOffsets(), info.reorderMap);
    SmallVector<OpFoldResult> newShape =
        permuteByMap<OpFoldResult>(op.getMixedShape(), info.reorderMap);

    OpBuilder builder(op);
    auto newOrder = getContiguousOrder(rank);
    auto newOp = tts::MakeTensorPtrOp::create(
        builder, op.getLoc(), op.getBase(), newSizes, newStrides, newOffsets,
        newShape, newOrder);

    if (failed(rewriteUsers(op.getResult(), newOp.getResult(), info))) {
      return failure();
    }

    if (op->use_empty()) {
      op.erase();
    }

    return success();
  }

private:
  LogicalResult rewriteUsers(Value oldPtr, Value newPtr, const PermInfo &info) {
    if (!visited.insert(oldPtr).second)
      return success();

    SmallVector<OpOperand *> uses;
    for (auto &use : oldPtr.getUses())
      uses.push_back(&use);

    for (auto *use : uses) {
      if (failed(rewriteUse(use, newPtr, info)))
        return failure();
    }
    return success();
  }

  LogicalResult rewriteUse(OpOperand *use, Value newPtr, const PermInfo &info) {
    Operation *user = use->getOwner();
    OpBuilder builder(user);
    auto loc = user->getLoc();

    if (auto loadOp = dyn_cast<tts::LoadOp>(user)) {
      auto mixedMask = loadOp.getMixedMaskDims();
      if (!mixedMask.empty() && mixedMask.size() != info.reorderMap.size()) {
        loadOp->emitOpError("mask rank mismatch when normalizing order");
        return failure();
      }
      SmallVector<OpFoldResult> newMask;
      if (!mixedMask.empty())
        newMask = permuteByMap<OpFoldResult>(mixedMask, info.reorderMap);

      auto other = loadOp.getOther();
      auto newLoad = tts::LoadOp::create(builder, loc, newPtr, newMask, other);
      Value replacement = newLoad.getResult();

      if (!isIdentityPerm(info.permLoad)) {
        auto oldType = cast<RankedTensorType>(loadOp.getResult().getType());
        auto init = tensor::EmptyOp::create(builder, loc, oldType.getShape(),
                                            oldType.getElementType());
        auto transpose = linalg::TransposeOp::create(
            builder, loc, replacement, init.getResult(), info.permLoad);
        replacement = transpose.getResults()[0];
      }

      loadOp.replaceAllUsesWith(replacement);
      loadOp.erase();
      return success();
    }

    if (auto storeOp = dyn_cast<tts::StoreOp>(user)) {
      auto mixedMask = storeOp.getMixedMaskDims();
      if (!mixedMask.empty() && mixedMask.size() != info.reorderMap.size()) {
        storeOp->emitOpError("mask rank mismatch when normalizing order");
        return failure();
      }
      SmallVector<OpFoldResult> newMask;
      if (!mixedMask.empty())
        newMask = permuteByMap<OpFoldResult>(mixedMask, info.reorderMap);

      Value valueToStore = storeOp.getValue();
      if (!isIdentityPerm(info.permStore)) {
        auto valType = cast<RankedTensorType>(valueToStore.getType());
        SmallVector<int64_t> newShape;
        newShape.reserve(info.reorderMap.size());
        for (auto idx : info.reorderMap) {
          newShape.push_back(valType.getShape()[idx]);
        }
        auto init = tensor::EmptyOp::create(builder, loc, newShape,
                                            valType.getElementType());
        auto transpose = linalg::TransposeOp::create(
            builder, loc, valueToStore, init.getResult(), info.permStore);
        valueToStore = transpose.getResults()[0];
      }

      tts::StoreOp::create(builder, loc, newPtr, valueToStore, newMask);
      storeOp.erase();
      return success();
    }

    if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
      SmallVector<Value> newOffsets;
      newOffsets.reserve(info.reorderMap.size());
      auto oldOffsets = advanceOp.getOffsets();
      for (auto idx : info.reorderMap) {
        newOffsets.push_back(oldOffsets[idx]);
      }
      auto newAdvance = triton::AdvanceOp::create(
          builder, loc, newPtr.getType(), newPtr, newOffsets);

      if (failed(rewriteUsers(advanceOp.getResult(), newAdvance.getResult(),
                              info))) {
        return failure();
      }
      if (advanceOp->use_empty()) {
        advanceOp.erase();
      }
      return success();
    }

    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
      auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
      auto result = loopOp.getTiedLoopResult(use);

      if (!iterArg || !result) {
        loopOp->emitOpError("unsupported loop operand when normalizing order");
        return failure();
      }

      use->set(newPtr);

      iterArg.setType(newPtr.getType());
      result.setType(newPtr.getType());

      if (failed(rewriteUsers(iterArg, iterArg, info)))
        return failure();
      if (failed(rewriteUsers(result, result, info)))
        return failure();

      return success();
    }

    if (isa<scf::YieldOp>(user)) {
      use->set(newPtr);
      return success();
    }

    user->emitOpError("unsupported user when normalizing tts.make_tptr order");
    return failure();
  }

  DenseSet<Value> visited;
};

class NormalizeTensorPtrOrderPass
    : public triton::impl::NormalizeTensorPtrOrderBase<
          NormalizeTensorPtrOrderPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    triton::TritonDialect, tts::TritonStructuredDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OrderNormalizer normalizer;

    SmallVector<tts::MakeTensorPtrOp> candidates;
    moduleOp.walk([&](tts::MakeTensorPtrOp op) {
      if (op.isBlockPtr() && !isDescendingOrder(op.getOrder())) {
        candidates.push_back(op);
      }
    });

    for (auto op : candidates) {
      if (failed(normalizer.normalize(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
