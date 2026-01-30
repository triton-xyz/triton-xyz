#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;

#define DEBUG_TYPE "triton-unstructured-fallback"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONUNSTRUCTUREDFALLBACK
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isTensorOfPointers(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return false;
  }
  return isa<triton::PointerType>(tensorType.getElementType());
}

static Value getDimValue(OpBuilder &b, Location loc, Value tensor, int64_t dim,
                         int64_t dimSize) {
  if (!ShapedType::isDynamic(dimSize)) {
    return arith::ConstantOp::create(b, loc, b.getIndexAttr(dimSize));
  }
  return tensor::DimOp::create(b, loc, tensor, dim);
}

static Value extractElement(OpBuilder &b, Location loc, Value value,
                            ArrayRef<Value> indices) {
  if (!value) {
    return Value();
  }
  if (isa<RankedTensorType>(value.getType())) {
    return tensor::ExtractOp::create(b, loc, value, indices);
  }
  return value;
}

static Value createScalarLoad(OpBuilder &b, Location loc, triton::LoadOp op,
                              Value ptr, Value mask, Value other) {
  if (mask && other) {
    return triton::LoadOp::create(b, loc, ptr, mask, other, op.getCache(),
                                  op.getEvict(), op.getIsVolatile());
  }
  if (mask) {
    return triton::LoadOp::create(b, loc, ptr, mask, op.getCache(),
                                  op.getEvict(), op.getIsVolatile());
  }
  return triton::LoadOp::create(b, loc, ptr, op.getCache(), op.getEvict(),
                                op.getIsVolatile());
}

static void createScalarStore(OpBuilder &b, Location loc, triton::StoreOp op,
                              Value ptr, Value value, Value mask) {
  if (mask) {
    triton::StoreOp::create(b, loc, ptr, value, mask, op.getCache(),
                            op.getEvict());
    return;
  }
  triton::StoreOp::create(b, loc, ptr, value, op.getCache(), op.getEvict());
}

class ScalarizeTensorLoad : public OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!ptrType || !isTensorOfPointers(ptrType)) {
      return failure();
    }
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return failure();
    }

    auto loc = op.getLoc();
    Value ptrTensor = op.getPtr();
    auto shape = resultType.getShape();
    auto elementType = resultType.getElementType();
    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value init = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    SmallVector<Value> indices;

    auto buildLoop = [&](auto &self, int dim, Value iterTensor) -> Value {
      if (dim == static_cast<int>(shape.size())) {
        Value scalarPtr =
            tensor::ExtractOp::create(rewriter, loc, ptrTensor, indices);
        Value mask = extractElement(rewriter, loc, op.getMask(), indices);
        Value other = extractElement(rewriter, loc, op.getOther(), indices);
        Value scalar =
            createScalarLoad(rewriter, loc, op, scalarPtr, mask, other);
        return tensor::InsertOp::create(rewriter, loc, scalar, iterTensor,
                                        indices);
      }

      Value upper = getDimValue(rewriter, loc, ptrTensor, dim, shape[dim]);
      auto forOp = scf::ForOp::create(rewriter, loc, zero, upper, one,
                                      ValueRange{iterTensor});
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      indices.push_back(forOp.getInductionVar());
      Value next = self(self, dim + 1, forOp.getRegionIterArg(0));
      scf::YieldOp::create(rewriter, loc, next);
      indices.pop_back();
      return forOp.getResult(0);
    };

    Value result = buildLoop(buildLoop, 0, init);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ScalarizeTensorStore : public OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!ptrType || !isTensorOfPointers(ptrType)) {
      return failure();
    }

    auto loc = op.getLoc();
    Value ptrTensor = op.getPtr();
    auto shape = ptrType.getShape();
    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
    SmallVector<Value> indices;

    auto buildLoop = [&](auto &self, int dim) -> void {
      if (dim == static_cast<int>(shape.size())) {
        Value scalarPtr =
            tensor::ExtractOp::create(rewriter, loc, ptrTensor, indices);
        Value scalarVal = extractElement(rewriter, loc, op.getValue(), indices);
        Value mask = extractElement(rewriter, loc, op.getMask(), indices);
        createScalarStore(rewriter, loc, op, scalarPtr, scalarVal, mask);
        return;
      }

      Value upper = getDimValue(rewriter, loc, ptrTensor, dim, shape[dim]);
      auto forOp = scf::ForOp::create(rewriter, loc, zero, upper, one);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      indices.push_back(forOp.getInductionVar());
      self(self, dim + 1);
      indices.pop_back();
    };

    buildLoop(buildLoop, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

class ScalarizeTensorAtomicRMW : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!ptrType || !isTensorOfPointers(ptrType)) {
      return failure();
    }
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return failure();
    }

    auto loc = op.getLoc();
    Value ptrTensor = op.getPtr();
    auto shape = resultType.getShape();
    auto elementType = resultType.getElementType();
    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value init = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    SmallVector<Value> indices;

    auto buildLoop = [&](auto &self, int dim, Value iterTensor) -> Value {
      if (dim == static_cast<int>(shape.size())) {
        Value scalarPtr =
            tensor::ExtractOp::create(rewriter, loc, ptrTensor, indices);
        Value scalarVal = extractElement(rewriter, loc, op.getVal(), indices);
        Value mask = extractElement(rewriter, loc, op.getMask(), indices);
        Value scalar = triton::AtomicRMWOp::create(
            rewriter, loc, elementType, op.getAtomicRmwOpAttr(), scalarPtr,
            scalarVal, mask, op.getSemAttr(), op.getScopeAttr());
        return tensor::InsertOp::create(rewriter, loc, scalar, iterTensor,
                                        indices);
      }

      Value upper = getDimValue(rewriter, loc, ptrTensor, dim, shape[dim]);
      auto forOp = scf::ForOp::create(rewriter, loc, zero, upper, one,
                                      ValueRange{iterTensor});
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      indices.push_back(forOp.getInductionVar());
      Value next = self(self, dim + 1, forOp.getRegionIterArg(0));
      scf::YieldOp::create(rewriter, loc, next);
      indices.pop_back();
      return forOp.getResult(0);
    };

    Value result = buildLoop(buildLoop, 0, init);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ScalarizeTensorAtomicCAS : public OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!ptrType || !isTensorOfPointers(ptrType)) {
      return failure();
    }
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return failure();
    }

    auto loc = op.getLoc();
    Value ptrTensor = op.getPtr();
    auto shape = resultType.getShape();
    auto elementType = resultType.getElementType();
    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value init = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    SmallVector<Value> indices;

    auto buildLoop = [&](auto &self, int dim, Value iterTensor) -> Value {
      if (dim == static_cast<int>(shape.size())) {
        Value scalarPtr =
            tensor::ExtractOp::create(rewriter, loc, ptrTensor, indices);
        Value scalarCmp = extractElement(rewriter, loc, op.getCmp(), indices);
        Value scalarVal = extractElement(rewriter, loc, op.getVal(), indices);
        Value scalar = triton::AtomicCASOp::create(
            rewriter, loc, elementType, scalarPtr, scalarCmp, scalarVal,
            op.getSemAttr(), op.getScopeAttr());
        return tensor::InsertOp::create(rewriter, loc, scalar, iterTensor,
                                        indices);
      }

      Value upper = getDimValue(rewriter, loc, ptrTensor, dim, shape[dim]);
      auto forOp = scf::ForOp::create(rewriter, loc, zero, upper, one,
                                      ValueRange{iterTensor});
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      indices.push_back(forOp.getInductionVar());
      Value next = self(self, dim + 1, forOp.getRegionIterArg(0));
      scf::YieldOp::create(rewriter, loc, next);
      indices.pop_back();
      return forOp.getResult(0);
    };

    Value result = buildLoop(buildLoop, 0, init);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class TritonUnstructuredFallbackPass
    : public triton::impl::TritonUnstructuredFallbackBase<
          TritonUnstructuredFallbackPass> {
  using Base = triton::impl::TritonUnstructuredFallbackBase<
      TritonUnstructuredFallbackPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ScalarizeTensorLoad, ScalarizeTensorStore,
                 ScalarizeTensorAtomicRMW, ScalarizeTensorAtomicCAS>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
