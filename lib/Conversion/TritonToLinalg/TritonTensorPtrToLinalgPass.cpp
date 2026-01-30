#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONTENSORPTRTOLINALG
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

class AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op.getResult().getType();
    // This pattern only handles tensor of pointers, not scalar pointers
    if (!isa<RankedTensorType>(resType)) {
      return failure();
    }

    auto rank = cast<RankedTensorType>(resType).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        /*numResult + numOperands*/ 3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {op.getPtr()};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton op X operating on tensors of pointers to a linalg.generic
// wrapping op X to operate on single pointer.
// This pattern rewriter is almost identical to AddPtrConverter above, except
// that the out param for the linalg op is an empty op instead of reusing one
// of the existing operands. This is because depending on the templatized op,
// the type of the operands might be different, so we cannot pick a default
// operand to reuse for all cases.
template <typename OpType>
class TensorOpConverter : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTensorType =
        dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType) {
      return failure();
    }
    auto rank = resultTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {tensor::EmptyOp::create(
        rewriter, op->getLoc(), resultTensorType.getShape(),
        resultTensorType.getElementType())};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton store op operating on tensors of pointers to a linalg.generic
// wrapping op a triton store op on single pointer.
// Note that this linalg.generic op has an empty `out` param.
class StorePtrToLinalgConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeTensorType = dyn_cast<RankedTensorType>(op.getValue().getType());
    if (!storeTensorType) {
      return failure();
    }
    auto rank = storeTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op.getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs;
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

class TritonTensorPtrToLinalgPass
    : public triton::impl::TritonTensorPtrToLinalgBase<
          TritonTensorPtrToLinalgPass> {
  using Base =
      triton::impl::TritonTensorPtrToLinalgBase<TritonTensorPtrToLinalgPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, triton::TritonDialect>();

    // Mark triton ops on tensor of pointers as illegal
    target.addDynamicallyLegalOp<triton::LoadOp, triton::StoreOp,
                                 triton::IntToPtrOp, triton::PtrToIntOp>(
        [](Operation *op) {
          if (auto load = dyn_cast<triton::LoadOp>(op)) {
            return !isa<RankedTensorType>(load.getResult().getType());
          }
          if (auto store = dyn_cast<triton::StoreOp>(op)) {
            return !isa<RankedTensorType>(store.getValue().getType());
          }
          // IntToPtr, PtrToInt
          return !isa<RankedTensorType>(op->getResult(0).getType());
        });

    target.addDynamicallyLegalOp<triton::BitcastOp>([](triton::BitcastOp op) {
      if (isa<RankedTensorType>(op.getType()) &&
          triton::isPtrTypeLike(op.getType())) {
        return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
      return !isa<RankedTensorType>(op.getResult().getType());
    });

    // Patterns for converting tensor pointer operations to linalg.generic
    patterns.add<StorePtrToLinalgConverter, TensorOpConverter<triton::LoadOp>,
                 TensorOpConverter<triton::IntToPtrOp>,
                 TensorOpConverter<triton::PtrToIntOp>,
                 TensorOpConverter<triton::BitcastOp>, AddPtrConverter>(
        patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
