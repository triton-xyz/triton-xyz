#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONSCANTOSCF
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

/// Lower tt.scan on a 1D tensor to memref-based scf.for loop computing prefix
/// sum. Produces memref-based output to avoid downstream bufferization issues.
struct ScanOpLowering : public OpRewritePattern<triton::ScanOp> {
  using OpRewritePattern<triton::ScanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ScanOp scanOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = scanOp.getResultTypes()[0];
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType || tensorType.getRank() != 1)
      return failure();

    auto loc = scanOp.getLoc();
    int64_t axis = scanOp.getAxis();
    bool reverse = scanOp.getReverse();
    if (axis != 0 || reverse)
      return failure();

    Value input = scanOp.getOperands()[0];
    int64_t dimSize = tensorType.getDimSize(0);
    if (dimSize == ShapedType::kDynamic)
      return failure();

    auto elemType = tensorType.getElementType();

    // Allocate result memref.
    auto memrefType = MemRefType::get({dimSize}, elemType);
    Value alloc =
        memref::AllocOp::create(rewriter, loc, memrefType).getMemref();

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1).getResult();
    Value dim =
        arith::ConstantIndexOp::create(rewriter, loc, dimSize).getResult();
    Value zeroElem = arith::ConstantOp::create(rewriter, loc, elemType,
                                               rewriter.getZeroAttr(elemType))
                         .getResult();

    // Build scf.for loop that reads from the input tensor/array
    // and writes prefix sum into our allocated memref.
    auto loop = scf::ForOp::create(
        rewriter, loc, c0, dim, c1, ValueRange{zeroElem},
        [&](OpBuilder &b, Location loc, Value i, ValueRange iterArgs) {
          Value running = iterArgs[0];

          // Read input element.  Input might be a tensor (from linalg path)
          // or already a memref (from TTA path). Handle both.
          Value val;
          if (isa<MemRefType>(input.getType())) {
            val = memref::LoadOp::create(b, loc, input, ValueRange{i})
                      .getResult();
          } else {
            val = tensor::ExtractOp::create(b, loc, input, ValueRange{i})
                      .getResult();
          }

          // Add to running sum.
          Value sum;
          if (isa<FloatType>(elemType)) {
            sum = arith::AddFOp::create(b, loc, running, val).getResult();
          } else {
            sum = arith::AddIOp::create(b, loc, running, val).getResult();
          }

          // Store into result memref.
          memref::StoreOp::create(b, loc, sum, alloc, ValueRange{i});
          scf::YieldOp::create(b, loc, ValueRange{sum});
        });

    // Convert result memref back to tensor for downstream consumers.
    Value resultTensor =
        bufferization::ToTensorOp::create(rewriter, loc, tensorType, alloc,
                                          /*restrict=*/true,
                                          /*writable=*/true)
            .getResult();
    rewriter.replaceOp(scanOp, resultTensor);
    return success();
  }
};

class TritonScanToSCFPass
    : public triton::impl::TritonScanToSCFBase<TritonScanToSCFPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ScanOpLowering>(&getContext());

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
