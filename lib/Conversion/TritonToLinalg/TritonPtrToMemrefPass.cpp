#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h" // IWYU pragma: keep
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <optional>

#define DEBUG_TYPE "triton-ptr-to-memref"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONPTRTOMEMREF
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct MemrefAccessFromTensorPtr {
  Value memref;
  ValueRange indices;
};

static std::optional<MemrefAccessFromTensorPtr>
getMemrefAccessFromTensorPtr(Value ptr) {
  auto extractOp = ptr.getDefiningOp<tensor::ExtractOp>();
  if (!extractOp) {
    return std::nullopt;
  }

  auto tensor = extractOp.getTensor();
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  auto ptrElemType = dyn_cast<triton::PointerType>(tensorType.getElementType());
  if (!ptrElemType) {
    return std::nullopt;
  }

  auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>();
  if (!castOp || castOp.getInputs().size() != 1) {
    return std::nullopt;
  }

  Value memref = castOp.getInputs().front();
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType) {
    return std::nullopt;
  }

  if (memrefType.getElementType() != ptrElemType.getPointeeType()) {
    return std::nullopt;
  }

  return MemrefAccessFromTensorPtr{memref, extractOp.getIndices()};
}

class TritonFunctionSignatureConverter : public TypeConverter {
public:
  TritonFunctionSignatureConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(),
                                     /*memorySpace=*/0);
    });
    addConversion([](RankedTensorType tensorType) -> std::optional<Type> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return MemRefType::get(tensorType.getShape(), ptrType.getPointeeType());
      }
      return std::nullopt;
    });

    auto createUnrealizedCast = [&](OpBuilder &builder, Type resultType,
                                    ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
  }
};

struct TensorPtrLoadToMemref : public OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    if (op.getMask() && isa<ShapedType>(op.getMask().getType())) {
      return failure();
    }

    auto memrefAccess = getMemrefAccessFromTensorPtr(op.getPtr());
    if (!memrefAccess) {
      return failure();
    }

    auto memref = memrefAccess->memref;
    auto indices = memrefAccess->indices;

    if (op.getMask()) {
      auto ifOp = scf::IfOp::create(
          rewriter, op->getLoc(), op.getMask(),
          [&](OpBuilder &b, Location loc) {
            Value val =
                memref::LoadOp::create(b, loc, memref, indices).getResult();
            scf::YieldOp::create(b, loc, val);
          },
          [&](OpBuilder &b, Location loc) {
            if (op.getOther()) {
              scf::YieldOp::create(b, loc, op.getOther());
            } else {
              auto elemType = op.getType();
              auto zeroAttr = b.getZeroAttr(elemType);
              assert(zeroAttr && "unexpected element type");
              Value val =
                  arith::ConstantOp::create(b, loc, zeroAttr).getResult();
              scf::YieldOp::create(b, loc, val);
            }
          });
      rewriter.replaceOp(op, ifOp);
    } else {
      auto val = memref::LoadOp::create(rewriter, op.getLoc(), memref, indices)
                     .getResult();
      rewriter.replaceOp(op, val);
    }

    return success();
  }
};

struct TensorPtrStoreToMemref : public OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getValue().getType())) {
      return failure();
    }
    if (op.getMask() && isa<ShapedType>(op.getMask().getType())) {
      return failure();
    }

    auto memrefAccess = getMemrefAccessFromTensorPtr(op.getPtr());
    if (!memrefAccess) {
      return failure();
    }

    auto memref = memrefAccess->memref;
    auto indices = memrefAccess->indices;

    IRRewriter::InsertionGuard g(rewriter);
    if (op.getMask()) {
      auto ifOp = scf::IfOp::create(rewriter, op->getLoc(), op.getMask(),
                                    /*withElseRegion*/ false);
      rewriter.setInsertionPointToStart(
          &ifOp.getThenRegion().getBlocks().front());
    }

    memref::StoreOp::create(rewriter, op->getLoc(), op.getValue(), memref,
                            indices);

    rewriter.eraseOp(op);
    return success();
  }
};

class TritonPtrToMemrefPass
    : public triton::impl::TritonPtrToMemrefBase<TritonPtrToMemrefPass> {
  using Base = triton::impl::TritonPtrToMemrefBase<TritonPtrToMemrefPass>;
  using Base::Base;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonFunctionSignatureConverter typeConverter;

    // Update function signature and call ops to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp, triton::FuncOp>([&](auto op) {
      return typeConverter.isSignatureLegal(
          cast<FunctionType>(cast<FunctionOpInterface>(op).getFunctionType()));
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getResultTypes()) &&
             typeConverter.isLegal(op.getOperandTypes());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet postPatterns(&getContext());
    postPatterns.add<TensorPtrLoadToMemref, TensorPtrStoreToMemref>(
        &getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(postPatterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
