#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <optional>

#define DEBUG_TYPE "triton-ptr-to-memref"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONPTRTOMEMREF
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct MemrefAccessFromTensorPtr {
  Value memref;
  ValueRange indices;
};

struct ScalarMemrefAccess {
  Value memref;
  Value index;
};

static bool isOneToOneUnrealizedCast(UnrealizedConversionCastOp op) {
  return op && op.getInputs().size() == 1 && op->getNumResults() == 1;
}

static FailureOr<Value> materializeValueAsType(Value value, Type targetType,
                                               PatternRewriter &rewriter,
                                               Location loc) {
  if (!value || !targetType) {
    return failure();
  }

  if (value.getType() == targetType) {
    return value;
  }

  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (isOneToOneUnrealizedCast(castOp) &&
        castOp.getInputs().front().getType() == targetType) {
      return castOp.getInputs().front();
    }
  }

  if (!isa<triton::PointerType>(value.getType()) ||
      !isa<BaseMemRefType>(targetType)) {
    return failure();
  }

  return UnrealizedConversionCastOp::create(rewriter, loc, targetType,
                                            ValueRange{value})
      .getResult(0);
}

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

static bool isElementwiseIdentityGeneric(linalg::GenericOp generic) {
  auto rank = generic.getNumLoops();
  for (auto map : generic.getIndexingMapsArray()) {
    if (!map.isIdentity() || map.getNumDims() != rank) {
      return false;
    }
  }
  for (auto iteratorType : generic.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return false;
    }
  }
  return true;
}

static Value castToIndex(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isIndex()) {
    return value;
  }
  if (isa<IntegerType>(value.getType())) {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), value);
  }
  return Value();
}

static std::optional<ScalarMemrefAccess>
getScalarMemrefAccessFromPtrTensor(Value ptrTensor, ValueRange indices,
                                   PatternRewriter &rewriter) {
  Value totalOffset;
  Value currentTensor = ptrTensor;
  Location loc = ptrTensor.getLoc();

  while (true) {
    if (auto fillOp = currentTensor.getDefiningOp<linalg::FillOp>()) {
      Value basePtr = fillOp.getInputs().front();
      auto castOp = basePtr.getDefiningOp<UnrealizedConversionCastOp>();
      if (!castOp || castOp.getInputs().size() != 1) {
        return std::nullopt;
      }
      Value memref = castOp.getInputs().front();
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      auto unrankedType = dyn_cast<UnrankedMemRefType>(memref.getType());
      if (!memrefType && !unrankedType) {
        return std::nullopt;
      }

      auto elementType = memrefType ? memrefType.getElementType()
                                    : unrankedType.getElementType();
      auto memorySpace = memrefType ? memrefType.getMemorySpace()
                                    : unrankedType.getMemorySpace();
      auto rank1Type = MemRefType::get({ShapedType::kDynamic}, elementType,
                                       AffineMap(), memorySpace);
      Value rank1Memref = memref;
      if (!memrefType || memrefType != rank1Type) {
        rank1Memref = memref::CastOp::create(rewriter, loc, rank1Type, memref);
      }

      if (!totalOffset) {
        totalOffset =
            arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0))
                .getResult();
      }
      return ScalarMemrefAccess{rank1Memref, totalOffset};
    }

    auto generic = currentTensor.getDefiningOp<linalg::GenericOp>();
    if (!generic || !isElementwiseIdentityGeneric(generic)) {
      return std::nullopt;
    }

    Block &body = generic.getRegion().front();
    Operation *payload = nullptr;
    for (auto &op : body.without_terminator()) {
      if (payload) {
        return std::nullopt;
      }
      payload = &op;
    }
    if (!payload) {
      return std::nullopt;
    }

    auto addPtr = dyn_cast<triton::AddPtrOp>(payload);
    if (!addPtr) {
      return std::nullopt;
    }

    auto ptrArg = dyn_cast<BlockArgument>(addPtr.getPtr());
    auto offArg = dyn_cast<BlockArgument>(addPtr.getOffset());
    if (!ptrArg || !offArg) {
      return std::nullopt;
    }

    auto numInputs = generic.getInputs().size();
    if (ptrArg.getArgNumber() >= numInputs ||
        offArg.getArgNumber() >= numInputs) {
      return std::nullopt;
    }

    Value nextTensor = generic.getInputs()[ptrArg.getArgNumber()];
    Value offsetTensor = generic.getInputs()[offArg.getArgNumber()];
    Value offsetElem =
        tensor::ExtractOp::create(rewriter, loc, offsetTensor, indices);
    Value offsetIndex = castToIndex(rewriter, loc, offsetElem);
    if (!offsetIndex) {
      return std::nullopt;
    }

    if (totalOffset) {
      totalOffset =
          arith::AddIOp::create(rewriter, loc, totalOffset, offsetIndex);
    } else {
      totalOffset = offsetIndex;
    }
    currentTensor = nextTensor;
  }
}

static std::optional<ScalarMemrefAccess>
getScalarMemrefAccessFromPtr(Value ptr, PatternRewriter &rewriter) {
  auto extractOp = ptr.getDefiningOp<tensor::ExtractOp>();
  if (!extractOp) {
    return std::nullopt;
  }
  auto tensor = extractOp.getTensor();
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType || !isa<triton::PointerType>(tensorType.getElementType())) {
    return std::nullopt;
  }
  return getScalarMemrefAccessFromPtrTensor(tensor, extractOp.getIndices(),
                                            rewriter);
}

static std::optional<Value> buildAtomicRMWUpdate(PatternRewriter &rewriter,
                                                 Location loc,
                                                 triton::RMWOp rmwOp,
                                                 Value current, Value value) {
  auto elemType = current.getType();
  if (isa<FloatType>(elemType)) {
    switch (rmwOp) {
    case triton::RMWOp::FADD:
      return arith::AddFOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::MAX:
      return arith::MaximumFOp::create(rewriter, loc, current, value)
          .getResult();
    case triton::RMWOp::MIN:
      return arith::MinimumFOp::create(rewriter, loc, current, value)
          .getResult();
    case triton::RMWOp::XCHG:
      return value;
    default:
      return std::nullopt;
    }
  }

  if (isa<IntegerType>(elemType)) {
    switch (rmwOp) {
    case triton::RMWOp::ADD:
      return arith::AddIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::AND:
      return arith::AndIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::OR:
      return arith::OrIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::XOR:
      return arith::XOrIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::MAX:
      return arith::MaxSIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::MIN:
      return arith::MinSIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::UMAX:
      return arith::MaxUIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::UMIN:
      return arith::MinUIOp::create(rewriter, loc, current, value).getResult();
    case triton::RMWOp::XCHG:
      return value;
    default:
      return std::nullopt;
    }
  }

  return std::nullopt;
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

struct FoldPtrSelectToMemrefSelect : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<triton::PointerType>(op.getType())) {
      return failure();
    }

    if (!op.getCondition().getType().isInteger(1)) {
      return failure();
    }

    SmallVector<UnrealizedConversionCastOp> castUsers;
    Type targetType;
    for (OpOperand &use : op.getResult().getUses()) {
      auto castOp = dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
      if (!castOp || !isOneToOneUnrealizedCast(castOp) ||
          castOp.getInputs().front() != op.getResult()) {
        return failure();
      }

      Type castResultType = castOp.getResult(0).getType();
      if (!isa<BaseMemRefType>(castResultType)) {
        return failure();
      }

      if (!targetType) {
        targetType = castResultType;
      } else if (targetType != castResultType) {
        return failure();
      }

      castUsers.push_back(castOp);
    }

    if (castUsers.empty() || !targetType) {
      return failure();
    }

    auto maybeTrueValue = materializeValueAsType(op.getTrueValue(), targetType,
                                                 rewriter, op.getLoc());
    if (failed(maybeTrueValue)) {
      return failure();
    }

    auto maybeFalseValue = materializeValueAsType(
        op.getFalseValue(), targetType, rewriter, op.getLoc());
    if (failed(maybeFalseValue)) {
      return failure();
    }

    Value memrefSelect =
        arith::SelectOp::create(rewriter, op.getLoc(), op.getCondition(),
                                *maybeTrueValue, *maybeFalseValue);

    for (UnrealizedConversionCastOp castUser : castUsers) {
      castUser.getResult(0).replaceAllUsesWith(memrefSelect);
      rewriter.eraseOp(castUser);
    }

    if (op->use_empty()) {
      rewriter.eraseOp(op);
    }

    return success();
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

struct TensorPtrAtomicRMWToMemref
    : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }
    if (op.getMask() && isa<ShapedType>(op.getMask().getType())) {
      return failure();
    }

    auto memrefAccess = getScalarMemrefAccessFromPtr(op.getPtr(), rewriter);
    if (!memrefAccess) {
      return failure();
    }

    auto loc = op.getLoc();
    auto generic = memref::GenericAtomicRMWOp::create(
        rewriter, loc, memrefAccess->memref, memrefAccess->index);
    Block &body = generic.getRegion().front();
    rewriter.setInsertionPointToStart(&body);

    Value current = body.getArgument(0);
    auto updated = buildAtomicRMWUpdate(rewriter, loc, op.getAtomicRmwOp(),
                                        current, op.getVal());
    if (!updated) {
      rewriter.eraseOp(generic);
      return failure();
    }

    Value finalValue = *updated;
    if (auto mask = op.getMask()) {
      finalValue =
          arith::SelectOp::create(rewriter, loc, mask, finalValue, current)
              .getResult();
    }

    memref::AtomicYieldOp::create(rewriter, loc, finalValue);
    rewriter.replaceOp(op, generic.getResult());
    return success();
  }
};

struct TensorPtrAtomicCASToMemref
    : public OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }

    auto memrefAccess = getScalarMemrefAccessFromPtr(op.getPtr(), rewriter);
    if (!memrefAccess) {
      return failure();
    }

    auto loc = op.getLoc();
    auto generic = memref::GenericAtomicRMWOp::create(
        rewriter, loc, memrefAccess->memref, memrefAccess->index);
    Block &body = generic.getRegion().front();
    rewriter.setInsertionPointToStart(&body);

    Value current = body.getArgument(0);
    Value cmp = op.getCmp();
    Value desired = op.getVal();

    Value equal;
    if (isa<FloatType>(current.getType())) {
      equal = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OEQ,
                                    current, cmp)
                  .getResult();
    } else if (isa<IntegerType>(current.getType())) {
      equal = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    current, cmp)
                  .getResult();
    } else {
      rewriter.eraseOp(generic);
      return failure();
    }

    Value finalValue =
        arith::SelectOp::create(rewriter, loc, equal, desired, current)
            .getResult();
    memref::AtomicYieldOp::create(rewriter, loc, finalValue);
    rewriter.replaceOp(op, generic.getResult());
    return success();
  }
};

class TritonPtrToMemrefPass
    : public triton::impl::TritonPtrToMemrefBase<TritonPtrToMemrefPass> {
  using Base = triton::impl::TritonPtrToMemrefBase<TritonPtrToMemrefPass>;
  using Base::Base;

public:
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
    postPatterns.add<FoldPtrSelectToMemrefSelect, TensorPtrLoadToMemref,
                     TensorPtrStoreToMemref, TensorPtrAtomicRMWToMemref,
                     TensorPtrAtomicCASToMemref>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(postPatterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
