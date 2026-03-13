#include "triton-shared/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep

#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace triton {
bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = llvm::TypeSwitch<Operation *, Value>(*op)
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
                  llvm_unreachable("unsupported scalar reconstruction op");
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    }

    if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          return Value();
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            builder, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
      continue;
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
      continue;
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
      continue;
    }

    return Value();
  }
}
} // namespace triton

} // namespace mlir
