#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h" // IWYU pragma: keep
#include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::triton {
#define GEN_PASS_DEF_VERIFYTTALOWERED
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

static bool typeContainsTTAAddr(Type type) {
  if (!type) {
    return false;
  }

  if (isa<tta::AddrType>(type)) {
    return true;
  }

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return typeContainsTTAAddr(shapedType.getElementType());
  }

  if (auto tupleType = dyn_cast<TupleType>(type)) {
    return llvm::any_of(tupleType.getTypes(), typeContainsTTAAddr);
  }

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(), typeContainsTTAAddr) ||
           llvm::any_of(functionType.getResults(), typeContainsTTAAddr);
  }

  return false;
}

class VerifyTTALoweredPass
    : public mlir::triton::impl::VerifyTTALoweredBase<VerifyTTALoweredPass> {
  using Base = mlir::triton::impl::VerifyTTALoweredBase<VerifyTTALoweredPass>;
  using Base::Base;

public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool hasError = false;

    moduleOp.walk([&](Operation *op) {
      if (op->getDialect() &&
          isa<tta::TritonAddressDialect>(op->getDialect())) {
        op->emitOpError("must be eliminated before backend-ready stage");
        hasError = true;
        return;
      }

      auto reportAddrTypeLeak = [&](Type type, StringRef where) {
        if (!typeContainsTTAAddr(type)) {
          return;
        }
        op->emitOpError() << where << " contains !tta.addr after tta lowering";
        hasError = true;
      };

      for (Type operandType : op->getOperandTypes()) {
        reportAddrTypeLeak(operandType, "operand type");
      }
      for (Type resultType : op->getResultTypes()) {
        reportAddrTypeLeak(resultType, "result type");
      }

      if (auto functionOp = dyn_cast<FunctionOpInterface>(op)) {
        if (typeContainsTTAAddr(functionOp.getFunctionType())) {
          op->emitOpError(
              "function signature contains !tta.addr after tta lowering");
          hasError = true;
        }
      }
    });

    if (hasError) {
      signalPassFailure();
    }
  }
};

} // namespace
