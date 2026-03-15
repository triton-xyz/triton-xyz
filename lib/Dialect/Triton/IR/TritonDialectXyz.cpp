#include "triton-shared/Dialect/Triton/IR/TritonDialectXyz.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace mlir::triton;

#define GET_OP_CLASSES
#include "triton-shared/Dialect/Triton/IR/TritonXyzOps.cpp.inc"

namespace mlir::triton::xyz {

void registerTritonDialectXyzExtension(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, triton::TritonDialect *dialect) {
    if (!RegisteredOperationName::lookup(NopOp::getOperationName(), ctx))
      RegisteredOperationName::insert<NopOp>(*dialect);
  });
}

} // namespace mlir::triton::xyz
