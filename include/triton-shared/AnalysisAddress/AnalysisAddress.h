#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/PtrExprAnalysis.h"

namespace mlir {
namespace triton {
namespace address {

struct AnalyzedAddress {
  Value base;
  SmallVector<int64_t> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> shape;
  SmallVector<int32_t> order;
};

class AnalysisAddress {
public:
  explicit AnalysisAddress(bool enableMakeGatherScatterTensorPtr = true)
      : ptrAnalysis(enableMakeGatherScatterTensorPtr) {}

  FailureOr<AnalyzedAddress> analyze(Value ptrLike, Location loc,
                                     OpBuilder &builder);

private:
  ptrexpr::PtrExprAnalysis ptrAnalysis;
};

class TTAEmitter {
public:
  static FailureOr<Value> emitMakeAddr(const AnalyzedAddress &address,
                                       Location loc, OpBuilder &builder);

  static FailureOr<SmallVector<OpFoldResult>>
  analyzeMaskDims(Value mask, Location loc, OpBuilder &builder,
                  bool useUnsafeMask = false);

  static FailureOr<Value> getScalarOther(Value other, Location loc,
                                         OpBuilder &builder);
};

} // namespace address
} // namespace triton
} // namespace mlir
