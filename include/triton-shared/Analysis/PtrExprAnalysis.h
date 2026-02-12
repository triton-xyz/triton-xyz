#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <functional>

namespace mlir {
namespace triton {
namespace ptrexpr {

struct PtrState {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> shape;
  SmallVector<int32_t> order;

  Value source;
  Value scalar;

  int32_t getRank() const;

  bool isEmpty() const;

  bool hasModulo() const;

  bool dimHasModulo(uint32_t dim) const;

  bool dimIsStructured(uint32_t dim) const;
  int32_t getNonStructuredDim() const;
  bool noStructuredDimExists() const;

  bool isStructured() const;

  bool isBlockPtr() const;

  void dump() const;

  LogicalResult rebuildAsUnsupportedOp(Value op);

  LogicalResult rebuildAsGatherScatter(Value op, int nonContinuousDim);

  LogicalResult addState(const PtrState &lhsState, const PtrState &rhsState,
                         bool isAnalysisingUnstructured, Operation *op,
                         OpBuilder &builder);

  LogicalResult mulState(const PtrState &lhsState, const PtrState &rhsState,
                         bool isAnalysisingUnstructured, Operation *op,
                         OpBuilder &builder);

  LogicalResult mergeUnstructuredState(const PtrState &other, Operation *op);
};

class PtrExprAnalysis {
public:
  using LoopResultResolver =
      std::function<FailureOr<PtrState>(scf::ForOp, Value)>;

  explicit PtrExprAnalysis(bool enableMakeGatherScatterTensorPtr)
      : enableMakeGatherScatterTensorPtr(enableMakeGatherScatterTensorPtr) {}

  void setLoopResultResolver(LoopResultResolver resolver) {
    loopResultResolver = std::move(resolver);
  }

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;

  LogicalResult visitOperand(Value operand, PtrState &state, const Location loc,
                             OpBuilder &builder);

  LogicalResult visitOperandForOp(scf::ForOp forOp, Value operand,
                                  PtrState &state, const Location loc,
                                  OpBuilder &builder);

  LogicalResult visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandRem(arith::RemSIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                      PtrState &state, Location loc,
                                      OpBuilder &builder);

  LogicalResult visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder);

  LogicalResult visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                      PtrState &state, const Location loc,
                                      OpBuilder &builder);

  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult visitOperandConstSplat(arith::ConstantOp op, PtrState &state,
                                       const Location loc, OpBuilder &builder);

  LogicalResult visitOperandExtSI(arith::ExtSIOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state,
                                   const Location loc, OpBuilder &builder);

  LogicalResult visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                          PtrState &state, const Location loc,
                                          OpBuilder &builder);

  LogicalResult visitOperandIntToPtr(triton::IntToPtrOp intToPtrOp,
                                     PtrState &state, const Location loc,
                                     OpBuilder &builder);

  LogicalResult visitOperandBitcast(triton::BitcastOp bitcastOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder);

protected:
  const bool enableMakeGatherScatterTensorPtr;
  bool isAnalysisingUnstructured = false;

private:
  LoopResultResolver loopResultResolver;
};

} // namespace ptrexpr
} // namespace triton
} // namespace mlir
