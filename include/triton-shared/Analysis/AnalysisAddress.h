#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/PtrExprAnalysis.h"

#include <optional>

namespace mlir {
namespace triton {
namespace address {

struct IndirectIndexRule {
  Value indexTensor;
  Value maskTensor;
};

struct WrapBoundary {
  OpFoldResult boundary;
};

struct DimRule {
  OpFoldResult size;
  OpFoldResult stride;
  OpFoldResult offset;
  std::optional<WrapBoundary> wrapBoundary;
  std::optional<IndirectIndexRule> indirect;

  bool isStructured() const {
    return !wrapBoundary.has_value() && !indirect.has_value();
  }
};

enum class LayoutKind {
  Strided,
  Block,
};

struct BlockLayout {
  SmallVector<OpFoldResult> parentShape;
  SmallVector<int32_t> order;
};

struct AddressDescriptor {
  Value base;
  Type elementType;
  int addressSpace = 0;
  int64_t rank = 0;
  SmallVector<DimRule> dims;
  LayoutKind layoutKind = LayoutKind::Strided;
  std::optional<BlockLayout> blockLayout;
};

enum class AddressClass {
  StructuredPtr,
  SplitPtr,
  BlockPtr,
  MixedPtr,
};

struct AddressFeatures {
  bool hasBlockLayout = false;
  bool hasWrapBoundary = false;
  bool hasIndirect = false;
};

struct AddressAnalysisOptions {
  bool enableRefine = true;
  bool enableValidation = true;
  bool enableDescriptorDebugDump = false;
  bool enableRelaxedSingleIndirectNonGatherDims = true;
};

struct AddressAnalysisResult {
  AddressDescriptor descriptor;
  AddressFeatures features;
  AddressClass addressClass = AddressClass::StructuredPtr;
};

AddressFeatures getAddressFeatures(const AddressDescriptor &descriptor);

AddressClass classifyAddress(const AddressDescriptor &descriptor);

class AnalysisAddress {
public:
  explicit AnalysisAddress(bool enableMakeGatherScatterTensorPtr = true)
      : ptrAnalysis(enableMakeGatherScatterTensorPtr) {}

  FailureOr<AddressAnalysisResult> analyzeDescriptorWithOptions(
      Value ptrLike, Location loc, OpBuilder &builder,
      const AddressAnalysisOptions &options = AddressAnalysisOptions(),
      std::optional<StringRef> *failureReason = nullptr);

  FailureOr<AddressDescriptor>
  analyzeDescriptor(Value ptrLike, Location loc, OpBuilder &builder,
                    std::optional<StringRef> *failureReason = nullptr);

private:
  FailureOr<AddressDescriptor> analyzeSeedDescriptor(
      Value ptrLike, Location loc, OpBuilder &builder,
      const AddressAnalysisOptions &options = AddressAnalysisOptions(),
      std::optional<StringRef> *failureReason = nullptr);

  FailureOr<AddressDescriptor> analyzeFromTTAChain(
      Value ptrLike, Location loc, OpBuilder &builder,
      const AddressAnalysisOptions &options = AddressAnalysisOptions(),
      std::optional<StringRef> *failureReason = nullptr);

  FailureOr<AddressDescriptor> analyzeFromPtrStateSeed(
      Value ptrLike, Location loc, OpBuilder &builder,
      const AddressAnalysisOptions &options = AddressAnalysisOptions(),
      std::optional<StringRef> *failureReason = nullptr);

  LogicalResult refineDescriptor(
      AddressDescriptor &descriptor, Value ptrLike, Location loc,
      OpBuilder &builder,
      const AddressAnalysisOptions &options = AddressAnalysisOptions(),
      std::optional<StringRef> *failureReason = nullptr);

  LogicalResult validateDescriptor(
      const AddressDescriptor &descriptor,
      const AddressAnalysisOptions &options = AddressAnalysisOptions(),
      std::optional<StringRef> *failureReason = nullptr);

  static void dumpDescriptor(const AddressDescriptor &descriptor,
                             StringRef stage);

  ptrexpr::PtrExprAnalysis ptrAnalysis;
};

class TTAEmitter {
public:
  static FailureOr<Value>
  emitAddress(const AddressDescriptor &descriptor, Location loc,
              OpBuilder &builder,
              std::optional<StringRef> *failureReason = nullptr);

  static FailureOr<Value>
  emitMakeAddr(const AddressDescriptor &descriptor, Location loc,
               OpBuilder &builder,
               std::optional<StringRef> *failureReason = nullptr);

  static FailureOr<SmallVector<OpFoldResult>>
  analyzeMaskDims(Value mask, Location loc, OpBuilder &builder,
                  bool useUnsafeMask = false);

  static FailureOr<Value> getScalarOther(Value other, Location loc,
                                         OpBuilder &builder);
};

} // namespace address
} // namespace triton
} // namespace mlir
