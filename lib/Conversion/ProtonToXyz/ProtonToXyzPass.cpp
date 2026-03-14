#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "proton/Dialect/include/Analysis/ScopeIdAllocation.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton-shared/Conversion/ProtonToXyz/Passes.h" // IWYU pragma: keep

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PROTONTOXYZ
#include "triton-shared/Conversion/ProtonToXyz/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

constexpr StringLiteral kCpuRecordStartFn = "proton_cpu_record_start";
constexpr StringLiteral kCpuRecordEndFn = "proton_cpu_record_end";

static func::FuncOp getOrCreateRuntimeHook(ModuleOp moduleOp, StringRef name) {
  if (auto hook = moduleOp.lookupSymbol<func::FuncOp>(name)) {
    return hook;
  }

  OpBuilder builder(moduleOp.getBodyRegion());
  builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  auto hookType =
      builder.getFunctionType(TypeRange{builder.getI64Type()}, TypeRange{});
  auto hook = func::FuncOp::create(builder, moduleOp.getLoc(), name, hookType);
  hook.setPrivate();
  return hook;
}

class ProtonToXyzPass
    : public mlir::triton::impl::ProtonToXyzBase<ProtonToXyzPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<mlir::triton::proton::RecordOp> recordOps;
    moduleOp.walk([&](mlir::triton::proton::RecordOp recordOp) {
      recordOps.push_back(recordOp);
    });
    if (recordOps.empty()) {
      return;
    }

    mlir::triton::proton::ModuleScopeIdAllocation scopeIds(moduleOp);
    auto startHook = getOrCreateRuntimeHook(moduleOp, kCpuRecordStartFn);
    auto endHook = getOrCreateRuntimeHook(moduleOp, kCpuRecordEndFn);

    for (auto recordOp : recordOps) {
      OpBuilder builder(recordOp);
      auto scopeId =
          static_cast<int64_t>(scopeIds.getOpScopeId(recordOp.getOperation()));
      auto scopeIdValue =
          arith::ConstantIntOp::create(builder, recordOp.getLoc(), scopeId, 64);
      auto target = recordOp.getIsStart() ? startHook : endHook;
      func::CallOp::create(builder, recordOp.getLoc(), target.getName(),
                           TypeRange{}, ValueRange{scopeIdValue});
      recordOp.erase();
    }
  }
};

} // namespace
