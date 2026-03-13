#include "Profiler/Cpu/CpuProfiler.h"

#include "Data/Metric.h"
#include "Device.h"

namespace proton {

thread_local CpuProfiler::TimePoint CpuProfiler::activeOpStart{};
thread_local DataToEntryMap CpuProfiler::activeOpDataToEntry{};
thread_local bool CpuProfiler::activeOpValid = false;
thread_local std::vector<CpuProfiler::ActiveScopeState>
    CpuProfiler::activeScopeStack{};

uint64_t CpuProfiler::toNs(TimePoint t) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch())
          .count());
}

void CpuProfiler::emitKernelMetric(DataToEntryMap &dataToEntry, TimePoint start,
                                   TimePoint end) {
  const auto startNs = toNs(start);
  const auto endNs = toNs(end);
  const auto safeEndNs = endNs >= startNs ? endNs : startNs;

  for (auto &[data, entry] : dataToEntry) {
    auto metric = std::make_unique<KernelMetric>(
        startNs, safeEndNs, /*invocations=*/1,
        /*deviceId=*/0, static_cast<uint64_t>(DeviceType::CPU),
        /*streamId=*/0);
    entry.upsertMetric(std::move(metric));
  }
}

void CpuProfiler::emitScalarMetrics(
    const DataToEntryMap &dataToEntry,
    const std::map<std::string, MetricValueType> &scalarMetrics) {
  if (scalarMetrics.empty()) {
    return;
  }
  for (const auto &[data, entry] : dataToEntry) {
    (void)data;
    entry.upsertFlexibleMetrics(scalarMetrics);
  }
}

void CpuProfiler::startOp(const Scope &scope) {
  activeOpStart = Clock::now();
  activeOpDataToEntry.clear();
  for (auto *data : getDataSet()) {
    activeOpDataToEntry.insert_or_assign(data, data->addOp(scope.name));
  }
  activeOpValid = true;
}

void CpuProfiler::stopOp(const Scope &scope) {
  (void)scope;
  if (!activeOpValid) {
    return;
  }
  emitKernelMetric(activeOpDataToEntry, activeOpStart, Clock::now());
  activeOpDataToEntry.clear();
  activeOpValid = false;
}

void CpuProfiler::enterScope(const Scope &scope) {
  ActiveScopeState state;
  state.scopeId = scope.scopeId;
  state.startTime = Clock::now();
  for (auto *data : getDataSet()) {
    state.dataToEntry.insert_or_assign(data, data->addOp(scope.name));
  }
  activeScopeStack.push_back(std::move(state));
}

void CpuProfiler::exitScope(const Scope &scope) {
  (void)scope;
  if (activeScopeStack.empty()) {
    return;
  }
  auto state = std::move(activeScopeStack.back());
  activeScopeStack.pop_back();
  emitKernelMetric(state.dataToEntry, state.startTime, Clock::now());
}

void CpuProfiler::doAddMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &scalarMetrics,
    const std::map<std::string, TensorMetric> &tensorMetrics) {
  (void)tensorMetrics;
  if (!activeOpDataToEntry.empty()) {
    emitScalarMetrics(activeOpDataToEntry, scalarMetrics);
    return;
  }

  if (!activeScopeStack.empty()) {
    emitScalarMetrics(activeScopeStack.back().dataToEntry, scalarMetrics);
    return;
  }

  if (!scalarMetrics.empty()) {
    for (auto *data : getDataSet()) {
      data->addMetrics(scopeId, scalarMetrics);
    }
  }
}

} // namespace proton
