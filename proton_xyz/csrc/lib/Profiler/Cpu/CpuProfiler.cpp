#include "Profiler/Cpu/CpuProfiler.h"

#include "Data/Metric.h"
#include "Device.h"

#include <functional>
#include <thread>

namespace proton {

thread_local CpuProfiler::TimePoint CpuProfiler::activeOpStart{};
thread_local DataToEntryMap CpuProfiler::activeOpDataToEntry{};
thread_local bool CpuProfiler::activeOpValid = false;

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
  const auto streamId = static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));

  for (auto &[data, entry] : dataToEntry) {
    auto metric = std::make_unique<KernelMetric>(
        startNs, safeEndNs, /*invocations=*/1,
        /*deviceId=*/0, static_cast<uint64_t>(DeviceType::CPU), streamId);
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
    if (scope.name.empty()) {
      activeOpDataToEntry.insert_or_assign(data, data->addOp());
    } else {
      activeOpDataToEntry.insert_or_assign(data, data->addOp(scope.name));
    }
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

void CpuProfiler::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &scalarMetrics,
    const std::map<std::string, TensorMetric> &tensorMetrics) {
  (void)tensorMetrics;
  if (!activeOpDataToEntry.empty()) {
    emitScalarMetrics(activeOpDataToEntry, scalarMetrics);
    return;
  }

  if (!scalarMetrics.empty()) {
    for (auto *data : getDataSet()) {
      data->addMetrics(scopeId, scalarMetrics);
    }
  }
}

} // namespace proton
