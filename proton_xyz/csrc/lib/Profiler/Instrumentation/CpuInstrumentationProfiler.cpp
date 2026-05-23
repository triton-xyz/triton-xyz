#include "Profiler/Instrumentation/CpuInstrumentationProfiler.h"
#include "Profiler/Instrumentation/CpuInstrumentationState.h"

#include "Data/Metric.h"
#include "Data/TreeData.h"
#include "Device.h"

#include <functional>
#include <iterator>
#include <stdexcept>
#include <thread>

namespace proton {

namespace {

uint64_t getCurrentThreadStreamId() {
  return static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

std::string getCurrentThreadContextName() {
  return "thread " + std::to_string(getCurrentThreadStreamId());
}

DataEntry addThreadAwareOp(Data *data, const Scope &scope,
                           bool insertThreadBeforeLeaf) {
  if (dynamic_cast<TreeData *>(data) == nullptr) {
    if (scope.name.empty()) {
      return data->addOp();
    }
    return data->addOp(scope.name);
  }

  auto contexts = data->getContexts();
  if (!scope.name.empty() &&
      (contexts.empty() || contexts.back().name != scope.name)) {
    contexts.emplace_back(scope.name);
  }
  auto insertIt = contexts.end();
  if (insertThreadBeforeLeaf && !contexts.empty()) {
    insertIt = std::prev(contexts.end());
  }
  contexts.insert(insertIt, Context(getCurrentThreadContextName()));
  return data->addOp(data->getPhaseInfo().current, Data::kRootEntryId,
                     contexts);
}

} // namespace

thread_local CpuInstrumentationProfiler::TimePoint
    CpuInstrumentationProfiler::activeKernelStart{};
thread_local DataToEntryMap
    CpuInstrumentationProfiler::activeKernelDataToEntry{};
thread_local bool CpuInstrumentationProfiler::activeKernelValid = false;
thread_local std::vector<CpuInstrumentationProfiler::ActiveScopeState>
    CpuInstrumentationProfiler::activeScopeStack{};

uint64_t CpuInstrumentationProfiler::toNs(TimePoint t) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch())
          .count());
}

void CpuInstrumentationProfiler::emitKernelMetric(DataToEntryMap &dataToEntry,
                                                  TimePoint start,
                                                  TimePoint end) {
  const auto startNs = toNs(start);
  const auto endNs = toNs(end);
  const auto safeEndNs = endNs >= startNs ? endNs : startNs;
  const auto streamId = getCurrentThreadStreamId();

  for (auto &[data, entry] : dataToEntry) {
    auto metric = std::make_unique<KernelMetric>(
        startNs, safeEndNs, /*invocations=*/1,
        /*deviceId=*/0, static_cast<uint64_t>(DeviceType::CPU), streamId);
    entry.upsertMetric(std::move(metric));
  }
}

void CpuInstrumentationProfiler::emitScalarMetrics(
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

void CpuInstrumentationProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  if (!modeAndOptions.empty() && !modeAndOptions.front().empty() &&
      modeAndOptions.front() != "cpu") {
    throw std::invalid_argument(
        "[PROTON_XYZ] CpuInstrumentationProfiler: unsupported mode prefix: " +
        modeAndOptions.front());
  }
}

void CpuInstrumentationProfiler::startOp(const Scope &scope) {
  activeKernelStart = Clock::now();
  activeKernelDataToEntry.clear();
  for (auto *data : getDataSet()) {
    activeKernelDataToEntry.insert_or_assign(
        data, addThreadAwareOp(data, scope, /*insertThreadBeforeLeaf=*/false));
  }
  activeKernelValid = true;
}

void CpuInstrumentationProfiler::stopOp(const Scope &scope) {
  (void)scope;
  if (!activeKernelValid) {
    return;
  }
  emitKernelMetric(activeKernelDataToEntry, activeKernelStart, Clock::now());
  activeKernelDataToEntry.clear();
  activeKernelValid = false;
}

void CpuInstrumentationProfiler::enterScope(const Scope &scope) {
  if (!isCpuInstrumentationScope(scope.scopeId)) {
    return;
  }

  ActiveScopeState state;
  state.scope = scope;
  state.startTime = Clock::now();
  for (auto *data : getDataSet()) {
    state.dataToEntry.insert_or_assign(
        data, addThreadAwareOp(data, scope, /*insertThreadBeforeLeaf=*/true));
  }
  activeScopeStack.push_back(std::move(state));
}

void CpuInstrumentationProfiler::exitScope(const Scope &scope) {
  if (!isCpuInstrumentationScope(scope.scopeId)) {
    return;
  }
  if (activeScopeStack.empty()) {
    return;
  }

  auto state = std::move(activeScopeStack.back());
  activeScopeStack.pop_back();
  if (state.scope != scope) {
    throw std::runtime_error(
        "[PROTON_XYZ] CpuInstrumentationProfiler: unbalanced scope exit");
  }
  emitKernelMetric(state.dataToEntry, state.startTime, Clock::now());
}

void CpuInstrumentationProfiler::doAddMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &scalarMetrics,
    const std::map<std::string, TensorMetric> &tensorMetrics) {
  (void)tensorMetrics;
  if (!activeScopeStack.empty()) {
    emitScalarMetrics(activeScopeStack.back().dataToEntry, scalarMetrics);
    return;
  }

  if (!activeKernelDataToEntry.empty()) {
    emitScalarMetrics(activeKernelDataToEntry, scalarMetrics);
    return;
  }

  if (!scalarMetrics.empty()) {
    for (auto *data : getDataSet()) {
      data->addMetrics(scopeId, scalarMetrics);
    }
  }
}

} // namespace proton
