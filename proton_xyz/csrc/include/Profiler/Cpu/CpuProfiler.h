#pragma once

#include "Context/Context.h"
#include "Data/Data.h"
#include "Profiler/Profiler.h"
#include "Utility/Singleton.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <vector>

namespace proton {

class CpuProfiler : public Profiler,
                    public OpInterface,
                    public ScopeInterface,
                    public Singleton<CpuProfiler> {
public:
  using Singleton<CpuProfiler>::instance;
  CpuProfiler() = default;
  ~CpuProfiler() override = default;

protected:
  void doStart() override {}
  void doStop() override {}
  void doFlush() override {}
  void doSetMode(const std::vector<std::string> &) override {}
  void doAddMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &scalarMetrics,
      const std::map<std::string, TensorMetric> &tensorMetrics) override;

  void startOp(const Scope &scope) override;
  void stopOp(const Scope &scope) override;
  void enterScope(const Scope &scope) override;
  void exitScope(const Scope &scope) override;

private:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  struct ActiveScopeState {
    size_t scopeId{};
    TimePoint startTime{};
    DataToEntryMap dataToEntry{};
  };

  static uint64_t toNs(TimePoint t);
  static void emitKernelMetric(DataToEntryMap &dataToEntry, TimePoint start,
                               TimePoint end);
  static void emitScalarMetrics(
      const DataToEntryMap &dataToEntry,
      const std::map<std::string, MetricValueType> &scalarMetrics);

  static thread_local TimePoint activeOpStart;
  static thread_local DataToEntryMap activeOpDataToEntry;
  static thread_local bool activeOpValid;
  static thread_local std::vector<ActiveScopeState> activeScopeStack;
};

} // namespace proton
