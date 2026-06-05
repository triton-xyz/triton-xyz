#pragma once

#include "Data/Data.h"
#include "Data/PhaseStore.h"

#include <cstdint>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace proton {

class CpuInstrumentationTraceData : public Data {
public:
  CpuInstrumentationTraceData(const std::string &path,
                              ContextSource *contextSource = nullptr);
  ~CpuInstrumentationTraceData() override = default;

  std::string toJsonString(size_t phase) const override;
  std::vector<uint8_t> toMsgPack(size_t phase) const override;

  DataEntry addOp(size_t phase, size_t entryId,
                  const std::vector<Context> &contexts) override;
  void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) override;

protected:
  void enterScope(const Scope &scope) override;
  void exitScope(const Scope &scope) override;

  void doDump(std::ostream &os, OutputFormat outputFormat,
              size_t phase) const override;

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::ChromeTrace;
  }

private:
  struct Event {
    size_t id{};
    size_t scopeId{Scope::DummyScopeId};
    uint64_t threadId{};
    std::string name{};
    uint64_t startNs{};
    uint64_t endNs{};
    DataEntry::MetricSet metricSet{};
  };

  struct Trace {
    size_t nextEventId{};
    std::map<size_t, Event> events{};
  };

  Event &addEvent(Trace &trace, const Scope &scope, uint64_t threadId,
                  uint64_t startNs);
  Event *getEvent(size_t phase, size_t eventId);
  uint64_t getCurrentThreadTraceId();

  PhaseStore<Trace> tracePhases;
  std::map<std::thread::id, uint64_t> threadIdToTraceId;
  uint64_t nextThreadTraceId{};
};

} // namespace proton
