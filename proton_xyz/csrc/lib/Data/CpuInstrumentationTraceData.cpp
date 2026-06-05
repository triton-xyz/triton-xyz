#include "Data/CpuInstrumentationTraceData.h"

#include "Utility/Errors.h"
#include "Utility/MsgPackWriter.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace proton {

namespace {

constexpr const char *kCpuProcessName = "proton_cpu";
constexpr size_t kMaxActiveEventStackCacheObjects = 10;

thread_local std::map<const CpuInstrumentationTraceData *, std::vector<size_t>>
    traceDataToActiveEventStack;

uint64_t getCurrentCpuTimestampNs() {
  using Clock = std::chrono::system_clock;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

double toChromeTraceUs(uint64_t ns) { return static_cast<double>(ns) / 1000.0; }

json makeProcessMetadata() {
  json metadata;
  metadata["ph"] = "M";
  metadata["name"] = "process_name";
  metadata["pid"] = kCpuProcessName;
  metadata["tid"] = 0;
  metadata["args"]["name"] = kCpuProcessName;
  return metadata;
}

json makeThreadMetadata(uint64_t threadId) {
  const auto threadName = "thread " + std::to_string(threadId);
  json metadata;
  metadata["ph"] = "M";
  metadata["name"] = "thread_name";
  metadata["pid"] = kCpuProcessName;
  metadata["tid"] = threadId;
  metadata["args"]["name"] = threadName;
  return metadata;
}

} // namespace

CpuInstrumentationTraceData::CpuInstrumentationTraceData(
    const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(tracePhases);
}

CpuInstrumentationTraceData::Event &
CpuInstrumentationTraceData::addEvent(Trace &trace, const Scope &scope,
                                      uint64_t threadId, uint64_t startNs) {
  auto id = trace.nextEventId++;
  auto [it, inserted] = trace.events.try_emplace(id);
  (void)inserted;
  auto &event = it->second;
  event.id = id;
  event.scopeId = scope.scopeId;
  event.threadId = threadId;
  event.name = scope.name;
  event.startNs = startNs;
  return event;
}

CpuInstrumentationTraceData::Event *
CpuInstrumentationTraceData::getEvent(size_t phase, size_t eventId) {
  auto *trace = phasePtrAs<Trace>(phase);
  auto it = trace->events.find(eventId);
  if (it == trace->events.end()) {
    return nullptr;
  }
  return &it->second;
}

uint64_t CpuInstrumentationTraceData::getCurrentThreadTraceId() {
  auto threadId = std::this_thread::get_id();
  auto it = threadIdToTraceId.find(threadId);
  if (it != threadIdToTraceId.end()) {
    return it->second;
  }
  auto traceThreadId = nextThreadTraceId++;
  threadIdToTraceId.emplace(threadId, traceThreadId);
  return traceThreadId;
}

void CpuInstrumentationTraceData::enterScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *trace = currentPhasePtrAs<Trace>();
  auto &event = addEvent(*trace, scope, getCurrentThreadTraceId(),
                         getCurrentCpuTimestampNs());
  traceDataToActiveEventStack[this].push_back(event.id);
}

void CpuInstrumentationTraceData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto activeEventStackIt = traceDataToActiveEventStack.find(this);
  if (activeEventStackIt == traceDataToActiveEventStack.end() ||
      activeEventStackIt->second.empty()) {
    return;
  }

  auto &activeEventStack = activeEventStackIt->second;
  const auto phase = currentPhase.load(std::memory_order_relaxed);
  auto eventIt = activeEventStack.end();
  while (eventIt != activeEventStack.begin()) {
    --eventIt;
    auto *event = getEvent(phase, *eventIt);
    if (event != nullptr && event->scopeId == scope.scopeId &&
        event->name == scope.name) {
      event->endNs = getCurrentCpuTimestampNs();
      activeEventStack.erase(eventIt);
      break;
    }
  }

  if (activeEventStack.empty() &&
      traceDataToActiveEventStack.size() > kMaxActiveEventStackCacheObjects) {
    traceDataToActiveEventStack.erase(this);
  }
}

DataEntry
CpuInstrumentationTraceData::addOp(size_t phase, size_t entryId,
                                   const std::vector<Context> &contexts) {
  (void)entryId;
  auto lock = lockIfCurrentOrVirtualPhase(phase);
  auto *trace = phasePtrAs<Trace>(phase);
  auto name = contexts.empty() ? std::string{} : contexts.back().name;
  auto &event = addEvent(*trace, Scope(name), getCurrentThreadTraceId(), 0);
  return DataEntry(event.id, phase, event.metricSet);
}

void CpuInstrumentationTraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  if (metrics.empty()) {
    return;
  }

  std::unique_lock<std::shared_mutex> lock(mutex);
  auto activeEventStackIt = traceDataToActiveEventStack.find(this);
  if (activeEventStackIt == traceDataToActiveEventStack.end()) {
    return;
  }

  const auto phase = currentPhase.load(std::memory_order_relaxed);
  for (auto eventIt = activeEventStackIt->second.rbegin();
       eventIt != activeEventStackIt->second.rend(); ++eventIt) {
    auto *event = getEvent(phase, *eventIt);
    if (event == nullptr || event->scopeId != scopeId) {
      continue;
    }
    DataEntry(event->id, phase, event->metricSet)
        .upsertFlexibleMetrics(metrics);
    return;
  }
}

std::string CpuInstrumentationTraceData::toJsonString(size_t phase) const {
  json traceJson = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
  traceJson["traceEvents"].push_back(makeProcessMetadata());

  tracePhases.withPtr(phase, [&](const Trace *trace) {
    if (trace == nullptr) {
      return;
    }
    std::set<uint64_t> seenThreadIds;
    for (const auto &[_, event] : trace->events) {
      if (event.startNs == 0 || event.endNs == 0 ||
          event.endNs < event.startNs || event.name.empty()) {
        continue;
      }

      if (seenThreadIds.insert(event.threadId).second) {
        traceJson["traceEvents"].push_back(makeThreadMetadata(event.threadId));
      }

      json traceEvent;
      traceEvent["name"] = event.name;
      traceEvent["cat"] = "cpu";
      traceEvent["ph"] = "X";
      traceEvent["pid"] = kCpuProcessName;
      traceEvent["tid"] = event.threadId;
      traceEvent["ts"] = toChromeTraceUs(event.startNs);
      traceEvent["dur"] = toChromeTraceUs(event.endNs - event.startNs);
      traceJson["traceEvents"].push_back(std::move(traceEvent));
    }
  });

  return traceJson.dump() + "\n";
}

std::vector<uint8_t>
CpuInstrumentationTraceData::toMsgPack(size_t phase) const {
  auto traceJson = toJsonString(phase);
  MsgPackWriter writer;
  writer.packStr(traceJson);
  return std::move(writer).take();
}

void CpuInstrumentationTraceData::doDump(std::ostream &os,
                                         OutputFormat outputFormat,
                                         size_t phase) const {
  if (outputFormat != OutputFormat::ChromeTrace) {
    throw makeInvalidArgument("Output format not supported");
  }
  os << toJsonString(phase);
}

} // namespace proton
