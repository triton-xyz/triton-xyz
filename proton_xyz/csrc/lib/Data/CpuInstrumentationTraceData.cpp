#include "Data/CpuInstrumentationTraceData.h"

#include "Utility/MsgPackWriter.h"
#include "nlohmann/json.hpp"

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using json = nlohmann::json;

namespace proton {

namespace {

constexpr const char *kCpuProcessName = "proton_cpu";

json makeProcessMetadata() {
  json metadata;
  metadata["ph"] = "M";
  metadata["name"] = "process_name";
  metadata["pid"] = kCpuProcessName;
  metadata["tid"] = 0;
  metadata["args"]["name"] = kCpuProcessName;
  return metadata;
}

json makeThreadMetadata(const std::string &threadName) {
  json metadata;
  metadata["ph"] = "M";
  metadata["name"] = "thread_name";
  metadata["pid"] = kCpuProcessName;
  metadata["tid"] = threadName;
  metadata["args"]["name"] = threadName;
  return metadata;
}

std::string getThreadName(const json &tidValue) {
  if (tidValue.is_string()) {
    return tidValue.get<std::string>();
  }
  if (tidValue.is_number_unsigned()) {
    return "thread " + std::to_string(tidValue.get<uint64_t>());
  }
  if (tidValue.is_number_integer()) {
    return "thread " + std::to_string(tidValue.get<int64_t>());
  }
  return {};
}

std::string normalizeChromeTrace(const std::string &traceText) {
  if (traceText.empty()) {
    return traceText;
  }

  std::istringstream input(traceText);
  std::string line;
  json merged = {{"displayTimeUnit", "us"}, {"traceEvents", json::array()}};
  merged["traceEvents"].push_back(makeProcessMetadata());

  std::set<std::string> seenThreads;
  bool sawTraceObject = false;
  bool sawKernelEvent = false;

  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    auto object = json::parse(line, nullptr, false);
    if (object.is_discarded() || !object.is_object()) {
      return traceText;
    }

    auto traceEventsIt = object.find("traceEvents");
    if (traceEventsIt == object.end() || !traceEventsIt->is_array()) {
      return traceText;
    }

    sawTraceObject = true;
    for (const auto &rawEvent : *traceEventsIt) {
      if (!rawEvent.is_object()) {
        return traceText;
      }

      auto event = rawEvent;
      auto tidIt = event.find("tid");
      if (tidIt == event.end()) {
        merged["traceEvents"].push_back(std::move(event));
        continue;
      }

      const auto threadName = getThreadName(*tidIt);
      if (threadName.empty()) {
        return traceText;
      }

      if (seenThreads.insert(threadName).second) {
        merged["traceEvents"].push_back(makeThreadMetadata(threadName));
      }

      event["pid"] = kCpuProcessName;
      event["tid"] = threadName;
      if (event.value("ph", "") == "X") {
        sawKernelEvent = true;
      }
      merged["traceEvents"].push_back(std::move(event));
    }
  }

  if (!sawTraceObject || !sawKernelEvent) {
    return traceText;
  }

  return merged.dump() + "\n";
}

} // namespace

std::string CpuInstrumentationTraceData::toJsonString(size_t phase) const {
  return normalizeChromeTrace(TraceData::toJsonString(phase));
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
    throw std::logic_error("Output format not supported");
  }
  os << toJsonString(phase);
}

} // namespace proton
