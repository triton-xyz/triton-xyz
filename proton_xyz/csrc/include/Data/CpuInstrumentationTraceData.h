#pragma once

#include "Data/TraceData.h"

namespace proton {

class CpuInstrumentationTraceData : public TraceData {
public:
  using TraceData::TraceData;

  std::string toJsonString(size_t phase) const override;
  std::vector<uint8_t> toMsgPack(size_t phase) const override;

protected:
  void doDump(std::ostream &os, OutputFormat outputFormat,
              size_t phase) const override;
};

} // namespace proton
