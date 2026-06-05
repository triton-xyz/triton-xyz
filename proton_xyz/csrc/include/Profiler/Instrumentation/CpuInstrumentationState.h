#pragma once

#include "Context/Context.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace proton {

using ScopeIdNameMap = std::map<size_t, std::string>;
using ScopeIdNameMapPtr = std::shared_ptr<const ScopeIdNameMap>;

struct ActiveCpuInstrumentationFunction {
  uint64_t functionId{};
  ScopeIdNameMapPtr scopeNames{};
};

void initCpuInstrumentationMetadata(
    uint64_t functionId,
    const std::vector<std::pair<size_t, std::string>> &scopeIdNames);

void destroyCpuInstrumentationMetadata(uint64_t functionId);

void enterCpuInstrumentation(uint64_t functionId);

void exitCpuInstrumentation(uint64_t functionId);

std::optional<Scope> lookupCpuInstrumentationScope(size_t scopeId);

bool isCpuInstrumentationScope(size_t scopeId);

} // namespace proton
