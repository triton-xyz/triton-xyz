#include "Profiler/Instrumentation/CpuInstrumentationState.h"

#include <algorithm>
#include <mutex>

namespace proton {

namespace {

std::mutex cpuInstrumentationMutex;
std::map<uint64_t, ScopeIdNameMapPtr> functionScopeNames;
thread_local std::vector<ActiveCpuInstrumentationFunction> activeFunctionScopes;

} // namespace

void initCpuInstrumentationMetadata(
    uint64_t functionId,
    const std::vector<std::pair<size_t, std::string>> &scopeIdNames) {
  auto scopeNameMap = std::make_shared<ScopeIdNameMap>();
  for (const auto &[scopeId, scopeName] : scopeIdNames) {
    scopeNameMap->insert_or_assign(scopeId, scopeName);
  }
  std::lock_guard<std::mutex> lock(cpuInstrumentationMutex);
  functionScopeNames.insert_or_assign(functionId, std::move(scopeNameMap));
}

void destroyCpuInstrumentationMetadata(uint64_t functionId) {
  std::lock_guard<std::mutex> lock(cpuInstrumentationMutex);
  functionScopeNames.erase(functionId);
}

void enterCpuInstrumentation(uint64_t functionId) {
  ScopeIdNameMapPtr scopeNames;
  {
    std::lock_guard<std::mutex> lock(cpuInstrumentationMutex);
    auto functionIt = functionScopeNames.find(functionId);
    if (functionIt != functionScopeNames.end()) {
      scopeNames = functionIt->second;
    }
  }
  activeFunctionScopes.push_back(
      ActiveCpuInstrumentationFunction{functionId, std::move(scopeNames)});
}

void exitCpuInstrumentation(uint64_t functionId) {
  auto it =
      std::find_if(activeFunctionScopes.rbegin(), activeFunctionScopes.rend(),
                   [&](const ActiveCpuInstrumentationFunction &activeFn) {
                     return activeFn.functionId == functionId;
                   });
  if (it == activeFunctionScopes.rend()) {
    return;
  }
  activeFunctionScopes.erase(std::next(it).base());
}

std::optional<Scope> lookupCpuInstrumentationScope(size_t scopeId) {
  if (activeFunctionScopes.empty()) {
    return std::nullopt;
  }

  const auto &scopeNames = activeFunctionScopes.back().scopeNames;
  if (!scopeNames) {
    return std::nullopt;
  }

  auto scopeIt = scopeNames->find(scopeId);
  if (scopeIt == scopeNames->end()) {
    return std::nullopt;
  }

  return Scope(scopeId, scopeIt->second);
}

bool isCpuInstrumentationScope(size_t scopeId) {
  if (activeFunctionScopes.empty()) {
    return false;
  }
  const auto &scopeNames = activeFunctionScopes.back().scopeNames;
  return scopeNames && scopeNames->find(scopeId) != scopeNames->end();
}

} // namespace proton
