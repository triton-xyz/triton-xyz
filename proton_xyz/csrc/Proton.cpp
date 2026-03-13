#include "Proton.h"

#include "pybind11/pybind11.h"
// #include "pybind11/stl.h"
// #include "pybind11/stl_bind.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#define initProton protonXyzInitUpstreamBindings
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable)                                      \
  static void protonXyzInitUpstreamModule(pybind11::module_ &variable)
#include "../../third_party/triton/third_party/proton/csrc/Proton.cpp"
#pragma pop_macro("PYBIND11_MODULE")
#undef initProton

using namespace proton;

namespace {

using ScopeIdNameMap = std::map<size_t, std::string>;

std::mutex cpuInstrumentationMutex;
std::map<uint64_t, ScopeIdNameMap> functionScopeNames;
thread_local std::vector<uint64_t> activeFunctionIds;

bool lookupScope(size_t scopeId, Scope &scope) {
  if (activeFunctionIds.empty()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(cpuInstrumentationMutex);
  auto functionIt = functionScopeNames.find(activeFunctionIds.back());
  if (functionIt == functionScopeNames.end()) {
    return false;
  }

  auto scopeIt = functionIt->second.find(scopeId);
  if (scopeIt == functionIt->second.end()) {
    return false;
  }

  scope = Scope(scopeId, scopeIt->second);
  return true;
}

void eraseActiveFunctionId(uint64_t functionId) {
  auto it = std::find(activeFunctionIds.rbegin(), activeFunctionIds.rend(),
                      functionId);
  if (it == activeFunctionIds.rend()) {
    return;
  }
  activeFunctionIds.erase(std::next(it).base());
}

void bindCpuInstrumentation(pybind11::module_ &m) {
  m.def("init_cpu_instrumentation_metadata",
        [](uint64_t functionId, const std::string &functionName,
           const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
           const std::vector<std::pair<size_t, size_t>> &scopeIdParents) {
          (void)functionName;
          (void)scopeIdParents;
          ScopeIdNameMap scopeNameMap;
          for (const auto &[scopeId, scopeName] : scopeIdNames) {
            scopeNameMap.insert_or_assign(scopeId, scopeName);
          }
          std::lock_guard<std::mutex> lock(cpuInstrumentationMutex);
          functionScopeNames.insert_or_assign(functionId,
                                              std::move(scopeNameMap));
        });

  m.def("enter_cpu_instrumentation",
        [](uint64_t functionId) { activeFunctionIds.push_back(functionId); });

  m.def("exit_cpu_instrumentation",
        [](uint64_t functionId) { eraseActiveFunctionId(functionId); });
}

} // namespace

extern "C" __attribute__((visibility("default"))) void
proton_cpu_record_start(int64_t scopeId) {
  Scope scope;
  if (!lookupScope(static_cast<size_t>(scopeId), scope)) {
    return;
  }
  SessionManager::instance().enterScope(scope);
}

extern "C" __attribute__((visibility("default"))) void
proton_cpu_record_end(int64_t scopeId) {
  Scope scope;
  if (!lookupScope(static_cast<size_t>(scopeId), scope)) {
    return;
  }
  SessionManager::instance().exitScope(scope);
}

PYBIND11_MODULE(libproton, m) {
  protonXyzInitUpstreamModule(m);
  auto protonModule =
      pybind11::reinterpret_borrow<pybind11::module_>(m.attr("proton"));
  bindCpuInstrumentation(protonModule);
}
