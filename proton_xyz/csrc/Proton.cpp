#include "Proton.h"

#include "Profiler/Instrumentation/CpuInstrumentationState.h"
#include "pybind11/pybind11.h"

#include <cstdint>

#define initProton protonXyzInitUpstreamBindings
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable)                                        \
  static void protonXyzInitUpstreamModule(pybind11::module_ &variable)
#include "../../third_party/triton/third_party/proton/csrc/Proton.cpp"
#pragma pop_macro("PYBIND11_MODULE")
#undef initProton

using namespace proton;

namespace {

void bindCpuInstrumentation(pybind11::module_ &m) {
  m.def("init_cpu_instrumentation_metadata",
        [](uint64_t functionId, const std::string &functionName,
           const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
           const std::vector<std::pair<size_t, size_t>> &scopeIdParents) {
          (void)functionName;
          (void)scopeIdParents;
          initCpuInstrumentationMetadata(functionId, scopeIdNames);
        });

  m.def("destroy_cpu_instrumentation_metadata", [](uint64_t functionId) {
    destroyCpuInstrumentationMetadata(functionId);
  });

  m.def("enter_cpu_instrumentation",
        [](uint64_t functionId) { enterCpuInstrumentation(functionId); });

  m.def("exit_cpu_instrumentation",
        [](uint64_t functionId) { exitCpuInstrumentation(functionId); });
}

} // namespace

extern "C" __attribute__((visibility("default"))) void
proton_cpu_record_start(int64_t scopeId) {
  Scope scope;
  if (!lookupCpuInstrumentationScope(static_cast<size_t>(scopeId), scope)) {
    return;
  }
  SessionManager::instance().enterScope(scope);
}

extern "C" __attribute__((visibility("default"))) void
proton_cpu_record_end(int64_t scopeId) {
  Scope scope;
  if (!lookupCpuInstrumentationScope(static_cast<size_t>(scopeId), scope)) {
    return;
  }
  SessionManager::instance().exitScope(scope);
}

extern "C" __attribute__((visibility("default"))) void
proton_cpu_instrumentation_enter(uint64_t functionId) {
  enterCpuInstrumentation(functionId);
}

extern "C" __attribute__((visibility("default"))) void
proton_cpu_instrumentation_exit(uint64_t functionId) {
  exitCpuInstrumentation(functionId);
}

PYBIND11_MODULE(libproton, m) {
  protonXyzInitUpstreamModule(m);
  auto protonModule =
      pybind11::reinterpret_borrow<pybind11::module_>(m.attr("proton"));
  bindCpuInstrumentation(protonModule);
}
