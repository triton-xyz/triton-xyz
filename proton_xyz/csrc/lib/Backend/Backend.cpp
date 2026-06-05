#include "Backend/Backend.h"

#include "Profiler/Cupti/CuptiProfiler.h"
#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Utility/String.h"

#include <algorithm>
#include <vector>

namespace proton {

const std::vector<BackendRegistration> &getBackendRegistrations() {
  static const std::vector<BackendRegistration> registrations = {};
  return registrations;
}

const std::vector<ProfilerRegistration> getProfilerRegistrations() {
  return {
      {"cupti", "cuda", []() { return &CuptiProfiler::instance(); }},
      {"roctracer", {}, []() { return &RoctracerProfiler::instance(); }},
      {"instrumentation",
       {},
       []() { return &InstrumentationProfiler::instance(); }},
  };
}

const std::vector<DeviceRegistration> getDeviceRegistrations() { return {}; }

const std::vector<RuntimeRegistration> getRuntimeRegistrations() { return {}; }

const std::vector<std::string> getRegisteredProfilerNames() {
  const auto profilers = getProfilerRegistrations();
  std::vector<std::string> profilerNames(profilers.size());
  std::transform(
      profilers.begin(), profilers.end(), profilerNames.begin(),
      [](const ProfilerRegistration &entry) { return entry.getName(); });
  return profilerNames;
}

const std::optional<std::string>
getProfilerForTritonBackend(const std::string &tritonBackend) {
  const auto profilers = getProfilerRegistrations();
  auto itr = std::find_if(profilers.begin(), profilers.end(),
                          [&](const ProfilerRegistration &entry) {
                            return proton::toLower(tritonBackend) ==
                                   proton::toLower(
                                       entry.getTritonBackend().value_or(""));
                          });
  if (itr == profilers.end()) {
    return {};
  }
  return itr->getName();
}

} // namespace proton
