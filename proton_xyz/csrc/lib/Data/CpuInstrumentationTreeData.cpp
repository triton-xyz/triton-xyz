#include "Data/CpuInstrumentationTreeData.h"

#include "Profiler/Instrumentation/CpuInstrumentationState.h"

namespace proton {

void CpuInstrumentationTreeData::enterScope(const Scope &scope) {
  if (isCpuInstrumentationScope(scope.scopeId)) {
    return;
  }
  TreeData::enterScope(scope);
}

void CpuInstrumentationTreeData::exitScope(const Scope &scope) {
  if (isCpuInstrumentationScope(scope.scopeId)) {
    return;
  }
  TreeData::exitScope(scope);
}

} // namespace proton
