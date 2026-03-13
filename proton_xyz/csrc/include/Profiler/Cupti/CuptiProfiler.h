#ifndef PROTON_PROFILER_CUPTI_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_CUPTI_PROFILER_H_

#include "Profiler/Cpu/CpuProfiler.h"

namespace proton {

// proton_xyz is CPU-only, so we reuse the upstream Session.cpp selection logic
// by mapping legacy profiler names onto the local CPU profiler.
using CuptiProfiler = CpuProfiler;

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_CUPTI_PROFILER_H_
