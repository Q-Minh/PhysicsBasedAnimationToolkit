#ifndef PBAT_PROFILING_PROFILING_H
#define PBAT_PROFILING_PROFILING_H

#include "PhysicsBasedAnimationToolkitExport.h"

#ifdef PBAT_HAS_TRACY_PROFILER
    #include <tracy/Tracy.hpp>
#endif // PBAT_HAS_TRACY_PROFILER

#ifdef PBAT_HAS_TRACY_PROFILER
    #define PBA_PROFILE_SCOPE             ZoneScoped
    #define PBA_PROFILE_NAMED_SCOPE(name) ZoneScopedN(name)
#else
    #define PBA_PROFILE_SCOPE
    #define PBA_PROFILE_NAMED_SCOPE(name)
#endif // PBAT_HAS_TRACY_PROFILER

#include <string_view>

namespace pbat {
namespace profiling {

PBAT_API void BeginFrame(std::string_view name);

PBAT_API void EndFrame(std::string_view name);

PBAT_API bool IsConnectedToServer();

} // namespace profiling
} // namespace pbat

#endif // PBAT_PROFILING_PROFILING_H
