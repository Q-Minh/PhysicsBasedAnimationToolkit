#ifndef PBAT_PROFILING_PROFILING_H
#define PBAT_PROFILING_PROFILING_H

#include "PhysicsBasedAnimationToolkitExport.h"

#ifdef PBAT_HAS_TRACY_PROFILER
    #include <tracy/Tracy.hpp>
#endif // PBAT_HAS_TRACY_PROFILER

#ifdef PBAT_HAS_TRACY_PROFILER
    #define PBAT_PROFILE_SCOPE             ZoneScoped
    #define PBAT_PROFILE_NAMED_SCOPE(name) ZoneScopedN(name)
#else
    #define PBAT_PROFILE_SCOPE
    #define PBAT_PROFILE_NAMED_SCOPE(name)
#endif // PBAT_HAS_TRACY_PROFILER

#include <string_view>
#include <type_traits>

namespace pbat {
namespace profiling {

PBAT_API void BeginFrame(std::string_view name);

PBAT_API void EndFrame(std::string_view name);

PBAT_API bool IsConnectedToServer();

template <class Func, class... Args>
std::invoke_result_t<Func, Args...> Profile(std::string_view zoneName, Func&& f, Args&&... args)
{
    PBAT_PROFILE_SCOPE;
    ZoneName(zoneName.data(), zoneName.size());
    return f(std::forward<Args>(args)...);
}

} // namespace profiling
} // namespace pbat

#endif // PBAT_PROFILING_PROFILING_H
