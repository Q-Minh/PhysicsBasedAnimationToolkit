#ifndef PBAT_PROFILING_PROFILING_H
#define PBAT_PROFILING_PROFILING_H

#if defined(__CUDACC__)
static_assert(false, "Cannot #include tracy headers in CUDA code.");
#endif // __CUDACC__

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

#include <map>
#include <string>
#include <string_view>
#include <type_traits>

namespace pbat {
namespace profiling {

PBAT_API void BeginFrame(std::string_view name);

PBAT_API void EndFrame(std::string_view name);

PBAT_API bool IsConnectedToServer();

template <class Func, class... Args>
std::invoke_result_t<Func, Args...> Profile(std::string const& zoneName, Func&& f, Args&&... args)
{
#ifdef PBAT_HAS_TRACY_PROFILER
    static auto constexpr line = (uint32_t)TracyLine;
    struct SourceLocationData
    {
        SourceLocationData(std::string_view zoneNameView)
            : name(zoneNameView), function(TracyFunction), file(TracyFile), data()
        {
            data.name     = name.data();
            data.function = function.data();
            data.file     = file.data();
            data.line     = line;
            data.color    = 0;
        }
        std::string name;
        std::string function;
        std::string file;
        tracy::SourceLocationData data;
    };
    static std::map<std::string, SourceLocationData> zones{};
    auto it = zones.find(zoneName);
    if (it == zones.end())
    {
        bool inserted{false};
        std::tie(it, inserted) = zones.insert({zoneName, SourceLocationData(zoneName)});
        assert(inserted);
    }
    SourceLocationData const& data = it->second;
    tracy::ScopedZone zone(&(data.data));
#endif // PBAT_HAS_TRACY_PROFILER
    return f(std::forward<Args>(args)...);
}

} // namespace profiling
} // namespace pbat

#endif // PBAT_PROFILING_PROFILING_H
