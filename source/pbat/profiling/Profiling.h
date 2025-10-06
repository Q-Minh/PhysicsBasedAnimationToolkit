/**
 * @file Profiling.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Profiling utilities for the Physics-Based Animation Toolkit (PBAT)
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_PROFILING_PROFILING_H
#define PBAT_PROFILING_PROFILING_H

#include "PhysicsBasedAnimationToolkitExport.h"

#if defined(PBAT_HAS_TRACY_PROFILER) and not defined(__CUDACC__)
    #define PBAT_CAN_USE_TRACY_CPP
    #include <tracy/Tracy.hpp>
    #define PBAT_PROFILE_SCOPE                 ZoneScoped
    #define PBAT_PROFILE_NAMED_SCOPE(name)     ZoneScopedN(name)
    #define PBAT_PROFILE_SCOPED_LOG(txt, size) ZoneText(txt, size)
    #define PBAT_PROFILE_LOG(txt, size)        TracyMessage(txt, size)
    #define PBAT_PROFILE_PLOT(name, value)     TracyPlot(name, value)
#else
    #define PBAT_PROFILE_SCOPE
    #define PBAT_PROFILE_NAMED_SCOPE(name)
    #define PBAT_PROFILE_LOG(txt, size)
    #define PBAT_PROFILE_SCOPED_LOG(txt, size)
    #define PBAT_PROFILE_PLOT(name, value)
#endif // defined(PBAT_HAS_TRACY_PROFILER) and not defined(__CUDACC__)

#include <cassert>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>

namespace pbat {
namespace profiling {

/**
 * @brief Begin a profiling frame with the given name
 *
 * @pre The frame name's length must not exceed 256 characters
 * @param name Frame name
 */
PBAT_API void BeginFrame(std::string_view name);

/**
 * @brief End the current profiling frame
 *
 * @pre The frame name's length must not exceed 256 characters
 * @param name Frame name
 */
PBAT_API void EndFrame(std::string_view name);

/**
 * @brief Check if PBAT's Tracy client is connected to the Tracy profiler server
 *
 * @return true if connected, false otherwise
 */
PBAT_API bool IsConnectedToServer();

/**
 * @brief Profile a function as a Tracy named zone
 *
 * @tparam Func Type of the function to profile
 * @tparam Args Types of the arguments to the function
 * @param zoneName Name of the zone
 * @param f Function to profile
 * @param args Arguments to the function
 * @return Result of the function
 * @note This function is only available if the Tracy profiler is enabled
 */
template <class Func, class... Args>
std::invoke_result_t<Func, Args...> Profile(std::string const& zoneName, Func&& f, Args&&... args)
{
#ifdef PBAT_CAN_USE_TRACY_CPP
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
#endif // PBAT_CAN_USE_TRACY
    return f(std::forward<Args>(args)...);
}

} // namespace profiling
} // namespace pbat

#endif // PBAT_PROFILING_PROFILING_H
