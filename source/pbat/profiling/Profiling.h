#ifndef PBAT_PROFILING_PROFILING_H
#define PBAT_PROFILING_PROFILING_H

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

#endif // PBAT_PROFILING_PROFILING_H
