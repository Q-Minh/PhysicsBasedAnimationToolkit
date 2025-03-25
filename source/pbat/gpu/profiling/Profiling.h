/**
 * @file Profiling.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Profiling utilities for host-side GPU code
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_PROFILING_PROFILING_H
#define PBAT_GPU_PROFILING_PROFILING_H

#if defined(PBAT_HAS_TRACY_PROFILER) and defined(__CUDACC__)
    #define PBAT_CAN_USE_TRACY_C
    #include <tracy/TracyC.h>
    #define PBAT_PROFILE_CUDA_HOST_SCOPE_START(var)             TracyCZone(var, true)
    #define PBAT_PROFILE_CUDA_NAMED_HOST_SCOPE_START(var, name) TracyCZoneN(var, name, true)
    #define PBAT_PROFILE_CUDA_HOST_SCOPE_END(var)               TracyCZoneEnd(var)
    #define PBAT_PROFILE_CUDA_LOG(txt, size)                    TracyCMessage(txt, size)
    #define PBAT_PROFILE_CUDA_SCOPED_LOG(ctx, txt, size)        TracyCZoneText(ctx, txt, size)
    #define PBAT_PROFILE_CUDA_PLOT(name, value)                 TracyCPlot(name, value)

    #define PBAT_PROFILE_CUDA_CONTEXT ___pbat_tracy_ctx
    #define PBAT_PROFILE_CUDA_NAMED_SCOPE(name)                                   \
        PBAT_PROFILE_CUDA_NAMED_HOST_SCOPE_START(PBAT_PROFILE_CUDA_CONTEXT, name) \
        pbat::gpu::profiling::Zone ___pbat_tracy_zone(&PBAT_PROFILE_CUDA_CONTEXT);

#else
    #define PBAT_PROFILE_CUDA_HOST_SCOPE_START(var)
    #define PBAT_PROFILE_CUDA_NAMED_HOST_SCOPE_START(var, name)
    #define PBAT_PROFILE_CUDA_HOST_SCOPE_END(var)
    #define PBAT_PROFILE_CUDA_LOG(txt, size)
    #define PBAT_PROFILE_CUDA_SCOPED_LOG(ctx, txt, size)
    #define PBAT_PROFILE_CUDA_PLOT(name, value)
    #define PBAT_PROFILE_CUDA_CONTEXT
    #define PBAT_PROFILE_CUDA_NAMED_SCOPE(name)
#endif // PBAT_CAN_USE_TRACY

#include <cstring>
#include <fmt/format.h>

#if defined(PBAT_CAN_USE_TRACY_C)
    #define PBAT_PROFILE_CUDA_SCOPED_CLOG(txt) \
        PBAT_PROFILE_CUDA_SCOPED_LOG(PBAT_PROFILE_CUDA_CONTEXT, txt, std::strlen(txt))
    #define PBAT_PROFILE_CUDA_CLOG(txt) PBAT_PROFILE_CUDA_LOG(txt, std::strlen(txt))
    #define PBAT_PROFILE_CUDA_SCOPED_SLOG(txt)                                                     \
        {                                                                                          \
            auto const& txtref = txt;                                                              \
            PBAT_PROFILE_CUDA_SCOPED_LOG(PBAT_PROFILE_CUDA_CONTEXT, txtref.c_str(), txtref.size()) \
        }
    #define PBAT_PROFILE_CUDA_SLOG(txt)                          \
        {                                                        \
            auto const& txtref = txt;                            \
            PBAT_PROFILE_CUDA_LOG(txtref.c_str(), txtref.size()) \
        }
    #define PBAT_PROFILE_CUDA_SCOPED_FLOG(fmtstr, ...)          \
        {                                                       \
            auto const& txt = fmt::format(fmtstr, __VA_ARGS__); \
            PBAT_PROFILE_CUDA_SCOPED_SLOG(txt)                  \
        }
    #define PBAT_PROFILE_CUDA_FLOG(fmtstr, ...)                 \
        {                                                       \
            auto const& txt = fmt::format(fmtstr, __VA_ARGS__); \
            PBAT_PROFILE_CUDA_LOG(txt.c_str(), txt.size());     \
        }
#endif // PBAT_CAN_USE_TRACY

/**
 * @namespace pbat::gpu::profiling
 * @brief Namespace for host-side GPU profiling utilities
 */
namespace pbat::gpu::profiling {

/**
 * @brief RAII class wrapping Tracy zones for CUDA host code
 */
class Zone
{
  public:
#ifdef PBAT_CAN_USE_TRACY_C
    Zone(TracyCZoneCtx* ctx);
#else
    Zone(void* ctx);
#endif
    ~Zone();

  private:
#ifdef PBAT_CAN_USE_TRACY_C
    TracyCZoneCtx* mContext;
#else
    void* mContext;
#endif
};

} // namespace pbat::gpu::profiling

#endif // PBAT_GPU_PROFILING_PROFILING_H
