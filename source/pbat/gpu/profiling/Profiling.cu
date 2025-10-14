// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Profiling.h"

#if defined(PBAT_HAS_TRACY_PROFILER)
    // Tracy's CUDACtx singleton uses locks for reference counting.
    // See https://github.com/microsoft/STL/issues/4730.
    #if defined(_MSC_VER)
        #define _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
    #endif // defined(_MSC_VER)
    #include <tracy/TracyCUDA.hpp>
#endif // defined(PBAT_HAS_TRACY_PROFILER)

namespace pbat::gpu::profiling {

CudaProfiler::CudaProfiler() : mContext(nullptr)
{
#if defined(PBAT_HAS_TRACY_PROFILER)
    mContext = TracyCUDAContext();
    // TracyCUDAContextName(
    //     static_cast<tracy::CUDACtx*>(mContext),
    //     context.data(),
    //     static_cast<uint16_t>(context.size()));
#endif // defined(PBAT_HAS_TRACY_PROFILER)
}

void CudaProfiler::Start()
{
#if defined(PBAT_HAS_TRACY_PROFILER)
    TracyCUDAStartProfiling(static_cast<tracy::CUDACtx*>(mContext));
#endif // defined(PBAT_HAS_TRACY_PROFILER)
}

void CudaProfiler::Stop()
{
#if defined(PBAT_HAS_TRACY_PROFILER)
    TracyCUDAStopProfiling(static_cast<tracy::CUDACtx*>(mContext));
#endif // defined(PBAT_HAS_TRACY_PROFILER)
}

CudaProfiler::~CudaProfiler()
{
#if defined(PBAT_HAS_TRACY_PROFILER)
    if (mContext)
    {
        TracyCUDAContextDestroy(static_cast<tracy::CUDACtx*>(mContext));
    }
#endif // defined(PBAT_HAS_TRACY_PROFILER)
}

} // namespace pbat::gpu::profiling

#include <doctest/doctest.h>

TEST_CASE("[gpu][profiling] CudaProfiler")
{
    CHECK_NOTHROW(pbat::gpu::profiling::CudaProfiler{});
    pbat::gpu::profiling::CudaProfiler profiler{};
    CHECK_NOTHROW(profiler.Start());
    CHECK_NOTHROW(profiler.Stop());
}