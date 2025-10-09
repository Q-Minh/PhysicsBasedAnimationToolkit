// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Profiling.h"

#if defined(PBAT_HAS_TRACY_PROFILER)
    #include <tracy/TracyCUDA.hpp>
#endif // defined(PBAT_HAS_TRACY_PROFILER)

namespace pbat::gpu::profiling {

CudaProfiler::CudaProfiler(std::string_view context)
    :
#if defined(PBAT_HAS_TRACY_PROFILER)
      mContext(TracyCUDAContext())
#else
      mContext(nullptr)
#endif // defined(PBAT_HAS_TRACY_PROFILER)
{
#if defined(PBAT_HAS_TRACY_PROFILER)
    TracyCUDAContextName(
        static_cast<tracy::CUDACtx*>(mContext),
        context.data(),
        static_cast<uint16_t>(context.size()));
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
    TracyCUDAContextDestroy(static_cast<tracy::CUDACtx*>(mContext));
#endif // defined(PBAT_HAS_TRACY_PROFILER)
}

} // namespace pbat::gpu::profiling