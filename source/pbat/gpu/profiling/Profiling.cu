// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Profiling.h"

namespace pbat::gpu::profiling {

Zone::Zone(TracyCZoneCtx* ctx) : mContext(ctx)
{
}

Zone::~Zone()
{
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(*mContext);
}

} // namespace pbat::gpu::profiling