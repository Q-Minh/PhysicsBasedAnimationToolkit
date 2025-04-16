// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "AndersonIntegrator.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/gpu/profiling/Profiling.h"

namespace pbat::gpu::impl::vbd {

AndersonIntegrator::AndersonIntegrator(Data const& data)
    : Integrator(data),
      Fk(data.X.size()),
      Fkm1(data.X.size()),
      Gkm1(data.X.size()),
      xkm1(data.X.size()),
      DFK(data.X.size() * data.mAndersonWindowSize),
      DGK(data.X.size() * data.mAndersonWindowSize),
      alpha(data.mAndersonWindowSize)
{
}

void AndersonIntegrator::Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE("pbat.gpu.impl.vbd.AndersonIntegrator.Solve");
    auto nVertices = static_cast<GpuIndex>(x.Size());
    auto kDims     = x.Dimensions();
    auto n         = nVertices * kDims;
    auto m         = DFK.Size() / n;

    xkm1 = x;
    RunVbdIteration(bdf);
    Gkm1 = x;
    
    for (auto k = 1; k < iterations; ++k)
    {
        RunVbdIteration(bdf);
    }
}

} // namespace pbat::gpu::impl::vbd