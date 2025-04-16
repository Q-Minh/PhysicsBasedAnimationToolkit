#include "AndersonIntegrator.cuh"

namespace pbat::gpu::impl::vbd {

AndersonIntegrator::AndersonIntegrator(Data const& data)
    : Integrator(data),
      Fk(data.X.size()),
      Fkm1(data.X.size()),
      Gkm1(data.X.size()),
      xkm1(data.X.size()),
      DFK(data.X.size() * data.mAndersonWindowSize),
      DGK(data.X.size()* data.mAndersonWindowSize),
      alpha(data.mAndersonWindowSize)
{
}

void AndersonIntegrator::Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations) {}

} // namespace pbat::gpu::impl::vbd