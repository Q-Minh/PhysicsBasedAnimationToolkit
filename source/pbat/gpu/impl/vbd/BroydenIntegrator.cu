// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "BroydenIntegrator.cuh"

namespace pbat::gpu::impl::vbd {

BroydenIntegrator::BroydenIntegrator(Data const& data) : Integrator(data) {}

void BroydenIntegrator::Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations)
{
    BaseType::Solve(bdf, iterations);
}

} // namespace pbat::gpu::impl::vbd