#include "BroydenIntegrator.cuh"

namespace pbat::gpu::impl::vbd {

BroydenIntegrator::BroydenIntegrator(Data const& data) : Integrator(data) {}

} // namespace pbat::gpu::impl::vbd