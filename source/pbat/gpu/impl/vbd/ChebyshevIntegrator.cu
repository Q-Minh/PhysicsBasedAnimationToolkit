// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "ChebyshevIntegrator.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Kernels.h"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat::gpu::impl::vbd {

ChebyshevIntegrator::ChebyshevIntegrator(Data const& data)
    : Integrator(data),
      rho(static_cast<GpuScalar>(data.rho)),
      xkm1(data.X.cols()),
      xkm2(data.X.cols())
{
}

void ChebyshevIntegrator::Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.vbd.ChebyshevIntegrator.Solve");
    GpuScalar rho2 = rho * rho;
    GpuScalar omega{};
    for (auto k = 0; k < iterations; ++k)
    {
        using pbat::sim::vbd::kernels::ChebyshevOmega;
        omega = ChebyshevOmega(k, rho2, omega);
        RunVbdIteration(bdf);
        UpdateIterates(k, omega);
    }
}

void ChebyshevIntegrator::UpdateIterates(GpuIndex k, GpuScalar omega)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.vbd.ChebyshevIntegrator.UpdateIterates");
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        [k = k, omega = omega, xkm2 = xkm2.Raw(), xkm1 = xkm1.Raw(), xk = x.Raw()] PBAT_DEVICE(
            auto i) {
            using pbat::sim::vbd::kernels::ChebyshevUpdate;
            using pbat::math::linalg::mini::FromBuffers;
            using pbat::math::linalg::mini::ToBuffers;
            auto xkm2i = FromBuffers<3, 1>(xkm2, i);
            auto xkm1i = FromBuffers<3, 1>(xkm1, i);
            auto xki   = FromBuffers<3, 1>(xk, i);
            ChebyshevUpdate(k, omega, xkm2i, xkm1i, xki);
            ToBuffers(xkm2i, xkm2, i);
            ToBuffers(xkm1i, xkm1, i);
            ToBuffers(xki, xk, i);
        });
}

} // namespace pbat::gpu::impl::vbd