#ifndef PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH
#define PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/math/linalg/Matrix.cuh"

#include <array>
#include <cstddef>

namespace pbat {
namespace gpu {
namespace vbd {
namespace kernels {

struct FInertialTarget
{
    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            xtilde[d][i] = xt[d][i] + dt * vt[d][i] + dt2 * aext[d][i];
        }
    }

    GpuScalar dt;
    GpuScalar dt2;
    std::array<GpuScalar const*, 3> xt;
    std::array<GpuScalar const*, 3> vt;
    std::array<GpuScalar const*, 3> aext;
    std::array<GpuScalar*, 3> xtilde;
}

struct FAdaptiveInitialization
{
    using Vector3 = pbat::gpu::math::linalg::Matrix<GpuScalar, 3>;

    __device__ Vector3 GetExternalAcceleration(auto i) const
    {
        Vector3 aexti;
        aexti(0) = aext[0][i];
        aexti(1) = aext[1][i];
        aexti(2) = aext[2][i];
        return aexti;
    }

    __device__ Vector3 GetAcceleration(auto i) const
    {
        Vector3 at;
        at(0) = (vt[0][i] - vtm1[0][i]) / dt;
        at(1) = (vt[1][i] - vtm1[1][i]) / dt;
        at(2) = (vt[2][i] - vtm1[2][i]) / dt;
        return at;
    }

    __device__ void operator()(auto i)
    {
        using namespace pbat::gpu::math::linalg;
        Vector3 const aexti    = GetExternalAcceleration(i);
        GpuScalar const atext  = Dot(GetAcceleration(i), aexti) / SquaredNorm(aexti, aexti);
        GpuScalar atilde       = min(max(atext, GpuScalar{0}), GpuScalar{1});
        bool const bWasClamped = (atilde == GpuScalar{0}) or (atilde == GpuScalar{1});
        atilde                 = (not bWasClamped) * atext + bWasClamped * atilde;
        for (auto d = 0; d < 3; ++d)
            x[d][i] = xt[d][i] + dt * vt[d][i] + dt2 * atilde * aext[d][i];
    }

    GpuScalar dt;
    GpuScalar dt2;
    std::array<GpuScalar const*, 3> xt;
    std::array<GpuScalar const*, 3> vtm1;
    std::array<GpuScalar const*, 3> vt;
    std::array<GpuScalar const*, 3> aext;
    std::array<GpuScalar*, 3> x;
};

struct FChebyshev
{
    FChebyshev(
        GpuScalar rho,
        std::array<GpuScalar const*, 3> xtm1,
        std::array<GpuScalar const*, 3> xt,
        std::array<GpuScalar*, 3> x)
        : rho2(rho * rho), omega(GpuScalar{1}), xtm1(xtm1), xt(xt), x(x)
    {
    }

    __host__ Update(auto k)
    {
        omega = (k == 1) ? omega = GpuScalar{2} / (GpuScalar{2} - rho2) :
                           GpuScalar{4} / (GpuScalar{4} - rho2 * omega);
    }

    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
            x[d][i] = omega * (x[d][i] - xt[d][i]) + xtm1[d][i];
    }

    GpuScalar rho2;
    GpuScalar omega;
    std::array<GpuScalar const*, 3> xtm1;
    std::array<GpuScalar const*, 3> xt;
    std::array<GpuScalar*, 3> x;
};

} // namespace kernels
} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH