#include "XpbdImpl.cuh"

#include <array>
#include <thrust/async/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace xpbd {

struct FInitializeSolution
{
    __device__ void operator()(GpuIndex i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            x[d][i] = xt[d][i] + dt * v[d][i] + dt2 * f[d][i] / m[i];
        }
    }

    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
    std::array<GpuScalar*, 3> f;
    GpuScalar* m;
    GpuScalar dt;
    GpuScalar dt2;
};

struct FProjectConstraint
{
    __device__ void operator()(GpuIndex c)
    {
        GpuIndex const v[4]     = {T[0][c], T[1][c], T[2][c], T[3][c]};
        GpuScalar const* DmInvC = DmInv + 9 * c;
        // dC/dx
        GpuScalar gradC[12] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        // dPsi/dF (i.e. Piola Kirchhoff stress)
        GpuScalar P[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        // Deformation gradient F = Ds * DmInv
        GpuScalar F[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        {
            // Column-major matrix Ds = [ (x1-x4) (x2-x4) (x3-x4) ]
            GpuScalar Ds[9];
            Ds[0] = x[0][v[0]] - x[0][v[3]];
            Ds[3] = x[0][v[1]] - x[0][v[3]];
            Ds[6] = x[0][v[2]] - x[0][v[3]];
            Ds[1] = x[1][v[0]] - x[1][v[3]];
            Ds[4] = x[1][v[1]] - x[1][v[3]];
            Ds[7] = x[1][v[2]] - x[1][v[3]];
            Ds[2] = x[2][v[0]] - x[2][v[3]];
            Ds[5] = x[2][v[1]] - x[2][v[3]];
            Ds[8] = x[2][v[2]] - x[2][v[3]];
            F[0]  = DmInvC[0] * Ds[0] + DmInvC[1] * Ds[3] + DmInvC[2] * Ds[6];
            F[1]  = DmInvC[0] * Ds[1] + DmInvC[1] * Ds[4] + DmInvC[2] * Ds[7];
            F[2]  = DmInvC[0] * Ds[2] + DmInvC[1] * Ds[5] + DmInvC[2] * Ds[8];
            F[3]  = DmInvC[3] * Ds[0] + DmInvC[4] * Ds[3] + DmInvC[5] * Ds[6];
            F[4]  = DmInvC[3] * Ds[1] + DmInvC[4] * Ds[4] + DmInvC[5] * Ds[7];
            F[5]  = DmInvC[3] * Ds[2] + DmInvC[4] * Ds[5] + DmInvC[5] * Ds[8];
            F[6]  = DmInvC[6] * Ds[0] + DmInvC[7] * Ds[3] + DmInvC[8] * Ds[6];
            F[7]  = DmInvC[6] * Ds[1] + DmInvC[7] * Ds[4] + DmInvC[8] * Ds[7];
            F[8]  = DmInvC[6] * Ds[2] + DmInvC[7] * Ds[5] + DmInvC[8] * Ds[8];
        }
        {
            // Compute volume conservation constraints, i.e. CH = det(F) - 1,
            // grad CH = [f2xf3, f3xf1, f1xf2]
            GpuScalar CH = F[0] * F[4] * F[8] - F[0] * F[5] * F[7] - F[1] * F[3] * F[8] +
                           F[1] * F[5] * F[6] + F[2] * F[3] * F[7] - F[2] * F[4] * F[6] - 1.f;
            P[0]      = F[4] * F[8] - F[5] * F[7];
            P[3]      = -F[3] * F[8] + F[5] * F[6];
            P[6]      = F[3] * F[7] - F[4] * F[6];
            P[1]      = -F[1] * F[8] + F[2] * F[7];
            P[4]      = F[0] * F[8] - F[2] * F[6];
            P[7]      = -F[0] * F[7] + F[1] * F[6];
            P[2]      = F[1] * F[5] - F[2] * F[4];
            P[5]      = -F[0] * F[5] + F[2] * F[3];
            P[8]      = F[0] * F[4] - F[1] * F[3];
            gradC[0]  = P[0] * DmInvC[0] + P[3] * DmInvC[3] + P[6] * DmInvC[6];
            gradC[1]  = P[1] * DmInvC[0] + P[4] * DmInvC[3] + P[7] * DmInvC[6];
            gradC[2]  = P[2] * DmInvC[0] + P[5] * DmInvC[3] + P[8] * DmInvC[6];
            gradC[3]  = P[0] * DmInvC[1] + P[3] * DmInvC[4] + P[6] * DmInvC[7];
            gradC[4]  = P[1] * DmInvC[1] + P[4] * DmInvC[4] + P[7] * DmInvC[7];
            gradC[5]  = P[2] * DmInvC[1] + P[5] * DmInvC[4] + P[8] * DmInvC[7];
            gradC[6]  = P[0] * DmInvC[2] + P[3] * DmInvC[5] + P[6] * DmInvC[8];
            gradC[7]  = P[1] * DmInvC[2] + P[4] * DmInvC[5] + P[7] * DmInvC[8];
            gradC[8]  = P[2] * DmInvC[2] + P[5] * DmInvC[5] + P[8] * DmInvC[8];
            gradC[9]  = -gradC[0] - gradC[3] - gradC[6];
            gradC[10] = -gradC[1] - gradC[4] - gradC[7];
            gradC[11] = -gradC[2] - gradC[5] - gradC[8];
        }
    }

    GpuIndex const* partition;

    std::array<GpuScalar*, 3> x;
    GpuScalar* lambda;

    std::array<GpuIndex*, 4> T;
    GpuScalar* m;
    GpuScalar* alpha;
    GpuScalar* DmInv;
};

struct FFinalizeSolution
{
    __device__ void operator()(GpuIndex i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            v[d][i]  = (x[d][i] - xt[d][i]) / dt;
            xt[d][i] = x[d][i];
        }
    }

    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
    GpuScalar dt;
};

void XpbdImpl::Step(GpuScalar dt)
{
    GpuScalar const sdt       = dt / static_cast<GpuScalar>(S);
    GpuScalar const sdt2      = sdt * sdt;
    GpuIndex const nParticles = static_cast<GpuIndex>(NumberOfParticles());
    // TODO: Detect collision candidates and setup collision constraint solve
    // ...

    for (auto s = 0; s < S; ++s)
    {
        // Reset "Lagrange" multipliers
        thrust::fill(lambda.Data(), lambda.Data() + lambda.Size(), GpuScalar{0.});
        // Initialize constraint solve
        thrust::device_event e = thrust::async::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            FInitializeSolution{xt.Raw(), mV.x.Raw(), v.Raw(), f.Raw(), m.Raw(), sdt, sdt2});
        // Solve constraints
        for (auto k = 0; k < K; ++k)
        {
            // Elastic constraints
            for (common::Buffer<GpuIndex> const& partition : mPartitions)
            {
                e = thrust::async::for_each(
                    thrust::device.after(e),
                    partition.Data(),
                    partition.Data() + partition.Size(),
                    FProjectConstraint{
                        partition.Raw(),
                        mV.x.Raw(),
                        lambda.Raw(),
                        mT.inds.Raw(),
                        m.Raw(),
                        alpha.Raw(),
                        DmInv.Raw()});
            }
            // TODO: Collision constraints
            // ...
        }
        // Update simulation state
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            FFinalizeSolution{xt.Raw(), mV.x.Raw(), v.Raw(), sdt});
        e.wait();
    }
}

std::size_t XpbdImpl::NumberOfParticles() const
{
    return mV.x.Size();
}

std::size_t XpbdImpl::NumberOfConstraints() const
{
    return mT.inds.Size();
}

} // namespace xpbd
} // namespace gpu
} // namespace pbat