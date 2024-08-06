#include "XpbdImpl.cuh"
#include "pbat/gpu/math/linalg/Matrix.cuh"

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
    __device__ void ProjectHydrostatic(
        GpuIndex c,
        math::linalg::Matrix<GpuScalar, 4, 1>& mc,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        Matrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 0) - Repeat<1, 3>(xc.Col(3))) * DmInvC;
        GpuScalar CH              = Determinant(F) - GpuScalar(1.);
        Matrix<GpuScalar, 3, 3> P{};
        P.Col(0) = Cross(F.Col(1), F.Col(2));
        P.Col(1) = Cross(F.Col(2), F.Col(0));
        P.Col(2) = Cross(F.Col(0), F.Col(1));
        Matrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 0) = P * DmInvC.Transpose();
        gradC.Col(3)            = -(gradC.Col(0) + gradC.Col(1) + gradC.Col(2));
        // TODO: Update lambda and project xc
    }

    __device__ void ProjectDeviatoric(
        GpuIndex c,
        math::linalg::Matrix<GpuScalar, 4, 1>& mc,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        Matrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 0) - Repeat<1, 3>(xc.Col(3))) * DmInvC;
        GpuScalar CD              = Norm(F);
        Matrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 0) = (F * DmInvC.Transpose()) /
                                  (CD + 1e-8 /*prevent numerical zero at fully collapsed state*/);
        gradC.Col(3) = -(gradC.Col(0) + gradC.Col(1) + gradC.Col(2));
        // TODO: Update lambda and project xc
    }

    __device__ void operator()(GpuIndex c)
    {
        using namespace pbat::gpu::math::linalg;

        GpuIndex const v[4] = {T[0][c], T[1][c], T[2][c], T[3][c]};
        Matrix<GpuScalar, 3, 4> xc{};
        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                xc(d, j) = x[d][v[j]];
        Matrix<GpuScalar, 4, 1> mc{};
        for (auto j = 0; j < 4; ++j)
            mc(j) = m[v[j]];

        ProjectHydrostatic(c, mc, xc);
        ProjectDeviatoric(c, mc, xc);

        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                x[d][v[j]] = xc(d, j);
    }

    std::array<GpuScalar*, 3> x;
    GpuScalar* lambda;

    std::array<GpuIndex*, 4> T;
    GpuScalar* m;
    std::array<GpuScalar*, 2> alpha;
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