// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VbdImpl.cuh"
#include "VbdImplKernels.cuh"

#include <cuda/api.hpp>
#include <thrust/async/copy.h>
#include <thrust/async/for_each.h>
#include <thrust/execution_policy.h>

namespace pbat {
namespace gpu {
namespace vbd {

VbdImpl::VbdImpl(
    Eigen::Ref<GpuMatrixX const> const& Xin,
    Eigen::Ref<GpuIndexMatrixX const> const& Vin,
    Eigen::Ref<GpuIndexMatrixX const> const& Fin,
    Eigen::Ref<GpuIndexMatrixX const> const& Tin)
    : X(Xin),
      V(Vin),
      F(Fin),
      T(Tin),
      mPositionsAtTMinus1(Xin.cols()),
      mPositionsAtT(Xin.cols()),
      mInertialTargetPositions(Xin.cols()),
      mVelocitiesAtT(Xin.cols()),
      mVelocities(Xin.cols()),
      mExternalAcceleration(Xin.cols()),
      mMass(Xin.cols()),
      mShapeFunctionGradients(Tin.cols() * 4 * 3),
      mLameCoefficients(Tin.cols()),
      mVertexTetrahedronNeighbours(),
      mVertexTetrahedronPrefix(Xin.cols() + 1),
      mVertexTetrahedronLocalVertexIndices(),
      mRayleighDamping(GpuScalar{0}),
      mCollisionPenalty(GpuScalar{1e3}),
      mMaxCollidingTrianglesPerVertex(8),
      mCollidingTriangles(8 * Xin.cols()),
      mCollidingTriangleCount(Xin.cols()),
      mPartitions()
{
    for (auto d = 0; d < X.Dimensions(); ++d)
    {
        thrust::copy(X.x[d].begin(), X.x[d].end(), mPositionsAtTMinus1[d].begin());
        thrust::copy(X.x[d].begin(), X.x[d].end(), mPositionsAtT[d].begin());
        thrust::fill(mVelocitiesAtT[d].begin(), mVelocitiesAtT[d].end(), GpuScalar{0});
        thrust::fill(mVelocities[d].begin(), mVelocities[d].end(), GpuScalar{0});
        thrust::fill(
            mExternalAcceleration[d].begin(),
            mExternalAcceleration[d].end(),
            GpuScalar{0});
    }
    thrust::fill(mMass.Data(), mMass.Data() + mMass.Size(), GpuScalar{1e3});
}

void VbdImpl::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps, GpuScalar rho)
{
    GpuScalar sdt            = dt / static_cast<GpuScalar>(substeps);
    GpuScalar sdt2           = sdt * sdt;
    GpuIndex const nVertices = static_cast<GpuIndex>(X.NumberOfPoints());

    kernels::BackwardEulerMinimization bdf{};
    bdf.dt                              = sdt;
    bdf.dt2                             = sdt2;
    bdf.m                               = mMass.Raw();
    bdf.xtilde                          = mInertialTargetPositions.Raw();
    bdf.xt                              = mPositionsAtT.Raw();
    bdf.x                               = X.x.Raw();
    bdf.T                               = T.inds.Raw();
    bdf.GP                              = mShapeFunctionGradients.Raw();
    bdf.lame                            = mLameCoefficients.Raw();
    bdf.GVTn                            = mVertexTetrahedronNeighbours.Raw();
    bdf.GVTp                            = mVertexTetrahedronPrefix.Raw();
    bdf.GVTilocal                       = mVertexTetrahedronLocalVertexIndices.Raw();
    bdf.kD                              = mRayleighDamping;
    bdf.kC                              = mCollisionPenalty;
    bdf.nMaxCollidingTrianglesPerVertex = mMaxCollidingTrianglesPerVertex;
    bdf.FC                              = mCollidingTriangles.Raw();
    bdf.nCollidingTriangles             = mCollidingTriangleCount.Raw();
    bdf.F                               = F.inds.Raw();

    for (auto s = 0; s < substeps; ++s)
    {
        // Compute inertial target positions
        thrust::device_event e = thrust::async::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nVertices),
            kernels::FInertialTarget{
                sdt,
                sdt2,
                X.x.Raw(),
                mVelocities.Raw(),
                mExternalAcceleration.Raw(),
                mInertialTargetPositions.Raw()});
        // Initialize block coordinate descent's, i.e. BCD's, solution
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nVertices),
            kernels::FAdaptiveInitialization{
                sdt,
                sdt2,
                X.x.Raw(),
                mVelocitiesAtT.Raw(),
                mVelocities.Raw(),
                mExternalAcceleration.Raw(),
                X.x.Raw()});
        // Prepare state history for Chebyshev acceleration, i.e.
        // x(t-2) <- x(t-1)
        // x(t-1) <- x(t)
        for (auto d = 0; d < X.x.Dimensions(); ++d)
        {
            e = thrust::async::copy(
                thrust::device.after(e),
                mPositionsAtT[d].begin(),
                mPositionsAtT[d].end(),
                mPositionsAtTMinus1[d].begin());
            e = thrust::async::copy(
                thrust::device.after(e),
                X.x[d].begin(),
                X.x[d].end(),
                mPositionsAtT[d].begin());
        }
        // Initialize Chebyshev semi-iterative method
        kernels::FChebyshev fChebyshev{
            rho,
            mPositionsAtTMinus1.Raw(),
            mPositionsAtT.Raw(),
            X.x.Raw()};
        // Share thrust's underlying CUDA stream with cuda-api-wrappers
        auto device                           = cuda::device::get(e.where());
        auto stream                           = device.default_stream();
        CUstream_st* thrustNativeStreamHandle = e.stream().get();
        CUstream_st* cawNativeStreamHandle    = stream.handle();
        assert(thrustNativeStreamHandle == cawNativeStreamHandle);
        auto bcdLaunchConfiguration = cuda::launch_config_builder()
                                          .grid_size(nVertices)
                                          .use_maximum_linear_block()
                                          .dynamic_shared_memory_size(1 << 16 /* 64kB */)
                                          .build();
        // Minimize Backward Euler, i.e. BDF1, objective
        for (auto k = 0; k < iterations; ++k)
        {
            for (auto& partition : mPartitions)
            {
                bdf.partition = partition.Raw();
                stream.enqueue.kernel_launch(
                    kernels::MinimizeBackwardEuler,
                    bcdLaunchConfiguration,
                    bdf);
            }
            e = thrust::async::for_each(
                thrust::device.on(stream.handle()),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nVertices),
                fChebyshev);
            fChebyshev.Update(k);
        }
        // Update velocities
        for (auto d = 0; d < mVelocities.Dimensions(); ++d)
        {
            e = thrust::async::copy(
                thrust::device.after(e),
                mVelocities[d].begin(),
                mVelocities[d].end(),
                mVelocitiesAtT[d].begin());
        }
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nVertices),
            kernels::FFinalizeSolution{sdt, mPositionsAtT.Raw(), X.x.Raw(), mVelocities.Raw()});

        e.wait();
    }
}

void VbdImpl::SetPositions(Eigen::Ref<GpuMatrixX const> const& Xin, bool bResetHistory)
{
    auto const nVertices = static_cast<GpuIndex>(X.x.Size());
    if (Xin.rows() != 3 and Xin.cols() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected positions of dimensions " << X.x.Dimensions() << "x" << X.x.Size()
           << ", but got " << Xin.rows() << "x" << Xin.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < X.x.Dimensions(); ++d)
    {
        thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), X.x[d].begin());
        if (bResetHistory)
        {
            thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), mPositionsAtTMinus1[d].begin());
            thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), mPositionsAtT[d].begin());
        }
    }
}

void VbdImpl::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v, bool bResetHistory)
{
    auto const nVertices = static_cast<GpuIndex>(mVelocities.Size());
    if (v.rows() != 3 and v.cols() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected velocities of dimensions " << mVelocities.Dimensions() << "x"
           << mVelocities.Size() << ", but got " << v.rows() << "x" << v.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mVelocities.Dimensions(); ++d)
    {
        thrust::copy(v.row(d).begin(), v.row(d).end(), mVelocities[d].begin());
        if (bResetHistory)
        {
            thrust::copy(v.row(d).begin(), v.row(d).end(), mVelocitiesAtT[d].begin());
        }
    }
}

void VbdImpl::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    auto const nVertices = static_cast<GpuIndex>(mExternalAcceleration.Size());
    if (aext.rows() != 3 and aext.cols() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected accelerations of dimensions " << mExternalAcceleration.Dimensions() << "x"
           << mExternalAcceleration.Size() << ", but got " << aext.rows() << "x" << aext.cols()
           << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mExternalAcceleration.Dimensions(); ++d)
        thrust::copy(aext.row(d).begin(), aext.row(d).end(), mExternalAcceleration[d].begin());
}

void VbdImpl::SetMass(Eigen::Ref<GpuVectorX const> const& m)
{
    auto const nVertices = static_cast<GpuIndex>(mMass.Size());
    if (m.size() != nVertices)
    {
        std::ostringstream ss{};
        ss << "Expected masses of dimensions " << nVertices << "x1 or its transpose, but got "
           << m.size() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(m.data(), m.data() + m.size(), mMass.Data());
}

void VbdImpl::SetShapeFunctionGradients(Eigen::Ref<GpuMatrixX const> const& GP)
{
    auto const nTetrahedra = static_cast<GpuIndex>(T.inds.Size());
    if (GP.rows() != 4 and GP.cols() != nTetrahedra * 3)
    {
        std::ostringstream ss{};
        ss << "Expected shape function gradients of dimensions 4x" << nTetrahedra * 3
           << ", but got " << GP.rows() << "x" << GP.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(GP.data(), GP.data() + GP.size(), mShapeFunctionGradients.Data());
}

void VbdImpl::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    auto const nTetrahedra = static_cast<GpuIndex>(T.inds.Size());
    if (l.rows() != 2 and l.cols() != nTetrahedra)
    {
        std::ostringstream ss{};
        ss << "Expected Lame coefficients of dimensions 2x" << nTetrahedra << ", but got "
           << l.rows() << "x" << l.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(l.data(), l.data() + l.size(), mLameCoefficients.Data());
}

void VbdImpl::SetVertexTetrahedronAdjacencyList(
    Eigen::Ref<GpuIndexVectorX const> const& GVTn,
    Eigen::Ref<GpuIndexVectorX const> const& GVTp,
    Eigen::Ref<GpuIndexVectorX const> const& GVTilocal)
{
    thrust::copy(GVTn.data(), GVTn.data() + GVTn.size(), mVertexTetrahedronNeighbours.Data());
    thrust::copy(GVTp.data(), GVTp.data() + GVTp.size(), mVertexTetrahedronPrefix.Data());
    thrust::copy(
        GVTilocal.data(),
        GVTilocal.data() + GVTilocal.size(),
        mVertexTetrahedronLocalVertexIndices.Data());
}

void VbdImpl::SetRayleighDampingCoefficient(GpuScalar kD)
{
    mRayleighDamping = kD;
}

void VbdImpl::SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mPartitions.resize(partitions.size());
    for (auto p = 0; p < partitions.size(); ++p)
    {
        mPartitions[p].Resize(partitions[p].size());
        thrust::copy(partitions[p].begin(), partitions[p].end(), mPartitions[p].Data());
    }
}

common::Buffer<GpuScalar, 3> const& VbdImpl::GetVelocity() const
{
    return mVelocities;
}

common::Buffer<GpuScalar, 3> const& VbdImpl::GetExternalAcceleration() const
{
    return mExternalAcceleration;
}

common::Buffer<GpuScalar> const& VbdImpl::GetMass() const
{
    return mMass;
}

common::Buffer<GpuScalar> const& VbdImpl::GetShapeFunctionGradients() const
{
    return mShapeFunctionGradients;
}

common::Buffer<GpuScalar> const& VbdImpl::GetLameCoefficients() const
{
    return mLameCoefficients;
}

std::vector<common::Buffer<GpuIndex>> const& VbdImpl::GetPartitions() const
{
    return mPartitions;
}

} // namespace vbd
} // namespace gpu
} // namespace pbat
