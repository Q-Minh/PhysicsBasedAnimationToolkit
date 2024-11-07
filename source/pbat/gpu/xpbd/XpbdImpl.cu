// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "XpbdImpl.cuh"
#include "XpbdImplKernels.cuh"
#include "pbat/gpu/common/Eigen.cuh"

#include <array>
#include <exception>
#include <sstream>
#include <string>
#include <thrust/async/for_each.h>
#include <thrust/async/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace xpbd {

XpbdImpl::XpbdImpl(
    Eigen::Ref<GpuMatrixX const> const& Xin,
    Eigen::Ref<GpuIndexMatrixX const> const& Vin,
    Eigen::Ref<GpuIndexMatrixX const> const& Fin,
    Eigen::Ref<GpuIndexMatrixX const> const& Tin,
    Eigen::Ref<GpuIndexVectorX const> const& BVin,
    Eigen::Ref<GpuIndexVectorX const> const& BFin,
    std::size_t nMaxVertexTetrahedronOverlaps,
    std::size_t nMaxVertexTriangleContacts)
    : X(Xin),
      V(Vin),
      F(Fin),
      T(Tin),
      BV(BVin),
      BF(BFin),
      Tbvh(Tin.cols(), 0),
      Fbvh(Fin.cols(), 0),
      Vquery(Vin.cols(), nMaxVertexTetrahedronOverlaps, nMaxVertexTriangleContacts),
      mPositions(Xin.cols()),
      mVelocities(Xin.cols()),
      mExternalForces(Xin.cols()),
      mMassInverses(Xin.cols()),
      mLame(2 * Tin.cols()),
      mShapeMatrixInverses(9 * Tin.cols()),
      mRestStableGamma(Tin.cols()),
      mLagrangeMultipliers(),
      mCompliance(),
      mPartitions(),
      mStaticFrictionCoefficient{GpuScalar{0.5}},
      mDynamicFrictionCoefficient{GpuScalar{0.3}}
{
    mLagrangeMultipliers[StableNeoHookean].Resize(2 * T.NumberOfSimplices());
    mCompliance[StableNeoHookean].Resize(2 * T.NumberOfSimplices());
    mLagrangeMultipliers[Collision].Resize(V.NumberOfSimplices());
    mCompliance[Collision].Resize(V.NumberOfSimplices());
    // Initialize particle data
    mPositions = X.x;
    mVelocities.SetConstant(GpuScalar(0));
    mExternalForces.SetConstant(GpuScalar(0));
    mMassInverses.SetConstant(GpuScalar(1e-3));
}

void XpbdImpl::PrepareConstraints()
{
    mCompliance[Collision].SetConstant(GpuScalar(0));
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<GpuIndex>(0),
        thrust::make_counting_iterator<GpuIndex>(T.NumberOfSimplices()),
        XpbdImplKernels::FInitializeNeoHookeanConstraint{
            X.x.Raw(),
            T.inds.Raw(),
            mLame.Raw(),
            mShapeMatrixInverses.Raw(),
            mCompliance[StableNeoHookean].Raw(),
            mRestStableGamma.Raw()});
}

void XpbdImpl::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
{
    GpuScalar const sdt       = dt / static_cast<GpuScalar>(substeps);
    GpuScalar const sdt2      = sdt * sdt;
    GpuIndex const nParticles = static_cast<GpuIndex>(NumberOfParticles());
    // Detect collision candidates and setup collision constraint solve
    GpuScalar constexpr expansion{0};
    Tbvh.Build(X, T, Smin, Smax, expansion);
    Fbvh.Build(X, F, Smin, Smax, expansion);
    Vquery.Build(X, V, Smin, Smax, expansion);
    Vquery.DetectOverlaps(X, V, T, Tbvh);
    Vquery.DetectContactPairsFromOverlaps(X, V, F, BV, BF, Fbvh);
    common::SynchronizedList<ContactPairType>& contacts = Vquery.neighbours;
    GpuIndex const nContacts                            = contacts.Size();

    auto& nextPositions = X.x;
    for (auto s = 0; s < substeps; ++s)
    {
        // Reset "Lagrange" multipliers
        for (auto d = 0; d < kConstraintTypes; ++d)
        {
            thrust::fill(
                thrust::device,
                mLagrangeMultipliers[d].Data(),
                mLagrangeMultipliers[d].Data() + mLagrangeMultipliers[d].Size(),
                GpuScalar{0.});
        }
        // Initialize constraint solve
        thrust::device_event e = thrust::async::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            XpbdImplKernels::FInitializeSolution{
                mPositions.Raw(),
                nextPositions.Raw(),
                mVelocities.Raw(),
                mExternalForces.Raw(),
                mMassInverses.Raw(),
                sdt,
                sdt2});
        // Solve constraints
        for (auto k = 0; k < iterations; ++k)
        {
            // Elastic constraints
            for (common::Buffer<GpuIndex> const& partition : mPartitions)
            {
                e = thrust::async::for_each(
                    thrust::device.after(e),
                    partition.Data(),
                    partition.Data() + partition.Size(),
                    XpbdImplKernels::FStableNeoHookeanConstraint{
                        nextPositions.Raw(),
                        mLagrangeMultipliers[StableNeoHookean].Raw(),
                        T.inds.Raw(),
                        mMassInverses.Raw(),
                        mCompliance[StableNeoHookean].Raw(),
                        mShapeMatrixInverses.Raw(),
                        mRestStableGamma.Raw(),
                        sdt2});
            }
            // Collision constraints
            e = thrust::async::for_each(
                thrust::device.after(e),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nContacts),
                XpbdImplKernels::FVertexTriangleContactConstraint{
                    nextPositions.Raw(),
                    mLagrangeMultipliers[Collision].Raw(),
                    contacts.Raw(),
                    V.inds.Raw(),
                    F.inds.Raw(),
                    mMassInverses.Raw(),
                    mCompliance[Collision].Raw(),
                    mPositions.Raw(),
                    sdt2,
                    mStaticFrictionCoefficient,
                    mDynamicFrictionCoefficient});
        }
        // Update simulation state
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            XpbdImplKernels::FUpdateSolution{
                mPositions.Raw(),
                nextPositions.Raw(),
                mVelocities.Raw(),
                sdt});
        e.wait();
    }
}

std::size_t XpbdImpl::NumberOfParticles() const
{
    return X.x.Size();
}

std::size_t XpbdImpl::NumberOfConstraints() const
{
    return T.inds.Size();
}

void XpbdImpl::SetPositions(Eigen::Ref<GpuMatrixX const> const& Xin)
{
    common::ToBuffer(Xin, X.x);
    mPositions = X.x;
}

void XpbdImpl::SetVelocities(Eigen::Ref<GpuMatrixX const> const& vIn)
{
    common::ToBuffer(vIn, mVelocities);
}

void XpbdImpl::SetExternalForces(Eigen::Ref<GpuMatrixX const> const& fIn)
{
    common::ToBuffer(fIn, mExternalForces);
}

void XpbdImpl::SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv)
{
    common::ToBuffer(minv, mMassInverses);
}

void XpbdImpl::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    common::ToBuffer(l, mLame);
}

void XpbdImpl::SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint)
{
    common::ToBuffer(alpha, mCompliance[eConstraint]);
}

void XpbdImpl::SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mPartitions.resize(partitions.size());
    for (auto p = 0; p < partitions.size(); ++p)
    {
        mPartitions[p][0].resize(partitions[p].size());
        thrust::copy(partitions[p].begin(), partitions[p].end(), mPartitions[p].Data());
    }
}

void XpbdImpl::SetFrictionCoefficients(GpuScalar muS, GpuScalar muK)
{
    mStaticFrictionCoefficient  = muS;
    mDynamicFrictionCoefficient = muK;
}

void XpbdImpl::SetSceneBoundingBox(
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    Smin = min;
    Smax = max;
}

common::Buffer<GpuScalar, 3> const& XpbdImpl::GetVelocity() const
{
    return mVelocities;
}

common::Buffer<GpuScalar, 3> const& XpbdImpl::GetExternalForce() const
{
    return mExternalForces;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetMassInverse() const
{
    return mMassInverses;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetLameCoefficients() const
{
    return mLame;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetShapeMatrixInverse() const
{
    return mShapeMatrixInverses;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetRestStableGamma() const
{
    return mRestStableGamma;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    return mLagrangeMultipliers[eConstraint];
}

common::Buffer<GpuScalar> const& XpbdImpl::GetCompliance(EConstraint eConstraint) const
{
    return mCompliance[eConstraint];
}

std::vector<common::Buffer<GpuIndex>> const& XpbdImpl::GetPartitions() const
{
    return mPartitions;
}

std::vector<typename XpbdImpl::CollisionCandidateType>
XpbdImpl::GetVertexTetrahedronCollisionCandidates() const
{
    return Vquery.overlaps.Get();
}

std::vector<typename XpbdImpl::ContactPairType> XpbdImpl::GetVertexTriangleContactPairs() const
{
    return Vquery.neighbours.Get();
}

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/physics/HyperElasticity.h"

#include <doctest/doctest.h>

#pragma nv_diag_suppress 177

TEST_CASE("[gpu][xpbd] Xpbd")
{
    using namespace pbat;
    // Arrange
    GpuMatrixX X(3, 4);
    GpuIndexMatrixX V{
        GpuIndexVectorX::LinSpaced(X.cols(), GpuIndex{0}, static_cast<GpuIndex>(X.cols()) - 1)};
    V.transposeInPlace();
    GpuIndexMatrixX F(3, 4);
    GpuIndexMatrixX T(4, 1);
    GpuMatrixX lame(2, 1);
    auto constexpr Y        = 1e6;
    auto constexpr nu       = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    lame(0, 0)              = static_cast<GpuScalar>(mu);
    lame(1, 0)              = static_cast<GpuScalar>(lambda);
    // Unit tetrahedron
    // clang-format off
    X << 0.f, 1.f, 0.f, 0.f,
         0.f, 0.f, 1.f, 0.f,
         0.f, 0.f, 0.f, 1.f;
    F << 0, 1, 2, 0,
         1, 2, 0, 2,
         3, 3, 3, 1;
    T << 0, 
         1, 
         2, 
         3;
    // clang-format on
    GpuIndexVectorX BV                    = GpuIndexVectorX::Zero(V.cols());
    GpuIndexVectorX BF                    = GpuIndexVectorX::Zero(F.cols());
    GpuScalar constexpr tetVolumeExpected = GpuScalar{1.} / GpuScalar{6.};
    GpuMatrixX alphaExpected(2, 1);
    alphaExpected(0, 0)           = GpuScalar{1.} / (tetVolumeExpected * lame(0, 0));
    alphaExpected(1, 0)           = GpuScalar{1.} / (tetVolumeExpected * lame(1, 0));
    GpuScalar const gammaExpected = GpuScalar{1.} + lame(0, 0) / lame(1, 0);
    GpuScalar constexpr zero      = 1e-10f;

    // Act
    using pbat::gpu::xpbd::XpbdImpl;
    auto const nMaxOverlaps = static_cast<std::size_t>(10 * V.cols());
    auto const nMaxContacts = 8 * nMaxOverlaps;
    XpbdImpl xpbd{X, V, F, T, BV, BF, nMaxOverlaps, nMaxContacts};
    xpbd.SetLameCoefficients(lame);
    xpbd.PrepareConstraints();
    // Assert
    auto const& alphaGpu = xpbd.GetCompliance(XpbdImpl::EConstraint::StableNeoHookean);
    CHECK_EQ(alphaGpu.Size(), 2);
    GpuMatrixX alpha = common::ToEigen(alphaGpu.Get()).reshaped(2, alphaGpu.Size() / 2);
    CHECK(alpha.isApprox(alphaExpected, zero));
    auto const& DmInvGpu = xpbd.GetShapeMatrixInverse();
    CHECK_EQ(DmInvGpu.Size(), 9);
    GpuMatrixX DmInv = common::ToEigen(DmInvGpu.Get()).reshaped(3, DmInvGpu.Size() / 3);
    CHECK(DmInv.isApprox(GpuMatrixX::Identity(3, 3), zero));
    auto const& gammaGpu = xpbd.GetRestStableGamma();
    CHECK_EQ(gammaGpu.Size(), 1);
    GpuVectorX gamma = common::ToEigen(gammaGpu.Get());
    CHECK_LE(std::abs(gamma(0) - gammaExpected), zero);
}