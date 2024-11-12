// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "IntegratorImpl.cuh"
#include "pbat/gpu/common/Eigen.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/sim/xpbd/Kernels.h"

#include <thrust/async/for_each.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace xpbd {

IntegratorImpl::IntegratorImpl(
    Data const& data,
    std::size_t nMaxVertexTetrahedronOverlaps,
    std::size_t nMaxVertexTriangleContacts)
    : X(data.x.cast<GpuScalar>()),
      V(data.V.cast<GpuIndex>().transpose()),
      F(data.F.cast<GpuIndex>()),
      T(data.T.cast<GpuIndex>()),
      BV(data.BV.cast<GpuIndex>()),
      Tbvh(data.T.cols(), 0),
      Fbvh(data.F.cols(), 0),
      Vquery(data.V.cols(), nMaxVertexTetrahedronOverlaps, nMaxVertexTriangleContacts),
      mPositions(data.x.cols()),
      mVelocities(data.v.cols()),
      mExternalAcceleration(data.aext.cols()),
      mMassInverses(data.minv.size()),
      mLame(data.lame.size()),
      mShapeMatrixInverses(data.DmInv.size()),
      mRestStableGamma(data.gammaSNH.size()),
      mLagrangeMultipliers(),
      mCompliance(),
      mPartitions(),
      mStaticFrictionCoefficient{static_cast<GpuScalar>(data.muS)},
      mDynamicFrictionCoefficient{static_cast<GpuScalar>(data.muD)}
{
    // Initialize particle data
    common::ToBuffer(data.x, mPositions);
    common::ToBuffer(data.v, mVelocities);
    common::ToBuffer(data.aext, mExternalAcceleration);
    common::ToBuffer(data.minv, mMassInverses);
    common::ToBuffer(data.lame, mLame);
    common::ToBuffer(data.DmInv, mShapeMatrixInverses);
    common::ToBuffer(data.gammaSNH, mRestStableGamma);
    // Setup constraints
    int const snhConstraintId = static_cast<int>(EConstraint::StableNeoHookean);
    mLagrangeMultipliers[snhConstraintId].Resize(data.lambda[snhConstraintId].size());
    mCompliance[snhConstraintId].Resize(data.alpha[snhConstraintId].size());
    common::ToBuffer(data.lambda[snhConstraintId], mLagrangeMultipliers[snhConstraintId]);
    common::ToBuffer(data.alpha[snhConstraintId], mCompliance[snhConstraintId]);

    int const collisionConstraintId = static_cast<int>(EConstraint::Collision);
    mLagrangeMultipliers[collisionConstraintId].Resize(data.lambda[collisionConstraintId].size());
    mCompliance[collisionConstraintId].Resize(data.alpha[collisionConstraintId].size());
    common::ToBuffer(
        data.lambda[collisionConstraintId],
        mLagrangeMultipliers[collisionConstraintId]);
    common::ToBuffer(data.alpha[collisionConstraintId], mCompliance[collisionConstraintId]);
    // Setup partitions
    mPartitions.resize(data.partitions.size());
    for (auto p = 0; p < data.partitions.size(); ++p)
    {
        mPartitions[p].Resize(data.partitions[p].size());
        thrust::copy(data.partitions[p].begin(), data.partitions[p].end(), mPartitions[p].Data());
    }
}

void IntegratorImpl::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
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
    Vquery.DetectContactPairsFromOverlaps(X, V, F, BV, Fbvh);
    common::SynchronizedList<ContactPairType>& contacts = Vquery.neighbours;
    GpuIndex const nContacts                            = contacts.Size();

    auto& nextPositions = X.x;
    for (auto s = 0; s < substeps; ++s)
    {
        // Reset "Lagrange" multipliers
        for (auto d = 0; d < kConstraintTypes; ++d)
        {
            mLagrangeMultipliers[d].SetConstant(GpuScalar(0));
        }
        // Initialize constraint solve
        thrust::device_event e = thrust::async::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            [xt   = mPositions.Raw(),
             x    = nextPositions.Raw(),
             vt   = mVelocities.Raw(),
             aext = mExternalAcceleration.Raw(),
             dt   = sdt,
             dt2  = sdt2] PBAT_DEVICE(GpuIndex i) {
                using pbat::sim::xpbd::kernels::InitialPosition;
                using namespace pbat::math::linalg::mini;
                auto xi = InitialPosition(
                    FromBuffers<3, 1>(xt, i),
                    FromBuffers<3, 1>(vt, i),
                    FromBuffers<3, 1>(aext, i),
                    dt,
                    dt2);
                ToBuffers(xi, x, i);
            });
        // Solve constraints
        for (auto k = 0; k < iterations; ++k)
        {
            // Elastic constraints
            auto const snhConstraintId = static_cast<int>(EConstraint::StableNeoHookean);
            for (common::Buffer<GpuIndex> const& partition : mPartitions)
            {
                e = thrust::async::for_each(
                    thrust::device.after(e),
                    partition.Data(),
                    partition.Data() + partition.Size(),
                    [x      = nextPositions.Raw(),
                     lambda = mLagrangeMultipliers[snhConstraintId].Raw(),
                     T      = T.inds.Raw(),
                     minv   = mMassInverses.Raw(),
                     alpha  = mCompliance[snhConstraintId].Raw(),
                     DmInv  = mShapeMatrixInverses.Raw(),
                     gamma  = mRestStableGamma.Raw(),
                     dt2    = sdt2] PBAT_DEVICE(GpuIndex c) {
                        using pbat::sim::xpbd::kernels::ProjectDeviatoric;
                        using pbat::sim::xpbd::kernels::ProjectHydrostatic;
                        using namespace pbat::math::linalg::mini;
                        SVector<GpuIndex, 4> Tc         = FromBuffers<4, 1>(T, c);
                        SVector<GpuScalar, 4> minvc     = FromFlatBuffer(minv, Tc);
                        SVector<GpuScalar, 2> atildec   = FromFlatBuffer<2, 1>(alpha, c) / dt2;
                        SMatrix<GpuScalar, 3, 3> DmInvc = FromFlatBuffer<3, 3>(DmInv, c);
                        SVector<GpuScalar, 2> lambdac   = FromFlatBuffer<2, 1>(lambda, c);
                        SMatrix<GpuScalar, 3, 4> xc     = FromBuffers(x, Tc.Transpose());
                        ProjectDeviatoric(c, minvc, atildec(0), DmInvc, lambdac(0), xc);
                        ProjectHydrostatic(c, minvc, atildec(1), gamma[c], DmInvc, lambdac(1), xc);
                        ToFlatBuffer(lambdac, lambda, c);
                        ToBuffers(xc, Tc.Transpose(), x);
                    });
            }
            // Collision constraints
            auto const collisionConstraintId = static_cast<int>(EConstraint::Collision);
            e                                = thrust::async::for_each(
                thrust::device.after(e),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nContacts),
                [x      = nextPositions.Raw(),
                 xt     = mPositions.Raw(),
                 lambda = mLagrangeMultipliers[collisionConstraintId].Raw(),
                 alpha  = mCompliance[collisionConstraintId].Raw(),
                 pairs  = contacts.Raw(),
                 CV     = V.inds.Raw(),
                 CF     = F.inds.Raw(),
                 minv   = mMassInverses.Raw(),
                 dt2    = sdt2,
                 muS    = mStaticFrictionCoefficient,
                 muD    = mDynamicFrictionCoefficient] PBAT_DEVICE(GpuIndex c) {
                    using pbat::sim::xpbd::kernels::ProjectVertexTriangle;
                    using namespace pbat::math::linalg::mini;
                    auto v = CV[0][pairs[c].first];
                    SVector<GpuIndex, 3> f{
                        CF[0][pairs[c].second],
                        CF[1][pairs[c].second],
                        CF[2][pairs[c].second]};
                    GpuScalar minvv              = minv[v];
                    SVector<GpuScalar, 3> minvf  = FromFlatBuffer(minv, f);
                    SVector<GpuScalar, 3> xvt    = FromBuffers<3, 1>(xt, v);
                    SVector<GpuScalar, 3> xv     = FromBuffers<3, 1>(x, v);
                    SMatrix<GpuScalar, 3, 3> xft = FromBuffers(xt, f.Transpose());
                    SMatrix<GpuScalar, 3, 3> xf  = FromBuffers(x, f.Transpose());
                    GpuScalar atildec            = alpha[c] / dt2;
                    GpuScalar lambdac            = lambda[c];
                    bool bProject                = ProjectVertexTriangle(
                        minvv,
                        minvf,
                        xvt,
                        xft,
                        xf,
                        muS,
                        muD,
                        atildec,
                        lambdac,
                        xv);
                    if (bProject)
                    {
                        lambda[c] = lambdac;
                        ToBuffers(xv, x, v);
                    }
                });
        }
        // Update simulation state
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            [xt = mPositions.Raw(),
             x  = nextPositions.Raw(),
             v  = mVelocities.Raw(),
             dt = sdt] PBAT_DEVICE(GpuIndex i) {
                using pbat::sim::xpbd::kernels::IntegrateVelocity;
                using namespace pbat::math::linalg::mini;
                auto vi = IntegrateVelocity(FromBuffers<3, 1>(xt, i), FromBuffers<3, 1>(x, i), dt);
                ToBuffers(vi, v, i);
            });
        e.wait();
    }
}

std::size_t IntegratorImpl::NumberOfParticles() const
{
    return X.x.Size();
}

std::size_t IntegratorImpl::NumberOfConstraints() const
{
    return T.inds.Size();
}

void IntegratorImpl::SetPositions(Eigen::Ref<GpuMatrixX const> const& Xin)
{
    common::ToBuffer(Xin, X.x);
    mPositions = X.x;
}

void IntegratorImpl::SetVelocities(Eigen::Ref<GpuMatrixX const> const& vIn)
{
    common::ToBuffer(vIn, mVelocities);
}

void IntegratorImpl::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    common::ToBuffer(aext, mExternalAcceleration);
}

void IntegratorImpl::SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv)
{
    common::ToBuffer(minv, mMassInverses);
}

void IntegratorImpl::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    common::ToBuffer(l, mLame);
}

void IntegratorImpl::SetCompliance(
    Eigen::Ref<GpuMatrixX const> const& alpha,
    EConstraint eConstraint)
{
    common::ToBuffer(alpha, mCompliance[static_cast<int>(eConstraint)]);
}

void IntegratorImpl::SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mPartitions.resize(partitions.size());
    for (auto p = 0; p < partitions.size(); ++p)
    {
        mPartitions[p][0].resize(partitions[p].size());
        thrust::copy(partitions[p].begin(), partitions[p].end(), mPartitions[p].Data());
    }
}

void IntegratorImpl::SetFrictionCoefficients(GpuScalar muS, GpuScalar muK)
{
    mStaticFrictionCoefficient  = muS;
    mDynamicFrictionCoefficient = muK;
}

void IntegratorImpl::SetSceneBoundingBox(
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    Smin = min;
    Smax = max;
}

common::Buffer<GpuScalar, 3> const& IntegratorImpl::GetVelocity() const
{
    return mVelocities;
}

common::Buffer<GpuScalar, 3> const& IntegratorImpl::GetExternalForce() const
{
    return mExternalAcceleration;
}

common::Buffer<GpuScalar> const& IntegratorImpl::GetMassInverse() const
{
    return mMassInverses;
}

common::Buffer<GpuScalar> const& IntegratorImpl::GetLameCoefficients() const
{
    return mLame;
}

common::Buffer<GpuScalar> const& IntegratorImpl::GetShapeMatrixInverse() const
{
    return mShapeMatrixInverses;
}

common::Buffer<GpuScalar> const& IntegratorImpl::GetRestStableGamma() const
{
    return mRestStableGamma;
}

common::Buffer<GpuScalar> const&
IntegratorImpl::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    return mLagrangeMultipliers[static_cast<int>(eConstraint)];
}

common::Buffer<GpuScalar> const& IntegratorImpl::GetCompliance(EConstraint eConstraint) const
{
    return mCompliance[static_cast<int>(eConstraint)];
}

std::vector<common::Buffer<GpuIndex>> const& IntegratorImpl::GetPartitions() const
{
    return mPartitions;
}

std::vector<typename IntegratorImpl::CollisionCandidateType>
IntegratorImpl::GetVertexTetrahedronCollisionCandidates() const
{
    return Vquery.overlaps.Get();
}

std::vector<typename IntegratorImpl::ContactPairType>
IntegratorImpl::GetVertexTriangleContactPairs() const
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

TEST_CASE("[gpu][xpbd] Integrator") {}