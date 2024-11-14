// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "IntegratorImpl.cuh"
#include "pbat/gpu/common/Eigen.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/sim/xpbd/Kernels.h"

#include <thrust/async/for_each.h>
#include <thrust/copy.h>
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
      Vquery(data.V.size(), nMaxVertexTetrahedronOverlaps, nMaxVertexTriangleContacts),
      mPositions(data.x.cols()),
      mPositionBuffer(nMaxVertexTriangleContacts),
      mVelocities(data.v.cols()),
      mExternalAcceleration(data.aext.cols()),
      mMassInverses(data.minv.size()),
      mLame(data.lame.size()),
      mShapeMatrixInverses(data.DmInv.size()),
      mRestStableGamma(data.gammaSNH.size()),
      mLagrangeMultipliers(),
      mCompliance(),
      mDamping(),
      mPptr(data.Pptr),
      mPadj(data.Padj.size()),
      mSGptr(data.SGptr),
      mSGadj(data.SGadj.size()),
      mCptr(data.Cptr.size()),
      mCadj(data.Cadj.size()),
      mPenalty(data.muV.size()),
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
    mDamping[snhConstraintId].Resize(data.beta[snhConstraintId].size());
    common::ToBuffer(data.lambda[snhConstraintId], mLagrangeMultipliers[snhConstraintId]);
    common::ToBuffer(data.alpha[snhConstraintId], mCompliance[snhConstraintId]);
    common::ToBuffer(data.beta[snhConstraintId], mDamping[snhConstraintId]);

    int const collisionConstraintId = static_cast<int>(EConstraint::Collision);
    mLagrangeMultipliers[collisionConstraintId].Resize(data.lambda[collisionConstraintId].size());
    mCompliance[collisionConstraintId].Resize(data.alpha[collisionConstraintId].size());
    mDamping[collisionConstraintId].Resize(data.beta[collisionConstraintId].size());
    common::ToBuffer(
        data.lambda[collisionConstraintId],
        mLagrangeMultipliers[collisionConstraintId]);
    common::ToBuffer(data.alpha[collisionConstraintId], mCompliance[collisionConstraintId]);
    common::ToBuffer(data.beta[collisionConstraintId], mDamping[collisionConstraintId]);
    // Setup partitions
    common::ToBuffer(pbat::common::ToEigen(data.Padj).cast<GpuIndex>().eval(), mPadj);
    bool const bHasClusteredPartitions = not mSGptr.empty();
    if (bHasClusteredPartitions)
    {
        common::ToBuffer(pbat::common::ToEigen(data.SGadj).cast<GpuIndex>().eval(), mSGadj);
        common::ToBuffer(pbat::common::ToEigen(data.Cptr).cast<GpuIndex>().eval(), mCptr);
        common::ToBuffer(pbat::common::ToEigen(data.Cadj).cast<GpuIndex>().eval(), mCadj);
    }
    // Copy collision data
    common::ToBuffer(data.muV, mPenalty);
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
        // Store previous positions
        mPositions = nextPositions;
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
            // Collision constraints
            auto const collisionConstraintId = static_cast<int>(EConstraint::Collision);
            e                                = thrust::async::for_each(
                thrust::device.after(e),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nContacts),
                [x      = nextPositions.Raw(),
                 xb     = mPositionBuffer.Raw(),
                 xt     = mPositions.Raw(),
                 lambda = mLagrangeMultipliers[collisionConstraintId].Raw(),
                 alpha  = mCompliance[collisionConstraintId].Raw(),
                 beta   = mDamping[collisionConstraintId].Raw(),
                 pairs  = contacts.Raw(),
                 CV     = V.inds.Raw(),
                 CF     = F.inds.Raw(),
                 minv   = mMassInverses.Raw(),
                 dt     = sdt,
                 dt2    = sdt2,
                 muC    = mPenalty.Raw(),
                 muS    = mStaticFrictionCoefficient,
                 muD    = mDynamicFrictionCoefficient] PBAT_DEVICE(GpuIndex c) {
                    using pbat::sim::xpbd::kernels::ProjectVertexTriangle;
                    using namespace pbat::math::linalg::mini;
                    auto sv = pairs[c].first;
                    auto v  = CV[0][sv];
                    SVector<GpuIndex, 3> f{
                        CF[0][pairs[c].second],
                        CF[1][pairs[c].second],
                        CF[2][pairs[c].second]};
                    GpuScalar minvv              = minv[v];
                    SVector<GpuScalar, 3> xvt    = FromBuffers<3, 1>(xt, v);
                    SVector<GpuScalar, 3> xv     = FromBuffers<3, 1>(x, v);
                    SMatrix<GpuScalar, 3, 3> xft = FromBuffers(xt, f.Transpose());
                    SMatrix<GpuScalar, 3, 3> xf  = FromBuffers(x, f.Transpose());
                    GpuScalar atildec            = alpha[c] / dt2;
                    GpuScalar gammac             = atildec * beta[c] * dt;
                    GpuScalar lambdac            = lambda[c];
                    GpuScalar muc                = muC[sv];
                    bool const bProject          = ProjectVertexTriangle(
                        minvv,
                        xvt,
                        xft,
                        xf,
                        muc,
                        muS,
                        muD,
                        atildec,
                        gammac,
                        lambdac,
                        xv);
                    if (bProject)
                    {
                        lambda[c] = lambdac;
                    }
                    ToBuffers(xv, xb, c);
                });
            e = thrust::async::for_each(
                thrust::device.after(e),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nContacts),
                [x     = nextPositions.Raw(),
                 xb    = mPositionBuffer.Raw(),
                 pairs = contacts.Raw(),
                 CV    = V.inds.Raw()] PBAT_DEVICE(GpuIndex c) {
                    using namespace pbat::math::linalg::mini;
                    auto v                   = CV[0][pairs[c].first];
                    SVector<GpuScalar, 3> xv = FromBuffers<3, 1>(xb, c);
                    ToBuffers(xv, x, v);
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

common::Buffer<GpuScalar, 3> const& IntegratorImpl::GetExternalAcceleration() const
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

void IntegratorImpl::ProjectBlockNeoHookeanConstraints(
    thrust::device_event& e,
    Scalar dt,
    Scalar dt2)
{
    auto const snhConstraintId = static_cast<int>(EConstraint::StableNeoHookean);
    auto const nPartitions     = static_cast<Index>(mPptr.size()) - 1;
    for (auto p = 0; p < nPartitions; ++p)
    {
        auto pbegin = mPptr[p];
        auto pend   = mPptr[p + 1];
        e           = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator(pbegin),
            thrust::make_counting_iterator(pend),
            [partition = mPadj.Raw(),
             x         = X.x.Raw(),
             xt        = mPositions.Raw(),
             lambda    = mLagrangeMultipliers[snhConstraintId].Raw(),
             T         = T.inds.Raw(),
             minv      = mMassInverses.Raw(),
             alpha     = mCompliance[snhConstraintId].Raw(),
             beta      = mDamping[snhConstraintId].Raw(),
             DmInv     = mShapeMatrixInverses.Raw(),
             gammaSNH  = mRestStableGamma.Raw(),
             dt,
             dt2] PBAT_DEVICE(Index k) {
                using pbat::sim::xpbd::kernels::ProjectBlockNeoHookean;
                using namespace pbat::math::linalg::mini;
                GpuIndex c                    = partition[k];
                SVector<GpuIndex, 4> Tc       = FromBuffers<4, 1>(T, c);
                SVector<GpuScalar, 4> minvc   = FromFlatBuffer(minv, Tc);
                SVector<GpuScalar, 2> atildec = FromFlatBuffer<2, 1>(alpha, c) / dt2;
                SVector<GpuScalar, 2> betac   = FromFlatBuffer<2, 1>(beta, c);
                SVector<GpuScalar, 2> gammac{
                    atildec(0) * betac(0) * dt,
                    atildec(1) * betac(1) * dt};
                SMatrix<GpuScalar, 3, 3> DmInvc = FromFlatBuffer<3, 3>(DmInv, c);
                SVector<GpuScalar, 2> lambdac   = FromFlatBuffer<2, 1>(lambda, c);
                SMatrix<GpuScalar, 3, 4> xtc    = FromBuffers(xt, Tc.Transpose());
                SMatrix<GpuScalar, 3, 4> xc     = FromBuffers(x, Tc.Transpose());
                ProjectBlockNeoHookean(
                    minvc,
                    DmInvc,
                    gammaSNH[c],
                    atildec,
                    gammac,
                    xtc,
                    lambdac,
                    xc);
                ToFlatBuffer(lambdac, lambda, c);
                ToBuffers(xc, Tc.Transpose(), x);
            });
    }
}

void IntegratorImpl::ProjectClusteredBlockNeoHookeanConstraints(Scalar dt, Scalar dt2) {}

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/physics/HyperElasticity.h"

#include <doctest/doctest.h>

#pragma nv_diag_suppress 177

TEST_CASE("[gpu][xpbd] Integrator") {}