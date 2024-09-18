// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Xpbd.h"
#include "XpbdImpl.cuh"
#include "pbat/gpu/common/Buffer.cuh"

#include <thrust/copy.h>

namespace pbat {
namespace gpu {
namespace xpbd {

Xpbd::Xpbd(
    Eigen::Ref<GpuMatrixX const> const& X,
    Eigen::Ref<GpuIndexMatrixX const> const& V,
    Eigen::Ref<GpuIndexMatrixX const> const& F,
    Eigen::Ref<GpuIndexMatrixX const> const& T,
    Eigen::Ref<GpuIndexVectorX const> const& BV,
    Eigen::Ref<GpuIndexVectorX const> const& BF,
    std::size_t nMaxVertexTetrahedronOverlaps,
    std::size_t nMaxVertexTriangleContacts)
    : mImpl(new XpbdImpl{
          X,
          V,
          F,
          T,
          BV,
          BF,
          nMaxVertexTetrahedronOverlaps,
          nMaxVertexTriangleContacts})
{
}

Xpbd::Xpbd(Xpbd&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Xpbd& Xpbd::operator=(Xpbd&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

void Xpbd::PrepareConstraints()
{
    mImpl->PrepareConstraints();
}

void Xpbd::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
{
    mImpl->Step(dt, iterations, substeps);
}

GpuMatrixX Xpbd::Positions() const
{
    GpuMatrixX X(mImpl->X.Dimensions(), mImpl->X.NumberOfPoints());
    for (auto d = 0; d < X.rows(); ++d)
    {
        thrust::copy(mImpl->X.x[d].begin(), mImpl->X.x[d].end(), X.row(d).begin());
    }
    return X;
}

std::size_t Xpbd::NumberOfParticles() const
{
    return mImpl->NumberOfParticles();
}

std::size_t Xpbd::NumberOfConstraints() const
{
    return mImpl->NumberOfConstraints();
}

void Xpbd::SetPositions(Eigen::Ref<GpuMatrixX const> const& X)
{
    mImpl->SetPositions(X);
}

void Xpbd::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v)
{
    mImpl->SetVelocities(v);
}

void Xpbd::SetExternalForces(Eigen::Ref<GpuMatrixX const> const& f)
{
    mImpl->SetExternalForces(f);
}

void Xpbd::SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv)
{
    mImpl->SetMassInverse(minv);
}

void Xpbd::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    mImpl->SetLameCoefficients(l);
}

void Xpbd::SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint)
{
    mImpl->SetCompliance(alpha, static_cast<XpbdImpl::EConstraint>(eConstraint));
}

void Xpbd::SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mImpl->SetConstraintPartitions(partitions);
}

void Xpbd::SetFrictionCoefficients(GpuScalar muS, GpuScalar muK)
{
    mImpl->SetFrictionCoefficients(muS, muK);
}

void Xpbd::SetSceneBoundingBox(
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    mImpl->SetSceneBoundingBox(min, max);
}

GpuMatrixX Xpbd::GetVelocity() const
{
    auto const& vGpu = mImpl->GetVelocity();
    GpuMatrixX v(vGpu.Dimensions(), vGpu.Size());
    for (auto d = 0; d < vGpu.Dimensions(); ++d)
    {
        thrust::copy(vGpu[d].begin(), vGpu[d].end(), v.row(d).begin());
    }
    return v;
}

GpuMatrixX Xpbd::GetExternalForce() const
{
    auto const& fGpu = mImpl->GetExternalForce();
    GpuMatrixX f(fGpu.Dimensions(), fGpu.Size());
    for (auto d = 0; d < fGpu.Dimensions(); ++d)
    {
        thrust::copy(fGpu[d].begin(), fGpu[d].end(), f.row(d).begin());
    }
    return f;
}

GpuVectorX Xpbd::GetMassInverse() const
{
    auto const& minvGpu = mImpl->GetMassInverse();
    GpuVectorX minv(minvGpu.Size());
    thrust::copy(minvGpu.Data(), minvGpu.Data() + minvGpu.Size(), minv.begin());
    return minv;
}

GpuMatrixX Xpbd::GetLameCoefficients() const
{
    auto const& lameGpu = mImpl->GetLameCoefficients();
    GpuMatrixX lame(2, lameGpu.Size() / 2);
    thrust::copy(lameGpu.Data(), lameGpu.Data() + lameGpu.Size(), lame.data());
    return lame;
}

GpuMatrixX Xpbd::GetShapeMatrixInverse() const
{
    auto const& DmInvGpu = mImpl->GetShapeMatrixInverse();
    GpuMatrixX DmInv(3, DmInvGpu.Size() / 3);
    thrust::copy(DmInvGpu.Data(), DmInvGpu.Data() + DmInvGpu.Size(), DmInv.data());
    return DmInv;
}

GpuMatrixX Xpbd::GetRestStableGamma() const
{
    auto const& gammaGpu = mImpl->GetRestStableGamma();
    GpuMatrixX gamma(2, gammaGpu.Size() / 2);
    thrust::copy(gammaGpu.Data(), gammaGpu.Data() + gammaGpu.Size(), gamma.data());
    return gamma;
}

GpuMatrixX Xpbd::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    auto const& lambdaGpu =
        mImpl->GetLagrangeMultiplier(static_cast<XpbdImpl::EConstraint>(eConstraint));
    GpuMatrixX lambda{};
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        lambda.resize(2, lambdaGpu.Size() / 2);
    }
    if (eConstraint == EConstraint::Collision)
    {
        lambda.resize(1, lambdaGpu.Size());
    }
    thrust::copy(lambdaGpu.Data(), lambdaGpu.Data() + lambdaGpu.Size(), lambda.data());
    return lambda;
}

GpuMatrixX Xpbd::GetCompliance(EConstraint eConstraint) const
{
    auto const& complianceGpu =
        mImpl->GetCompliance(static_cast<XpbdImpl::EConstraint>(eConstraint));
    GpuMatrixX compliance{};
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        compliance.resize(2, complianceGpu.Size() / 2);
    }
    if (eConstraint == EConstraint::Collision)
    {
        compliance.resize(1, complianceGpu.Size());
    }
    thrust::copy(
        complianceGpu.Data(),
        complianceGpu.Data() + complianceGpu.Size(),
        compliance.data());
    return compliance;
}

std::vector<std::vector<GpuIndex>> Xpbd::GetPartitions() const
{
    auto const& partitionsGpu = mImpl->GetPartitions();
    std::vector<std::vector<GpuIndex>> partitions{};
    partitions.resize(partitionsGpu.size());
    for (auto p = 0; p < partitionsGpu.size(); ++p)
    {
        partitions[p].resize(partitionsGpu[p].Size());
        thrust::copy(
            partitionsGpu[p].Data(),
            partitionsGpu[p].Data() + partitionsGpu[p].Size(),
            partitions[p].begin());
    }
    return partitions;
}

GpuIndexMatrixX Xpbd::GetVertexTetrahedronCollisionCandidates() const
{
    std::vector<typename XpbdImpl::CollisionCandidateType> const overlaps =
        mImpl->GetVertexTetrahedronCollisionCandidates();
    GpuIndexMatrixX O(2, overlaps.size());
    for (auto o = 0; o < overlaps.size(); ++o)
    {
        O(0, o) = overlaps[o].first;
        O(1, o) = overlaps[o].second;
    }
    return O;
}

GpuIndexMatrixX Xpbd::GetVertexTriangleContactPairs() const
{
    std::vector<typename XpbdImpl::CollisionCandidateType> const contacts =
        mImpl->GetVertexTriangleContactPairs();
    GpuIndexMatrixX C(2, contacts.size());
    for (auto c = 0; c < contacts.size(); ++c)
    {
        C(0, c) = contacts[c].first;
        C(1, c) = contacts[c].second;
    }
    return C;
}

Xpbd::~Xpbd()
{
    if (mImpl != nullptr)
        delete mImpl;
}

} // namespace xpbd
} // namespace gpu
} // namespace pbat