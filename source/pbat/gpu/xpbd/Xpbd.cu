// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Xpbd.h"
#include "XpbdImpl.cuh"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/Eigen.cuh"

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
    return common::ToEigen(mImpl->X.x);
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
    return common::ToEigen(mImpl->GetVelocity());
}

GpuMatrixX Xpbd::GetExternalForce() const
{
    return common::ToEigen(mImpl->GetExternalForce());
}

GpuVectorX Xpbd::GetMassInverse() const
{
    return common::ToEigen(mImpl->GetMassInverse());
}

GpuMatrixX Xpbd::GetLameCoefficients() const
{
    auto lame = common::ToEigen(mImpl->GetLameCoefficients());
    return lame.reshaped(2, lame.size() / 2);
}

GpuMatrixX Xpbd::GetShapeMatrixInverse() const
{
    auto DmInv = common::ToEigen(mImpl->GetShapeMatrixInverse());
    return DmInv.reshaped(3, DmInv.size() / 3);
}

GpuMatrixX Xpbd::GetRestStableGamma() const
{
    auto gamma = common::ToEigen(mImpl->GetRestStableGamma());
    return gamma.reshaped(2, gamma.size() / 2);
}

GpuMatrixX Xpbd::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    auto lambda = common::ToEigen(
        mImpl->GetLagrangeMultiplier(static_cast<XpbdImpl::EConstraint>(eConstraint)));
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        lambda.resize(2, lambda.size() / 2);
    }
    return lambda;
}

GpuMatrixX Xpbd::GetCompliance(EConstraint eConstraint) const
{
    auto compliance =
        common::ToEigen(mImpl->GetCompliance(static_cast<XpbdImpl::EConstraint>(eConstraint)));
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        compliance.resize(2, compliance.size() / 2);
    }
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