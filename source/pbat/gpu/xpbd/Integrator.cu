// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Integrator.h"
#include "IntegratorImpl.cuh"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/Eigen.cuh"

namespace pbat {
namespace gpu {
namespace xpbd {

Integrator::Integrator(
    Data const& data,
    std::size_t nMaxVertexTetrahedronOverlaps,
    std::size_t nMaxVertexTriangleContacts)
    : mImpl(new IntegratorImpl{data, nMaxVertexTetrahedronOverlaps, nMaxVertexTriangleContacts})
{
}

Integrator::Integrator(Integrator&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Integrator& Integrator::operator=(Integrator&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

void Integrator::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
{
    mImpl->Step(dt, iterations, substeps);
}

GpuMatrixX Integrator::Positions() const
{
    return common::ToEigen(mImpl->X.x);
}

std::size_t Integrator::NumberOfParticles() const
{
    return mImpl->NumberOfParticles();
}

std::size_t Integrator::NumberOfConstraints() const
{
    return mImpl->NumberOfConstraints();
}

void Integrator::SetPositions(Eigen::Ref<GpuMatrixX const> const& X)
{
    mImpl->SetPositions(X);
}

void Integrator::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v)
{
    mImpl->SetVelocities(v);
}

void Integrator::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    mImpl->SetExternalAcceleration(aext);
}

void Integrator::SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv)
{
    mImpl->SetMassInverse(minv);
}

void Integrator::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    mImpl->SetLameCoefficients(l);
}

void Integrator::SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint)
{
    mImpl->SetCompliance(alpha, static_cast<IntegratorImpl::EConstraint>(eConstraint));
}

void Integrator::SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mImpl->SetConstraintPartitions(partitions);
}

void Integrator::SetFrictionCoefficients(GpuScalar muS, GpuScalar muK)
{
    mImpl->SetFrictionCoefficients(muS, muK);
}

void Integrator::SetSceneBoundingBox(
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    mImpl->SetSceneBoundingBox(min, max);
}

GpuMatrixX Integrator::GetVelocity() const
{
    return common::ToEigen(mImpl->GetVelocity());
}

GpuMatrixX Integrator::GetExternalAcceleration() const
{
    return common::ToEigen(mImpl->GetExternalAcceleration());
}

GpuVectorX Integrator::GetMassInverse() const
{
    return common::ToEigen(mImpl->GetMassInverse());
}

GpuMatrixX Integrator::GetLameCoefficients() const
{
    auto lame = common::ToEigen(mImpl->GetLameCoefficients());
    return lame.reshaped(2, lame.size() / 2);
}

GpuMatrixX Integrator::GetShapeMatrixInverse() const
{
    auto DmInv = common::ToEigen(mImpl->GetShapeMatrixInverse());
    return DmInv.reshaped(3, DmInv.size() / 3);
}

GpuMatrixX Integrator::GetRestStableGamma() const
{
    auto gamma = common::ToEigen(mImpl->GetRestStableGamma());
    return gamma.reshaped(2, gamma.size() / 2);
}

GpuMatrixX Integrator::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    auto lambda = common::ToEigen(
        mImpl->GetLagrangeMultiplier(static_cast<IntegratorImpl::EConstraint>(eConstraint)));
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        lambda.resize(2, lambda.size() / 2);
    }
    return lambda;
}

GpuMatrixX Integrator::GetCompliance(EConstraint eConstraint) const
{
    auto compliance = common::ToEigen(
        mImpl->GetCompliance(static_cast<IntegratorImpl::EConstraint>(eConstraint)));
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        compliance.resize(2, compliance.size() / 2);
    }
    return compliance;
}

std::vector<std::vector<GpuIndex>> Integrator::GetPartitions() const
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

GpuIndexMatrixX Integrator::GetVertexTetrahedronCollisionCandidates() const
{
    std::vector<typename IntegratorImpl::CollisionCandidateType> const overlaps =
        mImpl->GetVertexTetrahedronCollisionCandidates();
    GpuIndexMatrixX O(2, overlaps.size());
    for (auto o = 0; o < overlaps.size(); ++o)
    {
        O(0, o) = overlaps[o].first;
        O(1, o) = overlaps[o].second;
    }
    return O;
}

GpuIndexMatrixX Integrator::GetVertexTriangleContactPairs() const
{
    std::vector<typename IntegratorImpl::CollisionCandidateType> const contacts =
        mImpl->GetVertexTriangleContactPairs();
    GpuIndexMatrixX C(2, contacts.size());
    for (auto c = 0; c < contacts.size(); ++c)
    {
        C(0, c) = contacts[c].first;
        C(1, c) = contacts[c].second;
    }
    return C;
}

Integrator::~Integrator()
{
    if (mImpl != nullptr)
        delete mImpl;
}

} // namespace xpbd
} // namespace gpu
} // namespace pbat