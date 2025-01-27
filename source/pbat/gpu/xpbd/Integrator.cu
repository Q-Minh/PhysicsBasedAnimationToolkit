// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Integrator.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/xpbd/Integrator.cuh"

namespace pbat {
namespace gpu {
namespace xpbd {

Integrator::Integrator(
    Data const& data,
    GpuIndex nMaxVertexTetrahedronOverlaps,
    GpuIndex nMaxVertexTriangleContacts)
    : mImpl(new impl::xpbd::Integrator{
          data,
          nMaxVertexTetrahedronOverlaps,
          nMaxVertexTriangleContacts})
{
}

Integrator::Integrator(Integrator&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Integrator& Integrator::operator=(Integrator&& other) noexcept
{
    if (this != &other)
    {
        if (mImpl != nullptr)
            delete mImpl;
        mImpl       = other.mImpl;
        other.mImpl = nullptr;
    }
    return *this;
}

void Integrator::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
{
    mImpl->Step(dt, iterations, substeps);
}

GpuMatrixX Integrator::Positions() const
{
    return impl::common::ToEigen(mImpl->X.x);
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
    mImpl->SetCompliance(alpha, static_cast<impl::xpbd::Integrator::EConstraint>(eConstraint));
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
    return impl::common::ToEigen(mImpl->GetVelocity());
}

GpuMatrixX Integrator::GetExternalAcceleration() const
{
    return impl::common::ToEigen(mImpl->GetExternalAcceleration());
}

GpuVectorX Integrator::GetMassInverse() const
{
    return impl::common::ToEigen(mImpl->GetMassInverse());
}

GpuMatrixX Integrator::GetLameCoefficients() const
{
    auto lame = impl::common::ToEigen(mImpl->GetLameCoefficients());
    return lame.reshaped(2, lame.size() / 2);
}

GpuMatrixX Integrator::GetShapeMatrixInverse() const
{
    auto DmInv = impl::common::ToEigen(mImpl->GetShapeMatrixInverse());
    return DmInv.reshaped(3, DmInv.size() / 3);
}

GpuMatrixX Integrator::GetRestStableGamma() const
{
    auto gamma = impl::common::ToEigen(mImpl->GetRestStableGamma());
    return gamma.reshaped(2, gamma.size() / 2);
}

GpuMatrixX Integrator::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    auto lambda = impl::common::ToEigen(mImpl->GetLagrangeMultiplier(
        static_cast<impl::xpbd::Integrator::EConstraint>(eConstraint)));
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        lambda.resize(2, lambda.size() / 2);
    }
    return lambda;
}

GpuMatrixX Integrator::GetCompliance(EConstraint eConstraint) const
{
    auto compliance = impl::common::ToEigen(
        mImpl->GetCompliance(static_cast<impl::xpbd::Integrator::EConstraint>(eConstraint)));
    if (eConstraint == EConstraint::StableNeoHookean)
    {
        compliance.resize(2, compliance.size() / 2);
    }
    return compliance;
}

GpuIndexMatrixX Integrator::GetVertexTetrahedronCollisionCandidates() const
{
    std::vector<typename impl::xpbd::Integrator::CollisionCandidateType> const overlaps =
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
    std::vector<typename impl::xpbd::Integrator::CollisionCandidateType> const contacts =
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