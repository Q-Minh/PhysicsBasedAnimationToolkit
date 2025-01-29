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

Integrator::Integrator(Data const& data) : mImpl(new impl::xpbd::Integrator{data}) {}

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
    return impl::common::ToEigen(mImpl->x);
}

void Integrator::SetPositions(Eigen::Ref<GpuMatrixX const> const& X)
{
    impl::common::ToBuffer(X, mImpl->x);
}

void Integrator::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v)
{
    impl::common::ToBuffer(v, mImpl->v);
}

void Integrator::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    impl::common::ToBuffer(aext, mImpl->aext);
}

void Integrator::SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv)
{
    impl::common::ToBuffer(minv, mImpl->minv);
}

void Integrator::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    impl::common::ToBuffer(l, mImpl->lame);
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
    return impl::common::ToEigen(mImpl->v);
}

GpuMatrixX Integrator::GetExternalAcceleration() const
{
    return impl::common::ToEigen(mImpl->aext);
}

GpuVectorX Integrator::GetMassInverse() const
{
    return impl::common::ToEigen(mImpl->minv);
}

GpuMatrixX Integrator::GetLameCoefficients() const
{
    auto lame = impl::common::ToEigen(mImpl->lame);
    return lame.reshaped(2, lame.size() / 2);
}

GpuMatrixX Integrator::GetShapeMatrixInverse() const
{
    auto DmInv = impl::common::ToEigen(mImpl->DmInv);
    return DmInv.reshaped(3, DmInv.size() / 3);
}

GpuMatrixX Integrator::GetRestStableGamma() const
{
    auto gamma = impl::common::ToEigen(mImpl->gamma);
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

Integrator::~Integrator()
{
    if (mImpl != nullptr)
        delete mImpl;
}

} // namespace xpbd
} // namespace gpu
} // namespace pbat