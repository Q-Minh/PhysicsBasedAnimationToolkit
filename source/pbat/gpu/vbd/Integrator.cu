// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Integrator.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/vbd/AndersonIntegrator.cuh"
#include "pbat/gpu/impl/vbd/ChebyshevIntegrator.cuh"
#include "pbat/gpu/impl/vbd/Integrator.cuh"
#include "pbat/gpu/impl/vbd/TrustRegionIntegrator.cuh"

namespace pbat::gpu::vbd {

Integrator::Integrator(Data const& data) : mImpl(nullptr)
{
    using EAccelerationStrategy = pbat::sim::vbd::EAccelerationStrategy;
    switch (data.eAcceleration)
    {
        case EAccelerationStrategy::None: mImpl = new impl::vbd::Integrator(data); break;
        case EAccelerationStrategy::AcceleratedAnderson: [[fallthrough]];
        case EAccelerationStrategy::Anderson:
            mImpl = new impl::vbd::AndersonIntegrator(data);
            break;
        case EAccelerationStrategy::Chebyshev:
            mImpl = new impl::vbd::ChebyshevIntegrator(data);
            break;
        case EAccelerationStrategy::TrustRegion:
            mImpl = new impl::vbd::TrustRegionIntegrator(data);
            break;
        default: mImpl = new impl::vbd::Integrator(data); break;
    }
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

Integrator::~Integrator()
{
    if (mImpl != nullptr)
        delete mImpl;
}

void Integrator::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
{
    mImpl->Step(dt, iterations, substeps);
}

void Integrator::TracedStep(
    GpuScalar dt,
    GpuIndex iterations,
    GpuIndex substeps,
    GpuIndex t,
    std::string_view dir)
{
    mImpl->TracedStep(dt, iterations, substeps, t, dir);
}

void Integrator::SetPositions(Eigen::Ref<GpuMatrixX const> const& X)
{
    impl::common::ToBuffer(X, mImpl->x);
    impl::common::ToBuffer(X, mImpl->mPositionsAtT);
}

void Integrator::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v)
{
    impl::common::ToBuffer(v, mImpl->mVelocities);
    impl::common::ToBuffer(v, mImpl->mVelocitiesAtT);
}

void Integrator::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    impl::common::ToBuffer(aext, mImpl->mExternalAcceleration);
}

void Integrator::SetNumericalZeroForHessianDeterminant(GpuScalar zero)
{
    mImpl->mDetHZero = zero;
}

void Integrator::SetRayleighDampingCoefficient(GpuScalar kD)
{
    mImpl->mRayleighDamping = kD;
}

void Integrator::SetInitializationStrategy(EInitializationStrategy strategy)
{
    mImpl->mInitializationStrategy = strategy;
}

void Integrator::SetBlockSize(GpuIndex blockSize)
{
    mImpl->SetBlockSize(blockSize);
}

void Integrator::SetSceneBoundingBox(
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    mImpl->SetSceneBoundingBox(min, max);
}

GpuMatrixX Integrator::GetPositions() const
{
    return impl::common::ToEigen(mImpl->x);
}

GpuMatrixX Integrator::GetVelocities() const
{
    return impl::common::ToEigen(mImpl->mVelocities);
}

} // namespace pbat::gpu::vbd