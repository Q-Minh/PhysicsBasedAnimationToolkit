// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Integrator.h"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/vbd/Integrator.cuh"

namespace pbat {
namespace gpu {
namespace vbd {

Integrator::Integrator(Data const& data) : mImpl(new impl::vbd::Integrator(data)) {}

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

void Integrator::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps, GpuScalar rho)
{
    mImpl->Step(dt, iterations, substeps, rho);
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

void Integrator::SetMass(Eigen::Ref<GpuVectorX const> const& m)
{
    mImpl->SetMass(m);
}

void Integrator::SetQuadratureWeights(Eigen::Ref<GpuVectorX const> const& wg)
{
    mImpl->SetQuadratureWeights(wg);
}

void Integrator::SetShapeFunctionGradients(Eigen::Ref<GpuMatrixX const> const& GP)
{
    mImpl->SetShapeFunctionGradients(GP);
}

void Integrator::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    mImpl->SetLameCoefficients(l);
}

void Integrator::SetNumericalZeroForHessianDeterminant(GpuScalar zero)
{
    mImpl->SetNumericalZeroForHessianDeterminant(zero);
}

void Integrator::SetVertexTetrahedronAdjacencyList(
    Eigen::Ref<GpuIndexVectorX const> const& GVTp,
    Eigen::Ref<GpuIndexVectorX const> const& GVTn,
    Eigen::Ref<GpuIndexVectorX const> const& GVTilocal)
{
    mImpl->SetVertexTetrahedronAdjacencyList(GVTp, GVTn, GVTilocal);
}

void Integrator::SetRayleighDampingCoefficient(GpuScalar kD)
{
    mImpl->SetRayleighDampingCoefficient(kD);
}

void Integrator::SetVertexPartitions(
    Eigen::Ref<GpuIndexVectorX const> const& Pptr,
    Eigen::Ref<GpuIndexVectorX const> const& Padj)
{
    mImpl->SetVertexPartitions(Pptr, Padj);
}

void Integrator::SetInitializationStrategy(EInitializationStrategy strategy)
{
    mImpl->SetInitializationStrategy(strategy);
}

void Integrator::SetBlockSize(GpuIndex blockSize)
{
    mImpl->SetBlockSize(blockSize);
}

GpuMatrixX Integrator::GetPositions() const
{
    return impl::common::ToEigen(mImpl->X.x);
}

GpuMatrixX Integrator::GetVelocities() const
{
    return impl::common::ToEigen(mImpl->GetVelocity());
}

} // namespace vbd
} // namespace gpu
} // namespace pbat