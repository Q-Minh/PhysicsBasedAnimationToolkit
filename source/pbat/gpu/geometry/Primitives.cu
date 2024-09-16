// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Primitives.h"
#include "PrimitivesImpl.cuh"
#include "pbat/common/Eigen.h"

#include <array>
#include <thrust/copy.h>

namespace pbat {
namespace gpu {
namespace geometry {

Points::Points(Eigen::Ref<GpuMatrixX const> const& V) : mImpl(new PointsImpl(V)) {}

Points::Points(Points&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Points& Points::operator=(Points&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

void Points::Update(Eigen::Ref<GpuMatrixX const> const& V)
{
    mImpl->Update(V);
}

PointsImpl* Points::Impl()
{
    return mImpl;
}

PointsImpl const* Points::Impl() const
{
    return mImpl;
}

GpuMatrixX Points::Get() const
{
    GpuMatrixX V(mImpl->x.Dimensions(), mImpl->x.Size());
    for (auto d = 0; d < V.rows(); ++d)
    {
        thrust::copy(mImpl->x[d].begin(), mImpl->x[d].end(), V.row(d).begin());
    }
    return V;
}

Points::~Points()
{
    if (mImpl != nullptr)
        delete mImpl;
}

Simplices::Simplices(Eigen::Ref<GpuIndexMatrixX const> const& C) : mImpl(new SimplicesImpl(C)) {}

Simplices::Simplices(Simplices&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Simplices& Simplices::operator=(Simplices&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

GpuIndexMatrixX Simplices::Get() const
{
    auto const m = static_cast<int>(mImpl->eSimplexType);
    GpuIndexMatrixX C(m, mImpl->NumberOfSimplices());
    for (auto k = 0; k < m; ++k)
    {
        thrust::copy(mImpl->inds[k].begin(), mImpl->inds[k].end(), C.row(k).begin());
    }
    return C;
}

Simplices::ESimplexType Simplices::Type() const
{
    return static_cast<ESimplexType>(mImpl->eSimplexType);
}

SimplicesImpl* Simplices::Impl()
{
    return mImpl;
}

SimplicesImpl const* Simplices::Impl() const
{
    return mImpl;
}

Simplices::~Simplices()
{
    if (mImpl != nullptr)
        delete mImpl;
}

Bodies::Bodies(Eigen::Ref<GpuIndexVectorX const> const& B) : mImpl(new BodiesImpl(B)) {}

Bodies::Bodies(Bodies&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Bodies& geometry::Bodies::operator=(Bodies&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

GpuIndexMatrixX geometry::Bodies::Get() const
{
    using pbat::common::ToEigen;
    return ToEigen(mImpl->body.Get());
}

std::size_t Bodies::NumberOfBodies() const
{
    return mImpl->NumberOfBodies();
}

BodiesImpl* geometry::Bodies::Impl()
{
    return mImpl;
}

BodiesImpl const* geometry::Bodies::Impl() const
{
    return mImpl;
}

geometry::Bodies::~Bodies()
{
    if (mImpl != nullptr)
        delete mImpl;
}

} // namespace geometry
} // namespace gpu
} // namespace pbat