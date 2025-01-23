// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Primitives.h"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/geometry/Primitives.cuh"

#include <array>
#include <thrust/copy.h>

namespace pbat {
namespace gpu {
namespace geometry {

Points::Points(Eigen::Ref<GpuMatrixX const> const& V) : mImpl(new impl::geometry::Points(V)) {}

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

impl::geometry::Points* Points::Impl()
{
    return mImpl;
}

impl::geometry::Points const* Points::Impl() const
{
    return mImpl;
}

GpuMatrixX Points::Get() const
{
    return impl::common::ToEigen(mImpl->x);
}

Points::~Points()
{
    if (mImpl != nullptr)
        delete mImpl;
}

Simplices::Simplices(Eigen::Ref<GpuIndexMatrixX const> const& C)
    : mImpl(new impl::geometry::Simplices(C))
{
}

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

impl::geometry::Simplices* Simplices::Impl()
{
    return mImpl;
}

impl::geometry::Simplices const* Simplices::Impl() const
{
    return mImpl;
}

Simplices::~Simplices()
{
    if (mImpl != nullptr)
        delete mImpl;
}

Bodies::Bodies(Eigen::Ref<GpuIndexVectorX const> const& B) : mImpl(new impl::geometry::Bodies(B)) {}

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
    return impl::common::ToEigen(mImpl->body);
}

std::size_t Bodies::NumberOfBodies() const
{
    return mImpl->NumberOfBodies();
}

impl::geometry::Bodies* geometry::Bodies::Impl()
{
    return mImpl;
}

impl::geometry::Bodies const* geometry::Bodies::Impl() const
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