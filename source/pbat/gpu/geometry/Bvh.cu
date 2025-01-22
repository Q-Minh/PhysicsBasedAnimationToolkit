// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Bvh.h"
#include "impl/Bvh.cuh"
#include "pbat/gpu/common/Eigen.cuh"

namespace pbat {
namespace gpu {
namespace geometry {

Bvh::Bvh(GpuIndex nPrimitives, [[maybe_unused]] GpuIndex nOverlaps)
    : mImpl(new impl::Bvh(nPrimitives))
{
}

Bvh::Bvh(Bvh&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

Bvh& Bvh::operator=(Bvh&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

void Bvh::Build(
    [[maybe_unused]] Points const& P,
    [[maybe_unused]] Simplices const& S,
    [[maybe_unused]] Eigen::Vector<GpuScalar, 3> const& min,
    [[maybe_unused]] Eigen::Vector<GpuScalar, 3> const& max,
    [[maybe_unused]] GpuScalar expansion)
{
    // mImpl->Build(*P.Impl(), *S.Impl(), min, max, expansion);
}

impl::Bvh* Bvh::Impl()
{
    return mImpl;
}

impl::Bvh const* Bvh::Impl() const
{
    return mImpl;
}

// GpuIndexMatrixX Bvh::DetectSelfOverlaps(Simplices const& S)
// {
//     mImpl->DetectSelfOverlaps(*S.Impl());
//     auto overlaps = mImpl->overlaps.Get();
//     GpuIndexMatrixX O(2, overlaps.size());
//     for (auto o = 0; o < overlaps.size(); ++o)
//     {
//         O(0, o) = overlaps[o].first;
//         O(1, o) = overlaps[o].second;
//     }
//     return O;
// }

GpuMatrixX Bvh::Min() const
{
    return gpu::common::ToEigen(mImpl->iaabbs.b);
}

GpuMatrixX Bvh::Max() const
{
    return gpu::common::ToEigen(mImpl->iaabbs.e);
}

GpuIndexVectorX Bvh::SimplexOrdering() const
{
    return gpu::common::ToEigen(mImpl->inds);
}

Eigen::Vector<typename Bvh::MortonCodeType, Eigen::Dynamic> Bvh::MortonCodes() const
{
    return gpu::common::ToEigen(mImpl->morton);
}

GpuIndexMatrixX Bvh::Child() const
{
    return gpu::common::ToEigen(mImpl->child).transpose();
}

GpuIndexVectorX Bvh::Parent() const
{
    return gpu::common::ToEigen(mImpl->parent);
}

GpuIndexMatrixX Bvh::Rightmost() const
{
    return gpu::common::ToEigen(mImpl->rightmost).transpose();
}

GpuIndexVectorX Bvh::Visits() const
{
    return gpu::common::ToEigen(mImpl->visits);
}

Bvh::~Bvh()
{
    if (mImpl != nullptr)
    {
        delete mImpl;
    }
}

} // namespace geometry
} // namespace gpu
} // namespace pbat