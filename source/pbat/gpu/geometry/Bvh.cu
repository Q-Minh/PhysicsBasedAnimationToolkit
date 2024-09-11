// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Bvh.h"
#include "BvhImpl.cuh"

namespace pbat {
namespace gpu {
namespace geometry {

Bvh::Bvh(std::size_t nPrimitives, std::size_t nOverlaps)
    : mImpl(new BvhImpl(nPrimitives, nOverlaps))
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
    Points const& P,
    Simplices const& S,
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max,
    GpuScalar expansion)
{
    mImpl->Build(*P.Impl(), *S.Impl(), min, max, expansion);
}

GpuIndexMatrixX Bvh::DetectSelfOverlaps(Simplices const& S)
{
    mImpl->DetectSelfOverlaps(*S.Impl());
    auto overlaps = mImpl->overlaps.Get();
    GpuIndexMatrixX O(2, overlaps.size());
    for (auto o = 0; o < overlaps.size(); ++o)
    {
        O(0, o) = overlaps[o].first;
        O(1, o) = overlaps[o].second;
    }
    return O;
}

Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Bvh::Min() const
{
    return mImpl->Min();
}

Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Bvh::Max() const
{
    return mImpl->Max();
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> Bvh::SimplexOrdering() const
{
    return mImpl->SimplexOrdering();
}

Eigen::Vector<typename Bvh::MortonCodeType, Eigen::Dynamic> Bvh::MortonCodes() const
{
    return mImpl->MortonCodes();
}

Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Bvh::Child() const
{
    return mImpl->Child();
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> Bvh::Parent() const
{
    return mImpl->Parent();
}

Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Bvh::Rightmost() const
{
    return mImpl->Rightmost();
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> Bvh::Visits() const
{
    return mImpl->Visits();
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