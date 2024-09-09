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