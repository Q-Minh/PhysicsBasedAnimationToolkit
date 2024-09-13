// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "BvhQuery.h"
#include "BvhQueryImpl.cuh"
#include "PrimitivesImpl.cuh"

namespace pbat {
namespace gpu {
namespace geometry {

BvhQuery::BvhQuery(std::size_t nPrimitives, std::size_t nOverlaps, std::size_t nNearestNeighbours)
    : mImpl(new BvhQueryImpl(nPrimitives, nOverlaps, nNearestNeighbours))
{
}

BvhQuery::BvhQuery(BvhQuery&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

BvhQuery& BvhQuery::operator=(BvhQuery&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

void BvhQuery::Build(
    Points const& P,
    Simplices const& S,
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max,
    GpuScalar expansion)
{
    mImpl->Build(*P.Impl(), *S.Impl(), min, max, expansion);
}

GpuIndexMatrixX
BvhQuery::DetectOverlaps(Points const& P, Simplices const& S1, Simplices const& S2, Bvh const& bvh)
{
    mImpl->DetectOverlaps(*P.Impl(), *S1.Impl(), *S2.Impl(), *bvh.Impl());
    auto overlaps = mImpl->overlaps.Get();
    GpuIndexMatrixX O(2, overlaps.size());
    for (auto o = 0; o < overlaps.size(); ++o)
    {
        O(0, o) = overlaps[o].first;
        O(1, o) = overlaps[o].second;
    }
    return O;
}

BvhQuery::~BvhQuery()
{
    if (mImpl != nullptr)
    {
        delete mImpl;
    }
}

} // namespace geometry
} // namespace gpu
} // namespace pbat