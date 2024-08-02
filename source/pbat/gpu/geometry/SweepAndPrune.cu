#include "SweepAndPrune.h"
#include "SweepAndPruneImpl.cuh"

namespace pbat {
namespace gpu {
namespace geometry {

SweepAndPrune::SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps)
    : mImpl(new SweepAndPruneImpl(nPrimitives, nOverlaps))
{
}

SweepAndPrune::SweepAndPrune(SweepAndPrune&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

SweepAndPrune& SweepAndPrune::operator=(SweepAndPrune&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

IndexMatrixX SweepAndPrune::SortAndSweep(
    Points const& P,
    Simplices const& S1,
    Simplices const& S2,
    Scalar expansion)
{
    mImpl->SortAndSweep(*P.Impl(), *S1.Impl(), *S2.Impl(), static_cast<GpuScalar>(expansion));
    auto const overlapPairs = mImpl->Overlaps();
    IndexMatrixX overlaps(2, overlapPairs.size());
    for (auto o = 0; o < overlapPairs.size(); ++o)
    {
        overlaps(0, o) = overlapPairs[o].first;
        overlaps(1, o) = overlapPairs[o].second;
    }
    return overlaps;
}

SweepAndPrune::~SweepAndPrune()
{
    if (mImpl != nullptr)
        delete mImpl;
}

} // namespace geometry
} // namespace gpu
} // namespace pbat