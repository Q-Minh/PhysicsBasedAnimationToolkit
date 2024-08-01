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
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

SweepAndPrune::~SweepAndPrune()
{
    if (mImpl != nullptr)
        delete mImpl;
}

} // namespace geometry
} // namespace gpu
} // namespace pbat