#ifndef PBAT_GPU_SWEEP_AND_PRUNE_H
#define PBAT_GPU_SWEEP_AND_PRUNE_H

#include "Primitives.h"
#include "pbat/Aliases.h"
#include "pbat/gpu/Aliases.h"

#include <cstddef>
#include <limits>

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndPruneImpl;

class SweepAndPrune
{
  public:
    SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps);

    SweepAndPrune(SweepAndPrune const&)            = delete;
    SweepAndPrune& operator=(SweepAndPrune const&) = delete;

    SweepAndPrune(SweepAndPrune&&) noexcept;
    SweepAndPrune& operator=(SweepAndPrune&&) noexcept;

    IndexMatrixX SortAndSweep(
        Points const& P,
        Simplices const& S1,
        Simplices const& S2,
        Scalar expansion = static_cast<Scalar>(std::numeric_limits<GpuScalar>::epsilon()));

    ~SweepAndPrune();

  private:
    SweepAndPruneImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_PRUNE_H