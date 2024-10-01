#ifndef PBAT_GPU_SWEEP_AND_PRUNE_H
#define PBAT_GPU_SWEEP_AND_PRUNE_H

#include "PhysicsBasedAnimationToolkitExport.h"
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
    PBAT_API SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps);

    SweepAndPrune(SweepAndPrune const&)            = delete;
    SweepAndPrune& operator=(SweepAndPrune const&) = delete;

    PBAT_API SweepAndPrune(SweepAndPrune&&) noexcept;
    PBAT_API SweepAndPrune& operator=(SweepAndPrune&&) noexcept;

    PBAT_API IndexMatrixX SortAndSweep(
        Points const& P,
        Simplices const& S1,
        Simplices const& S2,
        Scalar expansion = static_cast<Scalar>(std::numeric_limits<GpuScalar>::epsilon()));

    PBAT_API ~SweepAndPrune();

  private:
    SweepAndPruneImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_PRUNE_H