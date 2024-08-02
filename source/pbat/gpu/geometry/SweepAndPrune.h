#ifndef PBAT_GPU_SWEEP_AND_PRUNE_H
#define PBAT_GPU_SWEEP_AND_PRUNE_H

#define EIGEN_NO_CUDA
#include "pbat/Aliases.h"
#undef EIGEN_NO_CUDA

#include "Primitives.h"

#include <cstddef>

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

    IndexMatrixX
    SortAndSweep(Points const& P, Simplices const& S1, Simplices const& S2, Scalar expansion = 0.);

    ~SweepAndPrune();

  private:
    SweepAndPruneImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_PRUNE_H