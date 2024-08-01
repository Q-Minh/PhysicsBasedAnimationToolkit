#ifndef PBAT_GPU_SWEEP_AND_PRUNE_H
#define PBAT_GPU_SWEEP_AND_PRUNE_H

#define EIGEN_NO_CUDA
#include "pbat/Aliases.h"
#undef EIGEN_NO_CUDA

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

    IndexMatrixX SortAndSweep(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C1,
        Eigen::Ref<IndexMatrixX const> const& C2);

    ~SweepAndPrune();

  private:
    SweepAndPruneImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_PRUNE_H