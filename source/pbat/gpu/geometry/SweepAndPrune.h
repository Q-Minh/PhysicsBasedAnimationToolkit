#ifndef PBAT_GPU_GEOMETRY_SWEEPANDPRUNE_H
#define PBAT_GPU_GEOMETRY_SWEEPANDPRUNE_H

#include "Aabb.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"

#include <cstddef>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {

class SweepAndPrune;

} // namespace impl

class SweepAndPrune
{
  public:
    PBAT_API SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps);

    SweepAndPrune(SweepAndPrune const&)            = delete;
    SweepAndPrune& operator=(SweepAndPrune const&) = delete;

    PBAT_API SweepAndPrune(SweepAndPrune&&) noexcept;
    PBAT_API SweepAndPrune& operator=(SweepAndPrune&&) noexcept;
    /**
     * @brief
     *
     * @param aabbs
     * @return 2x|#overlaps| matrix of overlap pairs in aabbs
     */
    PBAT_API GpuIndexMatrixX SortAndSweep(Aabb& aabbs);
    /**
     * @brief
     *
     * @param set |#aabbs| map of indices of aabbs to their corresponding set, i.e. set[i] = j means that aabb
     * i belongs to set j.
     * @param aabbs 
     * @return 2x|#overlaps| matrix of overlap pairs between boxes of different sets
     */
    PBAT_API GpuIndexMatrixX
    SortAndSweep(Eigen::Ref<GpuIndexVectorX const> const& set, Aabb& aabbs);

    PBAT_API ~SweepAndPrune();

  private:
    void Deallocate();

    impl::SweepAndPrune* mImpl; ///<
    void* mOverlaps; ///< gpu::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_SWEEPANDPRUNE_H
