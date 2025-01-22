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
     * @param n Number of primitives in the first set [0, n)
     * @param aabbs AABBs over primitives of the first [0,n) and second set [n, aabbs.size())
     * @return 2x|#overlaps| matrix of overlap pairs between primitives of the first (row 0) and
     * second set (row 1)
     */
    PBAT_API GpuIndexMatrixX SortAndSweep(GpuIndex n, Aabb& aabbs);

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
