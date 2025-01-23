#ifndef PBAT_GPU_GEOMETRY_SWEEPANDPRUNE_H
#define PBAT_GPU_GEOMETRY_SWEEPANDPRUNE_H

#include "Aabb.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.h"

#include <cstddef>

namespace pbat::gpu::impl::geometry {
class SweepAndPrune;
} // namespace pbat::gpu::impl::geometry

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndPrune
{
  public:
    using Impl = impl::geometry::SweepAndPrune;

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
     * @param set |#aabbs| map of indices of aabbs to their corresponding set, i.e. set[i] = j means
     * that aabb i belongs to set j. Must be a 1D Buffer of type GpuIndex of the same size as aabbs.
     * @param aabbs
     * @return 2x|#overlaps| matrix of overlap pairs between boxes of different sets
     */
    PBAT_API GpuIndexMatrixX SortAndSweep(common::Buffer const& set, Aabb& aabbs);

    PBAT_API ~SweepAndPrune();

  private:
    void Deallocate();

    Impl* mImpl;     ///<
    void* mOverlaps; ///< gpu::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_SWEEPANDPRUNE_H
