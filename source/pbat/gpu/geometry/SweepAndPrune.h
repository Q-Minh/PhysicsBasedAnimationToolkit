/**
 * @file SweepAndPrune.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Sweep and Prune GPU implementation
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */

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

/**
 * @brief GPU Sweep and Prune public API
 */
class SweepAndPrune
{
  public:
    using Impl = impl::geometry::SweepAndPrune; ///< Implementation type

    /**
     * @brief Construct a new Sweep And Prune object with space allocated for nPrimitives and at
     * most nOverlaps
     * @param nPrimitives Number of primitives
     * @param nOverlaps Maximum number of overlaps
     */
    PBAT_API SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps);

    SweepAndPrune(SweepAndPrune const&)            = delete;
    SweepAndPrune& operator=(SweepAndPrune const&) = delete;
    /**
     * @brief Move constructor
     * @param other SweepAndPrune to move from
     */
    PBAT_API SweepAndPrune(SweepAndPrune&&) noexcept;
    /**
     * @brief Move assignment
     * @param other SweepAndPrune to move from
     * @return Reference to this
     */
    PBAT_API SweepAndPrune& operator=(SweepAndPrune&&) noexcept;
    /**
     * @brief Detect overlaps between the AABBs
     * @param aabbs Handle to the AABBs
     * @return `2 x |# overlaps|` matrix of overlap pairs in aabbs
     */
    PBAT_API GpuIndexMatrixX SortAndSweep(Aabb& aabbs);
    /**
     * @brief Detect overlaps between the AABBs of different sets
     * @param set `|# aabbs|` map of indices of aabbs to their corresponding set, i.e. `set[i] = j`
     * means that aabb `i` belongs to set `j`. Must be a 1D `Buffer` of type `GpuIndex` of the same
     * size as aabbs.
     * @param aabbs The AABBs over objects
     * @return `2x|# overlaps|` matrix of overlap pairs between boxes of different sets
     */
    PBAT_API GpuIndexMatrixX SortAndSweep(common::Buffer const& set, Aabb& aabbs);

    PBAT_API ~SweepAndPrune();

  private:
    void Deallocate();

    Impl* mImpl;     ///< Pointer to the implementation
    void* mOverlaps; ///< gpu::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_SWEEPANDPRUNE_H
