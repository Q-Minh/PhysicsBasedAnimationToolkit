#ifndef PBAT_GPU_GEOMETRY_IMPL_BVHQUERY_H
#define PBAT_GPU_GEOMETRY_IMPL_BVHQUERY_H

#include "Bvh.cuh"
#include "Primitives.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/gpu/common/Var.cuh"

#include <Eigen/Core>
#include <cuda/std/cmath>
#include <cuda/std/utility>
#include <limits>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {

/**
 * @brief Query object reporting geometric test results between simplex sets
 */
class BvhQuery
{
  public:
    using OverlapType              = typename Bvh::OverlapType;
    using NearestNeighbourPairType = cuda::std::pair<GpuIndex, GpuIndex>;
    using MortonCodeType           = typename Bvh::MortonCodeType;

    /**
     * @brief
     * @param nPrimitives
     * @param nOverlaps
     */
    BvhQuery(std::size_t nPrimitives, std::size_t nOverlaps, std::size_t nNearestNeighbours);

    /**
     * @brief
     * @param P
     * @param S
     * @param min
     * @param max
     * @param expansion
     */
    void Build(
        Points const& P,
        Simplices const& S,
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max,
        GpuScalar expansion = std::numeric_limits<GpuScalar>::epsilon());

    /**
     * @brief
     * @param P Simplex primitive vertex positions
     * @param S1 Query primitives
     * @param S2 Target primitives
     * @param bvh Bounding volume hierarchy over S2
     */
    void DetectOverlaps(Points const& P, Simplices const& S1, Simplices const& S2, Bvh const& bvh);

    /**
     * @brief
     * @param P Simplex primitive vertex positions
     * @param S1 Query primitives
     * @param S2 Target primitives
     * @param BV Bodies of vertex positions
     * @param bvh Bounding volume hierarchy over S2
     * @param dhat Radius of nearest neighbour search space for each query primitive in S1
     * @param dzero Floating point error considered negligible when comparing "duplicate" nearest
     * neighbours
     */
    void DetectContactPairsFromOverlaps(
        Points const& P,
        Simplices const& S1,
        Simplices const& S2,
        Bodies const& BV,
        Bvh const& bvh,
        GpuScalar dhat  = std::numeric_limits<GpuScalar>::max(),
        GpuScalar dzero = std::numeric_limits<GpuScalar>::epsilon());

    /**
     * @brief
     * @return
     */
    std::size_t NumberOfSimplices() const;
    std::size_t NumberOfAllocatedOverlaps() const;
    std::size_t NumberOfAllocatedNeighbours() const;

  private:
    common::Buffer<GpuIndex> simplex;      ///< Box/Simplex indices
    common::Buffer<MortonCodeType> morton; ///< Morton codes of simplices
    common::Buffer<GpuScalar, 3> b,
        e; ///< Simplex and internal node bounding boxes.

  public:
    common::SynchronizedList<OverlapType> overlaps;                ///< Detected overlaps
    common::SynchronizedList<NearestNeighbourPairType> neighbours; ///< Detected nearest neighbours
};

} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_IMPL_BVHQUERY_H
