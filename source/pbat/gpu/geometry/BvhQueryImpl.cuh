#ifndef PBAT_GPU_BVH_QUERY_IMPL_CUH
#define PBAT_GPU_BVH_QUERY_IMPL_CUH

#include "BvhImpl.cuh"
#include "PrimitivesImpl.cuh"
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

/**
 * @brief Query object reporting geometric test results between simplex sets
 */
class BvhQueryImpl
{
  public:
    using OverlapType              = typename BvhImpl::OverlapType;
    using NearestNeighbourPairType = cuda::std::pair<GpuIndex, GpuIndex>;
    using MortonCodeType           = typename BvhImpl::MortonCodeType;

    /**
     * @brief
     * @param nPrimitives
     * @param nOverlaps
     */
    BvhQueryImpl(std::size_t nPrimitives, std::size_t nOverlaps, std::size_t nNearestNeighbours);

    /**
     * @brief
     * @param P
     * @param S
     * @param min
     * @param max
     * @param expansion
     */
    void Build(
        PointsImpl const& P,
        SimplicesImpl const& S,
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
    void DetectOverlaps(
        PointsImpl const& P,
        SimplicesImpl const& S1,
        SimplicesImpl const& S2,
        BvhImpl const& bvh);

    /**
     * @brief
     * @param P Simplex primitive vertex positions
     * @param S1 Query primitives
     * @param S2 Target primitives
     * @param B1 Bodies of query primitives
     * @param B2 Bodies of target primitives
     * @param bvh Bounding volume hierarchy over S2
     * @param dhat Radius of nearest neighbour search space for each query primitive in S1
     * @param dzero Floating point error considered negligible when comparing "duplicate" nearest
     * neighbours
     */
    void DetectContactPairsFromOverlaps(
        PointsImpl const& P,
        SimplicesImpl const& S1,
        SimplicesImpl const& S2,
        BodiesImpl const& B1,
        BodiesImpl const& B2,
        BvhImpl const& bvh,
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

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_IMPL_CUH