#ifndef PBAT_GPU_BVH_IMPL_CUH
#define PBAT_GPU_BVH_IMPL_CUH

#include "PrimitivesImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/Morton.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/gpu/common/Var.cuh"

#include <cuda/std/cmath>
#include <cuda/std/utility>
#include <limits>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace geometry {

/**
 * @brief Radix-tree linear BVH
 *
 * See
 * https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf#page=4.43
 */
class BvhImpl
{
  public:
    using OverlapType    = cuda::std::pair<GpuIndex, GpuIndex>;
    using MortonCodeType = pbat::gpu::common::MortonCodeType;

    friend class BvhQueryImpl;

    static_assert(
        std::is_same_v<GpuIndex, int>,
        "gpu::BvhImpl only supported for 32-bit signed integer indices");

    /**
     * @brief
     * @param nPrimitives
     * @param nOverlaps
     */
    BvhImpl(std::size_t nPrimitives, std::size_t nOverlaps);

    /**
     * @brief
     * @param P
     * @param S
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
     * @param S The simplices which were used to build this BVH
     */
    void DetectSelfOverlaps(SimplicesImpl const& S);

    /**
     * @brief
     * @return
     */
    std::size_t NumberOfAllocatedBoxes() const;
    /**
     * @brief BVH nodes' box minimums
     * @return
     */
    Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Min() const;
    /**
     * @brief BVH nodes' box maximums
     * @return
     */
    Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Max() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<GpuIndex, Eigen::Dynamic> SimplexOrdering() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<MortonCodeType, Eigen::Dynamic> MortonCodes() const;
    /**
     * @brief
     * @return
     */
    Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Child() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<GpuIndex, Eigen::Dynamic> Parent() const;
    /**
     * @brief
     * @return
     */
    Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Rightmost() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<GpuIndex, Eigen::Dynamic> Visits() const;

  private:
    common::Buffer<GpuIndex> simplex;      ///< Box/Simplex indices
    common::Buffer<MortonCodeType> morton; ///< Morton codes of simplices
    common::Buffer<GpuIndex, 2>
        child; ///< Left and right children. If child[lr][i] > n - 2, then it is
               ///< a leaf node, otherwise an internal node. lr == 0 -> left
               ///< child buffer, while lr == 1 -> right child buffer. i == 0 -> root node.
    common::Buffer<GpuIndex> parent;       ///< parent[i] -> index of parent node of node i.
                                           ///< parent[0] == -1 <=> root node has no parent.
    common::Buffer<GpuIndex, 2> rightmost; ///< rightmost[lr][i] -> right most leaf in left (lr ==
                                           ///< 0) or right (lr == 1) subtree.
    common::Buffer<GpuScalar, 3> b,
        e; ///< Simplex and internal node bounding boxes. The first n-1 boxes are internal node
           ///< bounding boxes. The next n boxes are leaf node (i.e. simplex) bounding boxes. The
           ///< box 0 is always the root.
    common::Buffer<GpuIndex> visits; ///< Atomic counter of internal node visits
                                     ///< for bottom-up bounding box computations

  public:
    common::SynchronizedList<OverlapType> overlaps; ///< Detected overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_IMPL_CUH