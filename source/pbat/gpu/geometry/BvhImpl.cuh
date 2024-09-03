#ifndef PBAT_GPU_BVH_IMPL_CUH
#define PBAT_GPU_BVH_IMPL_CUH

#include "PrimitivesImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/Var.cuh"

#include <cuda/std/utility>
#include <limits>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace geometry {

/**
 * @brief Radix-tree linear BVH
 *
 * See https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf#page=4.43
 */
class BvhImpl
{
  public:
    using OverlapType = cuda::std::pair<GpuIndex, GpuIndex>;

    static_assert(
        std::is_same_v<GpuScalar, float>,
        "gpu::BvhImpl only supported for single precision floating point numbers");

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
        GpuScalar expansion = std::numeric_limits<GpuScalar>::min());

    /**
     * @brief
     * @return
     */
    std::size_t NumberOfAllocatedBoxes() const;

  private:
    common::Buffer<GpuIndex> simplex; ///< Box/Simplex indices
    common::Buffer<GpuIndex> morton;  ///< Morton codes of simplices
    common::Buffer<GpuIndex, 2>
        child; ///< Left and right children. If child[lr][i] > n - 2, then it is
               ///< a leaf node, otherwise an internal node. lr == 0 -> left
               ///< child buffer, while lr == 1 -> right child buffer. i == 0 -> root node.
    common::Buffer<GpuIndex> parent; ///< parent[i] -> index of parent node of node i
    common::Buffer<GpuScalar, 3> b,
        e; ///< Simplex and internal node bounding boxes. The first n-1 boxes are internal node
           ///< bounding boxes. The next n boxes are leaf node (i.e. simplex) bounding boxes. The
           ///< box 0 is always the root.

  public:
    common::Var<GpuIndex> no;      ///< Number of overlaps
    common::Buffer<OverlapType> o; ///< Overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_IMPL_CUH