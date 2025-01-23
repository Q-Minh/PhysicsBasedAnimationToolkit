#ifndef PBAT_GPU_IMPL_GEOMETRY_BVH_CUH
#define PBAT_GPU_IMPL_GEOMETRY_BVH_CUH

#include "Aabb.cuh"
#include "Morton.cuh"
#include "pbat/common/Stack.h"
#include "pbat/geometry/Morton.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/math/linalg/mini/Mini.h"

#include <exception>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace impl {
namespace geometry {

/**
 * @brief Radix-tree linear BVH
 *
 * See
 * https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf#page=4.43
 */
class Bvh
{
  public:
    static auto constexpr kDims = 3;
    using OverlapType           = cuda::std::pair<GpuIndex, GpuIndex>;
    using MortonCodeType        = pbat::geometry::MortonCodeType;

    friend class BvhQuery;

    static_assert(
        std::is_same_v<GpuIndex, std::int32_t>,
        "gpu::BvhImpl only supported for 32-bit signed integer indices");

    /**
     * @brief
     * @param nBoxes
     */
    Bvh(GpuIndex nBoxes);

    /**
     * @brief
     * @param aabbs Primitive aabbs
     * @param min World bounding box minimum
     * @param max World bounding box maximum
     */
    void Build(Aabb<kDims>& aabbs, Morton::Bound const& min, Morton::Bound const& max);
    /**
     * @brief
     *
     * @tparam FOnOverlapDetected
     * @param aabbs The same aabbs that were given to Build(), otherwise undefined behavior.
     * @param fOnOverlapDetected Callback called on detected overlaps with signature void
     * f(GpuIndex,GpuIndex)
     */
    template <class FOnOverlapDetected>
    void DetectOverlaps(Aabb<kDims> const& aabbs, FOnOverlapDetected&& fOnOverlapDetected);

    common::Buffer<GpuIndex> inds;         ///< n leaf box indices
    common::Buffer<MortonCodeType> morton; ///< n morton codes of leaf boxes
    common::Buffer<GpuIndex, 2>
        child; ///< (n-1)x2 left and right children. If child[lr][i] > n - 2, then it is
               ///< a leaf node, otherwise an internal node. lr == 0 -> left
               ///< child buffer, while lr == 1 -> right child buffer. i == 0 -> root node.
    common::Buffer<GpuIndex> parent; ///< (2n-1) parent map, s.t. parent[i] -> index of parent node
                                     ///< of node i. parent[0] == -1 <=> root node has no parent.
    common::Buffer<GpuIndex, 2>
        rightmost; ///< (n-1) rightmost map, s.t. rightmost[lr][i] -> right most leaf in left (lr ==
                   ///< 0) or right (lr == 1) subtree.
    Aabb<kDims> iaabbs; ///< (n-1) internal node bounding boxes for n leaf node bounding boxes. The
                        ///< box 0 is always the root.
    common::Buffer<GpuIndex> visits; ///< (n-1) atomic counters of internal node visits
                                     ///< for bottom-up bounding box computations
};

template <class FOnOverlapDetected>
inline void Bvh::DetectOverlaps(Aabb<kDims> const& aabbs, FOnOverlapDetected&& fOnOverlapDetected)
{
    auto const nLeafBoxes = aabbs.Size();
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nLeafBoxes),
        [inds      = inds.Raw(),
         child     = child.Raw(),
         rightmost = rightmost.Raw(),
         b         = aabbs.b.Raw(),
         e         = aabbs.e.Raw(),
         ib        = iaabbs.b.Raw(),
         ie        = iaabbs.e.Raw(),
         leafBegin = nLeafBoxes - 1,
         fOnOverlapDetected =
             std::forward<FOnOverlapDetected>(fOnOverlapDetected)] PBAT_DEVICE(GpuIndex s) mutable {
            // Traverse nodes depth-first starting from the root=0 node
            using pbat::common::Stack;
            using namespace pbat::math::linalg;
            using namespace pbat::geometry;
            auto const leaf = leafBegin + s;
            auto Ls         = mini::FromBuffers<3, 1>(b, s);
            auto Us         = mini::FromBuffers<3, 1>(e, s);
            Stack<GpuIndex, 64> stack{};
            stack.Push(0);
            do
            {
                assert(not stack.IsFull());
                GpuIndex const node = stack.Pop();
                // Check each child node for overlap.
                GpuIndex const lc       = child[0][node];
                GpuIndex const rc       = child[1][node];
                bool const bIsLeftLeaf  = lc >= leafBegin;
                bool const bIsRightLeaf = rc >= leafBegin;
                auto Llc                = bIsLeftLeaf ? mini::FromBuffers<3, 1>(b, lc - leafBegin) :
                                                        mini::FromBuffers<3, 1>(ib, lc);
                auto Ulc                = bIsLeftLeaf ? mini::FromBuffers<3, 1>(e, lc - leafBegin) :
                                                        mini::FromBuffers<3, 1>(ie, lc);
                auto Lrc = bIsRightLeaf ? mini::FromBuffers<3, 1>(b, rc - leafBegin) :
                                          mini::FromBuffers<3, 1>(ib, rc);
                auto Urc = bIsRightLeaf ? mini::FromBuffers<3, 1>(e, rc - leafBegin) :
                                          mini::FromBuffers<3, 1>(ie, rc);
                bool const bLeftBoxOverlaps =
                    OverlapQueries::AxisAlignedBoundingBoxes(Ls, Us, Llc, Ulc) and
                    (rightmost[0][node] > leaf);
                bool const bRightBoxOverlaps =
                    OverlapQueries::AxisAlignedBoundingBoxes(Ls, Us, Lrc, Urc) and
                    (rightmost[1][node] > leaf);

                // Leaf overlaps another leaf node
                if (bLeftBoxOverlaps and bIsLeftLeaf)
                {
                    GpuIndex const si = inds[s];
                    GpuIndex const sj = inds[lc - leafBegin];
                    fOnOverlapDetected(si, sj);
                }
                if (bRightBoxOverlaps and bIsRightLeaf)
                {
                    GpuIndex const si = inds[s];
                    GpuIndex const sj = inds[rc - leafBegin];
                    fOnOverlapDetected(si, sj);
                }

                // Leaf overlaps an internal node -> traverse
                bool const bTraverseLeft  = bLeftBoxOverlaps and not bIsLeftLeaf;
                bool const bTraverseRight = bRightBoxOverlaps and not bIsRightLeaf;
                if (bTraverseLeft)
                    stack.Push(lc);
                if (bTraverseRight)
                    stack.Push(rc);
            } while (not stack.IsEmpty());
        });
}

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVHIMPL_H
