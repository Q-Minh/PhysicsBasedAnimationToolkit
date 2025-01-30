#ifndef PBAT_GPU_IMPL_GEOMETRY_BVH_CUH
#define PBAT_GPU_IMPL_GEOMETRY_BVH_CUH

#include "Aabb.cuh"
#include "Morton.cuh"
#include "pbat/common/Queue.h"
#include "pbat/common/Stack.h"
#include "pbat/geometry/Morton.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"

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
     * @brief Build BVH from primitive aabbs
     * @param aabbs Primitive aabbs
     * @param min World bounding box minimum
     * @param max World bounding box maximum
     */
    void Build(Aabb<kDims>& aabbs, Morton::Bound const& min, Morton::Bound const& max);
    /**
     * @brief
     *
     * @param aabbs
     * @param min
     * @param max
     */
    void SortByMortonCode(Aabb<kDims>& aabbs, Morton::Bound const& min, Morton::Bound const& max);
    /**
     * @brief Builds the BVH's hierarchy, assuming primitives have been sorted.
     *
     * @param n Number of leaf boxes
     */
    void BuildTree(GpuIndex n);
    /**
     * @brief Computes internal node bounding boxes, assuming the BVH's hierarchy is built.
     *
     * @param aabbs
     */
    void ConstructBoxes(Aabb<kDims>& aabbs);
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
    /**
     * @brief
     *
     * @tparam FOnFound
     * @tparam kStackSize
     * @param leafAabbs
     * @param queryAabbs
     * @param fOnOverlapDetected
     */
    template <class FOnOverlapDetected, auto kStackSize = 64>
    void DetectOverlaps(
        Aabb<kDims> const& leafAabbs,
        Aabb<kDims> const& queryAabbs,
        FOnOverlapDetected fOnOverlapDetected);
    /**
     * @brief
     *
     * @tparam FGetQueryObject Callable with signature TQuery f(GpuIndex)
     * @tparam FGetLeafObject Callable with signature TLeaf f(GpuIndex leaf, GpuIndex i)
     * @tparam FMinDistanceToBox Callable with signature GpuScalar f(TQuery, Point, Point) where
     * Point=pbat::math::linalg::mini::SVector<GpuScalar, 3>
     * @tparam FDistanceToLeaf Callable with signature GpuScalar f(TQuery, TLeaf)
     * @tparam FDistanceUpperBound Callable with signature GpuScalar f(GpuIndex)
     * @tparam FOnNearestNeighbourFound Callable with signature void f(GpuIndex query, GpuIndex
     * i, GpuScalar d, GpuIndex k)
     * @tparam kQueueSize Nearest neighbour queue size
     * @tparam kStackSize Branch and bound minimization's depth-first search stack size
     *
     * @param aabbs The same aabbs that were given to Build(), otherwise undefined behavior.
     * @param nQueries Number of queries
     * @param fGetQueryObject Callable to get query object with signature TQuery f(GpuIndex)
     * @param fGetLeafObject Callable to get leaf object with signature TLeaf f(GpuIndex leaf,
     * GpuIndex i)
     * @param fMinDistanceToBox Callable to get minimum distance to box with signature GpuScalar
     * f(TQuery, Point, Point)
     * @param fDistanceToleaf Callable to get distance to leaf with signature GpuScalar f(TQuery,
     * TLeaf)
     * @param fDistanceUpperBound Callable to get query q's distance upper bound with signature
     * GpuScalar f(GpuIndex)
     * @param fOnNearestNeighbourFound Callable to get nearest neighbour with signature void
     * f(GpuIndex q, GpuIndex i, GpuScalar d, GpuIndex k)
     * @param eps Epsilon for distance comparison
     */
    template <
        class FGetQueryObject,
        class FGetLeafObject,
        class FMinDistanceToBox,
        class FDistanceToLeaf,
        class FDistanceUpperBound,
        class FOnNearestNeighbourFound,
        auto kQueueSize = 8,
        auto kStackSize = 64>
    void NearestNeighbours(
        Aabb<kDims> const& aabbs,
        GpuIndex nQueries,
        FGetQueryObject fGetQueryObject,
        FGetLeafObject fGetLeafObject,
        FMinDistanceToBox fMinDistanceToBox,
        FDistanceToLeaf fDistanceToleaf,
        FDistanceUpperBound fDistanceUpperBound,
        FOnNearestNeighbourFound fOnNearestNeighbourFound,
        GpuScalar eps = GpuScalar(0));
    /**
     * @brief
     *
     * @tparam FGetQueryObject Callable with signature TQuery f(GpuIndex)
     * @tparam FGetLeafObject Callable with signature TLeaf f(GpuIndex leaf, GpuIndex i)
     * @tparam FMinDistanceToBox Callable with signature GpuScalar f(TQuery, Point, Point) where
     * Point=pbat::math::linalg::mini::SVector<GpuScalar, 3>
     * @tparam FDistanceToLeaf Callable with signature GpuScalar f(TQuery, TLeaf)
     * @tparam FDistanceUpperBound Callable with signature GpuScalar f(GpuIndex)
     * @tparam FOnFound Callable with signature void f(GpuIndex q, TQuery query, GpuIndex leafIdx,
     * GpuIndex i, TLeaf leaf, GpuScalar d)
     * @tparam kStackSize Branch and bound minimization's depth-first search stack size
     *
     * @param aabbs The same aabbs that were given to Build(), otherwise undefined behavior.
     * @param nQueries Number of queries
     * @param fGetQueryObject Callable to get query object with signature TQuery f(GpuIndex)
     * @param fGetLeafObject Callable to get leaf object with signature TLeaf f(GpuIndex leaf,
     * GpuIndex i)
     * @param fMinDistanceToBox Callable to get minimum distance to box with signature GpuScalar
     * @param fDistanceToleaf Callable to get distance to leaf with signature GpuScalar
     * f(TQuery,TLeaf)
     * @param fDistanceUpperBound Callable to get query q's distance upper bound with signature
     * GpuScalar f(GpuIndex)
     * @param fOnFound Callable for handling satisfied queries with signature
     * void f(GpuIndex q, TQuery query, GpuIndex leafIdx, GpuIndex i, TLeaf leaf, GpuScalar d)
     */
    template <
        class FGetQueryObject,
        class FGetLeafObject,
        class FMinDistanceToBox,
        class FDistanceToLeaf,
        class FDistanceUpperBound,
        class FOnFound,
        auto kStackSize = 64>
    void RangeSearch(
        Aabb<kDims> const& aabbs,
        GpuIndex nQueries,
        FGetQueryObject fGetQueryObject,
        FGetLeafObject fGetLeafObject,
        FMinDistanceToBox fMinDistanceToBox,
        FDistanceToLeaf fDistanceToleaf,
        FDistanceUpperBound fDistanceUpperBound,
        FOnFound fOnFound);

    common::Buffer<GpuIndex> inds; ///< n leaf box indices
    Morton morton;                 ///< Morton codes of leaf boxes
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
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.geometry.Bvh.DetectOverlaps");
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
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

template <class FOnOverlapDetected, auto kStackSize>
inline void Bvh::DetectOverlaps(
    Aabb<kDims> const& leafAabbs,
    Aabb<kDims> const& queryAabbs,
    FOnOverlapDetected fOnOverlapDetected)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.geometry.Bvh.DetectOverlaps");

    static_assert(
        std::is_invocable_v<FOnOverlapDetected, GpuIndex, GpuIndex>,
        "FOnOverlapDetected must be callable with signature void f(GpuIndex,GpuIndex)");

    using namespace pbat::math::linalg;
    auto fGetQueryObject = [b = queryAabbs.b.Raw(),
                            e = queryAabbs.e.Raw()] PBAT_DEVICE(GpuIndex q) {
        mini::SMatrix<GpuScalar, 3, 2> LU;
        LU.Col(0) = mini::FromBuffers<3, 1>(b, q);
        LU.Col(1) = mini::FromBuffers<3, 1>(e, q);
        return LU;
    };
    auto fGetLeafObject =
        [b = leafAabbs.b.Raw(),
         e = leafAabbs.e.Raw()] PBAT_DEVICE(GpuIndex leaf, [[maybe_unused]] GpuIndex i) {
            mini::SMatrix<GpuScalar, 3, 2> LU;
            LU.Col(0) = mini::FromBuffers<3, 1>(b, leaf);
            LU.Col(1) = mini::FromBuffers<3, 1>(e, leaf);
            return LU;
        };
    auto fMinDistanceToBox = [] PBAT_DEVICE(
                                 mini::SMatrix<GpuScalar, 3, 2> const& LU1,
                                 mini::SVector<GpuScalar, 3> const& L2,
                                 mini::SVector<GpuScalar, 3> const& U2) {
        bool const bBoxesOverlap = pbat::geometry::OverlapQueries::AxisAlignedBoundingBoxes(
            LU1.Col(0),
            LU1.Col(1),
            L2,
            U2);
        return static_cast<GpuScalar>(not bBoxesOverlap);
    };
    auto fDistanceToLeaf = [] PBAT_DEVICE(
                               mini::SMatrix<GpuScalar, 3, 2> const& LU1,
                               mini::SMatrix<GpuScalar, 3, 2> const& LU2) {
        return GpuScalar(0);
    };
    auto fQueryUpperBound = [] PBAT_DEVICE(GpuIndex q) {
        return GpuScalar(0);
    };
    auto fOnFound = [fOnOverlapDetected] PBAT_DEVICE(
                        GpuIndex q,
                        [[maybe_unused]] mini::SMatrix<GpuScalar, 3, 2> const& LU1,
                        [[maybe_unused]] GpuIndex leaf,
                        GpuIndex i,
                        [[maybe_unused]] mini::SMatrix<GpuScalar, 3, 2> const& LU2,
                        [[maybe_unused]] GpuScalar d) {
        fOnOverlapDetected(q, i);
    };

    this->RangeSearch<
        decltype(fGetQueryObject),
        decltype(fGetLeafObject),
        decltype(fMinDistanceToBox),
        decltype(fDistanceToLeaf),
        decltype(fQueryUpperBound),
        decltype(fOnFound),
        kStackSize>(
        leafAabbs,
        queryAabbs.Size(),
        fGetQueryObject,
        fGetLeafObject,
        fMinDistanceToBox,
        fDistanceToLeaf,
        fQueryUpperBound,
        fOnFound);

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

template <
    class FGetQueryObject,
    class FGetLeafObject,
    class FMinDistanceToBox,
    class FDistanceToLeaf,
    class FDistanceUpperBound,
    class FOnNearestNeighbourFound,
    auto kQueueSize,
    auto kStackSize>
inline void Bvh::NearestNeighbours(
    Aabb<kDims> const& aabbs,
    GpuIndex nQueries,
    FGetQueryObject fGetQueryObject,
    FGetLeafObject fGetLeafObject,
    FMinDistanceToBox fMinDistanceToBox,
    FDistanceToLeaf fDistanceToLeaf,
    FDistanceUpperBound fDistanceUpperBound,
    FOnNearestNeighbourFound fOnNearestNeighbourFound,
    GpuScalar eps)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.geometry.Bvh.NearestNeighbours");

    using TQuery = std::invoke_result_t<FGetQueryObject, GpuIndex>;
    using TLeaf  = std::invoke_result_t<FGetLeafObject, GpuIndex, GpuIndex>;
    using Point  = pbat::math::linalg::mini::SVector<GpuScalar, kDims>;

    static_assert(
        std::is_invocable_v<FGetQueryObject, GpuIndex> and not std::is_same_v<TQuery, void>,
        "FGetQueryObject must be callable with signature TQuery f(GpuIndex)");
    static_assert(
        std::is_invocable_v<FGetLeafObject, GpuIndex, GpuIndex> and not std::is_same_v<TLeaf, void>,
        "FGetLeafObject must be callable with signature TLeaf f(GpuIndex,GpuIndex)");
    static_assert(
        std::is_invocable_v<FMinDistanceToBox, TQuery, Point, Point> and
            std::is_convertible_v<
                std::invoke_result_t<FMinDistanceToBox, TQuery, Point, Point>,
                GpuScalar>,
        "FMinDistanceToBox must be callable with signature GpuScalar f(TQuery, Point, Point) where "
        "Point=pbat::math::linalg::mini::SVector<GpuScalar, 3>");
    static_assert(
        std::is_invocable_v<FDistanceToLeaf, TQuery, TLeaf> and
            std::is_convertible_v<std::invoke_result_t<FDistanceToLeaf, TQuery, TLeaf>, GpuScalar>,
        "FDistanceToLeaf must be callable with signature GpuScalar f(TQuery, TLeaf)");
    static_assert(
        std::is_invocable_v<FDistanceUpperBound, GpuIndex> and
            std::is_convertible_v<std::invoke_result_t<FDistanceUpperBound, GpuIndex>, GpuScalar>,
        "FDistanceUpperBound must be callable with signature GpuScalar f(GpuIndex)");
    static_assert(
        std::is_invocable_v<FOnNearestNeighbourFound, GpuIndex, GpuIndex, GpuScalar, GpuIndex>,
        "FOnNearestNeighbourFound must be callable with signature void f(GpuIndex query, GpuIndex "
        "leaf, GpuScalar dmin, GpuIndex k)");

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nQueries),
        [eps,
         fGetQueryObject,
         fGetLeafObject,
         fMinDistanceToBox,
         fDistanceToLeaf,
         fDistanceUpperBound,
         fOnNearestNeighbourFound,
         leafBegin = aabbs.Size() - 1,
         inds      = inds.Raw(),
         b         = aabbs.b.Raw(),
         e         = aabbs.e.Raw(),
         ib        = iaabbs.b.Raw(),
         ie        = iaabbs.e.Raw(),
         child     = child.Raw()] PBAT_DEVICE(GpuIndex q) mutable {
            using pbat::math::linalg::mini::FromBuffers;
            using Stack = pbat::common::Stack<GpuIndex, kStackSize>;
            using Queue = pbat::common::Queue<GpuIndex, kQueueSize>;

            // Depth-first branch and bound distance minimization
            Stack dfs{};
            Queue nn{};
            GpuScalar dmin{fDistanceUpperBound(q)};
            TQuery const query{fGetQueryObject(q)};

            dfs.Push(0 /*root*/);
            do
            {
                assert(not dfs.IsFull());
                GpuIndex nodeIdx = dfs.Top();
                dfs.Pop();
                bool const bIsLeafNode = nodeIdx >= leafBegin;
                if (not bIsLeafNode)
                {
                    auto L = FromBuffers<3, 1>(ib, nodeIdx);
                    auto U = FromBuffers<3, 1>(ie, nodeIdx);
                    if (fMinDistanceToBox(query, L, U) < dmin)
                    {
                        dfs.Push(child[0][nodeIdx]);
                        dfs.Push(child[1][nodeIdx]);
                    }
                }
                else
                {
                    nodeIdx -= leafBegin;
                    auto L             = FromBuffers<3, 1>(b, nodeIdx);
                    auto U             = FromBuffers<3, 1>(e, nodeIdx);
                    GpuScalar const db = fMinDistanceToBox(query, L, U);
                    if (db < dmin)
                    {
                        GpuIndex const i  = inds[nodeIdx];
                        GpuScalar const d = fDistanceToLeaf(query, fGetLeafObject(nodeIdx, i));
                        if (d < dmin)
                        {
                            nn.Clear();
                            nn.Push(i);
                            dmin = d;
                        }
                        else if (d < dmin + eps and not nn.IsFull())
                        {
                            nn.Push(i);
                        }
                    }
                }
            } while (not dfs.IsEmpty());
            GpuIndex k{0};
            while (not nn.IsEmpty())
            {
                GpuIndex const i = nn.Top();
                nn.Pop();
                fOnNearestNeighbourFound(q, i, dmin, k++);
            }
        });

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

template <
    class FGetQueryObject,
    class FGetLeafObject,
    class FMinDistanceToBox,
    class FDistanceToLeaf,
    class FDistanceUpperBound,
    class FOnFound,
    auto kStackSize>
inline void Bvh::RangeSearch(
    Aabb<kDims> const& aabbs,
    GpuIndex nQueries,
    FGetQueryObject fGetQueryObject,
    FGetLeafObject fGetLeafObject,
    FMinDistanceToBox fMinDistanceToBox,
    FDistanceToLeaf fDistanceToleaf,
    FDistanceUpperBound fDistanceUpperBound,
    FOnFound fOnFound)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.geometry.Bvh.RangeSearch");

    using TQuery = std::invoke_result_t<FGetQueryObject, GpuIndex>;
    using TLeaf  = std::invoke_result_t<FGetLeafObject, GpuIndex, GpuIndex>;
    using Point  = pbat::math::linalg::mini::SVector<GpuScalar, kDims>;
    static_assert(
        std::is_invocable_v<FGetQueryObject, GpuIndex> and not std::is_same_v<TQuery, void>,
        "FGetQueryObject must be callable with signature TQuery f(GpuIndex)");
    static_assert(
        std::is_invocable_v<FGetLeafObject, GpuIndex, GpuIndex> and not std::is_same_v<TLeaf, void>,
        "FGetLeafObject must be callable with signature TLeaf f(GpuIndex,GpuIndex)");
    static_assert(
        std::is_invocable_v<FMinDistanceToBox, TQuery, Point, Point> and
            std::is_convertible_v<
                std::invoke_result_t<FMinDistanceToBox, TQuery, Point, Point>,
                GpuScalar>,
        "FMinDistanceToBox must be callable with signature GpuScalar f(TQuery, Point, Point) where "
        "Point=pbat::math::linalg::mini::SVector<GpuScalar, 3>");
    static_assert(
        std::is_invocable_v<FDistanceToLeaf, TQuery, TLeaf> and
            std::is_convertible_v<std::invoke_result_t<FDistanceToLeaf, TQuery, TLeaf>, GpuScalar>,
        "FDistanceToLeaf must be callable with signature GpuScalar f(TQuery, TLeaf)");
    static_assert(
        std::is_invocable_v<FDistanceUpperBound, GpuIndex> and
            std::is_convertible_v<std::invoke_result_t<FDistanceUpperBound, GpuIndex>, GpuScalar>,
        "FDistanceUpperBound must be callable with signature GpuScalar f(GpuIndex)");
    static_assert(
        std::is_invocable_v<FOnFound, GpuIndex, TQuery, GpuIndex, GpuIndex, TLeaf, GpuScalar>,
        "FOnFound must be callable with signature void f(GpuIndex q, TQuery query, GpuIndex "
        "leafIdx, GpuIndex i, TLeaf leaf, GpuScalar d)");

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nQueries),
        [leafBegin           = aabbs.Size() - 1,
         inds                = inds.Raw(),
         b                   = aabbs.b.Raw(),
         e                   = aabbs.e.Raw(),
         ib                  = iaabbs.b.Raw(),
         ie                  = iaabbs.e.Raw(),
         child               = child.Raw(),
         fGetQueryObject     = std::forward<FGetQueryObject>(fGetQueryObject),
         fGetLeafObject      = std::forward<FGetLeafObject>(fGetLeafObject),
         fMinDistanceToBox   = std::forward<FMinDistanceToBox>(fMinDistanceToBox),
         fDistanceToLeaf     = std::forward<FDistanceToLeaf>(fDistanceToleaf),
         fDistanceUpperBound = std::forward<FDistanceUpperBound>(fDistanceUpperBound),
         fOnFound            = std::forward<FOnFound>(fOnFound)] PBAT_DEVICE(GpuIndex q) mutable {
            using pbat::math::linalg::mini::FromBuffers;
            using Stack = pbat::common::Stack<GpuIndex, kStackSize>;

            // Depth-first branch and bound search
            Stack dfs{};
            GpuScalar const upper{fDistanceUpperBound(q)};
            TQuery const query{fGetQueryObject(q)};

            dfs.Push(0 /*root*/);
            do
            {
                assert(not dfs.IsFull());
                GpuIndex nodeIdx = dfs.Top();
                dfs.Pop();
                bool const bIsLeafNode = nodeIdx >= leafBegin;
                if (not bIsLeafNode)
                {
                    auto L = FromBuffers<3, 1>(ib, nodeIdx);
                    auto U = FromBuffers<3, 1>(ie, nodeIdx);
                    if (fMinDistanceToBox(query, L, U) <= upper)
                    {
                        dfs.Push(child[0][nodeIdx]);
                        dfs.Push(child[1][nodeIdx]);
                    }
                }
                else
                {
                    nodeIdx -= leafBegin;
                    auto L             = FromBuffers<3, 1>(b, nodeIdx);
                    auto U             = FromBuffers<3, 1>(e, nodeIdx);
                    GpuScalar const db = fMinDistanceToBox(query, L, U);
                    if (db <= upper)
                    {
                        GpuIndex const i  = inds[nodeIdx];
                        TLeaf const leaf  = fGetLeafObject(nodeIdx, i);
                        GpuScalar const d = fDistanceToLeaf(query, leaf);
                        if (d <= upper)
                        {
                            fOnFound(q, query, nodeIdx, i, leaf, d);
                        }
                    }
                }
            } while (not dfs.IsEmpty());
        });

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVHIMPL_H
