#ifndef PBAT_GEOMETRY_BOUNDING_VOLUME_HIERARCHY_H
#define PBAT_GEOMETRY_BOUNDING_VOLUME_HIERARCHY_H

#include "KdTree.h"

#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <queue>
#include <ranges>
#include <stack>
#include <tbb/parallel_for.h>
#include <vector>

namespace pbat {
namespace geometry {

/**
 * @brief CRTP base class for BVHs.
 *
 * @tparam TDerived Type of the child class (the concrete BVH implementation)
 * @tparam TPrimitive Type of primitives stored in the BVH
 * @tparam TBoundingVolume Type of bounding volumes used in the BVH tree
 * @tparam Dims Embedding dimensionality
 */
template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
class BoundingVolumeHierarchy
{
  public:
    using DerivedType           = TDerived;
    using BoundingVolumeType    = TBoundingVolume;
    using PrimitiveType         = TPrimitive;
    static auto constexpr kDims = Dims;

    template <class TDerived2, class TBoundingVolume2, class TPrimitive2, int Dims2>
    friend class BoundingVolumeHierarchy;

    BoundingVolumeHierarchy() = default;

    /**
     * @brief
     * @param maxPointsInLeaf
     */
    void Construct(std::size_t nPrimitives, std::size_t maxPointsInLeaf = 10u);
    /**
     * @brief
     * @return
     */
    std::vector<BoundingVolumeType> const& BoundingVolumes() const { return mBoundingVolumes; }
    /**
     * @brief Returns the indices of the primitives contained in the bounding volume bvIdx
     * @param bvIdx
     * @return
     */
    auto PrimitivesInBoundingVolume(Index bvIdx) const;
    /**
     * @brief
     * @param ibv Predicate pred(bv) evaluating to true if the bounding volume bv is intersected.
     * @param ip Predicate pred(p) evaluating to true if the primitive p is intersected.
     * @return
     */
    template <class FIntersectsBoundingVolume, class FIntersectsPrimitive>
    std::vector<Index> PrimitivesIntersecting(
        FIntersectsBoundingVolume&& ibv,
        FIntersectsPrimitive&& ip,
        std::size_t reserve = 50ULL) const;
    /**
     * @brief Obtains the k nearest neighbours (primitives of this BVH)
     * @tparam T
     * @param db Distance function d(b) between bounding volume b and user-owned shape
     * @param dp Distance function d(p) between primitive p and user-owned shape
     * @param K Number of nearest neighbours to query
     * @return
     */
    template <class FDistanceToBoundingVolume, class FDistanceToPrimitive>
    std::vector<Index>
    NearestPrimitivesTo(FDistanceToBoundingVolume&& db, FDistanceToPrimitive&& dp, std::size_t K)
        const;

    /**
     * @brief Update the bounding volumes of this BVH
     */
    void Update();

    // Static virtual functions (CRTP)
    PrimitiveType Primitive(Index p) const
    {
        return static_cast<TDerived const*>(this)->Primitive(p);
    }
    auto PrimitiveLocation(PrimitiveType const& primitive) const
    {
        return static_cast<TDerived const*>(this)->PrimitiveLocation(primitive);
    }
    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& primitiveIndexRange) const
    {
        return static_cast<TDerived const*>(this)->BoundingVolumeOf(primitiveIndexRange);
    }

  protected:
    template <
        class TDerived2,
        class TBoundingVolume2,
        class TPrimitive2,
        int Dims2,
        class FBoundingVolumesOverlap,
        class FPrimitivesOverlap,
        class FPrimitivesAreAdjacent>
    IndexMatrixX OverlappingPrimitivesImpl(
        BoundingVolumeHierarchy<TDerived2, TBoundingVolume2, TPrimitive2, Dims2> const& other,
        FBoundingVolumesOverlap&& bvo,
        FPrimitivesOverlap&& po,
        FPrimitivesAreAdjacent&& PrimitivesAreAdjacent =
            [](PrimitiveType const& p1, TPrimitive2 const& p2) -> bool { return false; },
        std::size_t reserve = 50ULL) const;

    std::vector<BoundingVolumeType> mBoundingVolumes;
    KdTree<kDims> mKdTree;
};

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
inline void BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::Construct(
    std::size_t nPrimitives,
    std::size_t maxPointsInLeaf)
{
    Matrix<Dims, Eigen::Dynamic> P(Dims, nPrimitives);
    for (auto p = 0; p < P.cols(); ++p)
    {
        P.col(p) = PrimitiveLocation(Primitive(p));
    }
    mKdTree.Construct(P, maxPointsInLeaf);
    namespace rng = std::ranges;
    namespace vi  = std::views;
    auto bvRange  = mKdTree.Nodes() | vi::transform([this](KdTreeNode const& node) {
                       return BoundingVolumeOf(mKdTree.PointsInNode(node));
                   });
    mBoundingVolumes.assign(rng::begin(bvRange), rng::end(bvRange));
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
inline auto
BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::PrimitivesInBoundingVolume(
    Index bvIdx) const
{
    return mKdTree.PointsInNode(mKdTree.Nodes()[bvIdx]);
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
inline void BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::Update()
{
    auto const& nodes = mKdTree.Nodes();
    tbb::parallel_for(std::size_t{0ULL}, nodes.size(), [this](std::size_t bvIdx) {
        KdTreeNode const& node  = nodes[bvIdx];
        mBoundingVolumes[bvIdx] = BoundingVolumeOf(mKdTree.PointsInNode(node));
    });
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
template <class FIntersectsBoundingVolume, class FIntersectsPrimitive>
inline std::vector<Index>
BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::PrimitivesIntersecting(
    FIntersectsBoundingVolume&& ibv,
    FIntersectsPrimitive&& ip,
    std::size_t reserve) const
{
    std::vector<Index> intersectingPrimitives{};
    intersectingPrimitives.reserve(reserve);

    mKdTree.DepthFirstSearch([&](Index bvIdx, KdTreeNode const& node) -> bool {
        if (node.IsLeafNode())
        {
            for (auto const idx : mKdTree.PointsInNode(node))
            {
                bool const bIntersects = ip(Primitive(idx));
                if (bIntersects)
                {
                    intersectingPrimitives.push_back(idx);
                }
            }
            return false; ///< Cannot visit deeper than a leaf node
        }
        else
        {
            auto const bvIdxStl          = static_cast<std::size_t>(bvIdx);
            BoundingVolumeType const& bv = mBoundingVolumes[bvIdxStl];
            return ibv(bv); ///< Visit deeper if this bounding volume overlaps with the
                            ///< queried shape
        }
    });

    return intersectingPrimitives;
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
template <class FDistanceToBoundingVolume, class FDistanceToPrimitive>
inline std::vector<Index>
BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::NearestPrimitivesTo(
    FDistanceToBoundingVolume&& db,
    FDistanceToPrimitive&& dp,
    std::size_t K) const
{
    std::vector<Index> neighbours{};
    neighbours.reserve(K);

    enum class EQueueItem { Volume, Primitive };
    struct QueueItem
    {
        EQueueItem type; ///< Indicates if this QueueItem holds a primitive or a volume
        Index idx;       ///< Index of the primitive, if this QueueItem holds a primitive, or index
                         ///< of the node, if this QueueItem holds a volume (recall that node_idx =
                         ///< bv_idx + 1)
        Scalar sd;       ///< Squared distance from this QueueItem to p
    };
    auto const MakeVolumeQueueItem = [&](Index bvIdx) {
        auto const bvIdxStl          = static_cast<std::size_t>(bvIdx);
        BoundingVolumeType const& bv = mBoundingVolumes[bvIdxStl];
        Scalar const sd              = db(bv);
        QueueItem const q{EQueueItem::Volume, bvIdx, sd};
        return q;
    };
    auto const MakePrimitiveQueueItem = [&](Index pIdx) {
        PrimitiveType const& p = Primitive(pIdx);
        Scalar const sd        = dp(p);
        QueueItem const q{EQueueItem::Primitive, pIdx, sd};
        return q;
    };

    auto const Greater = [](QueueItem const& q1, QueueItem const& q2) {
        return q1.sd > q2.sd;
    };
    using PriorityQueue = std::priority_queue<QueueItem, std::vector<QueueItem>, decltype(Greater)>;
    PriorityQueue queue{Greater};
    queue.push(MakeVolumeQueueItem(0));
    auto const& nodes = mKdTree.Nodes();
    while (!queue.empty())
    {
        QueueItem const q = queue.top();
        queue.pop();
        if (q.type == EQueueItem::Volume)
        {
            auto const qIdxStl     = static_cast<std::size_t>(q.idx);
            KdTreeNode const& node = nodes[qIdxStl];
            if (node.IsLeafNode())
            {
                for (auto const pIdx : mKdTree.PointsInNode(node))
                {
                    queue.push(MakePrimitiveQueueItem(pIdx));
                }
            }
            else
            {
                if (node.HasLeftChild())
                    queue.push(MakeVolumeQueueItem(node.lc));
                if (node.HasRightChild())
                    queue.push(MakeVolumeQueueItem(node.rc));
            }
        }
        else
        {
            // If the queue item at the beginning of the priority queue is a primitive,
            // it means that primitive is closer to the point p than any other primitive or
            // bounding volume in the queue, thus it is the current closest primitive to p.
            neighbours.push_back(q.idx);
        }

        if (neighbours.size() == K)
            break;
    }
    return neighbours;
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
template <
    class TDerived2,
    class TBoundingVolume2,
    class TPrimitive2,
    int Dims2,
    class FBoundingVolumesOverlap,
    class FPrimitivesOverlap,
    class FPrimitivesAreAdjacent>
inline IndexMatrixX
BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::OverlappingPrimitivesImpl(
    BoundingVolumeHierarchy<TDerived2, TBoundingVolume2, TPrimitive2, Dims2> const& other,
    FBoundingVolumesOverlap&& bvo,
    FPrimitivesOverlap&& po,
    FPrimitivesAreAdjacent&& pa,
    std::size_t reserve) const
{
    using BoundingVolumeType2 = TBoundingVolume2;
    using PrimitiveType2      = TPrimitive2;
    std::vector<Index> overlaps{};
    overlaps.reserve(reserve * 2);
    auto const& nodes1 = mKdTree.Nodes();
    auto const& nodes2 = other.mKdTree.Nodes();

    using PrimitivePairType = std::pair<Index, Index>;
    std::stack<PrimitivePairType> stack{};
    stack.push({0, 0}); ///< Root bounding volumes of *this and other
    while (!stack.empty())
    {
        auto const [n1, n2] = stack.top();
        auto const n1Stl    = static_cast<std::size_t>(n1);
        auto const n2Stl    = static_cast<std::size_t>(n2);
        stack.pop();
        BoundingVolumeType const& bv1  = mBoundingVolumes[n1Stl];
        BoundingVolumeType2 const& bv2 = other.mBoundingVolumes[n2Stl];
        if (!bvo(bv1, bv2))
            continue;

        KdTreeNode const& node1 = nodes1[n1Stl];
        KdTreeNode const& node2 = nodes2[n2Stl];
        bool const bIsNode1Leaf = node1.IsLeafNode();
        bool const bIsNode2Leaf = node2.IsLeafNode();
        if (bIsNode1Leaf and bIsNode2Leaf)
        {
            for (auto const p1 : mKdTree.PointsInNode(node1))
            {
                for (auto const p2 : other.mKdTree.PointsInNode(node2))
                {
                    // For self collision, prevent expensive narrow phase testing when mesh
                    // primitives are adjacent, since this gives false positives
                    if (pa(Primitive(p1), other.Primitive(p2)))
                        continue;

                    if (po(Primitive(p1), other.Primitive(p2)))
                    {
                        overlaps.push_back(p1);
                        overlaps.push_back(p2);
                    }
                }
            }
        }
        else if (bIsNode1Leaf and not bIsNode2Leaf)
        {
            if (node2.HasLeftChild())
                stack.push({n1, node2.lc});
            if (node2.HasRightChild())
                stack.push({n1, node2.rc});
        }
        else if (not bIsNode1Leaf and bIsNode2Leaf)
        {
            if (node1.HasLeftChild())
                stack.push({node1.lc, n2});
            if (node1.HasRightChild())
                stack.push({node1.rc, n2});
        }
        else
        {
            if (node1.depth < node2.depth)
            {
                if (node1.HasLeftChild())
                    stack.push({node1.lc, n2});
                if (node1.HasRightChild())
                    stack.push({node1.rc, n2});
            }
            else
            {
                if (node2.HasLeftChild())
                    stack.push({n1, node2.lc});
                if (node2.HasRightChild())
                    stack.push({n1, node2.rc});
            }
        }
    }
    return common::ToEigen(overlaps).reshaped(2, overlaps.size() / 2);
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_BOUNDING_VOLUME_HIERARCHY_H
