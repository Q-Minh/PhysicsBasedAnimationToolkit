/**
 * @file BoundingVolumeHierarchy.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Bounding volume hierarchy (BVH) implementation for spatial partitioning of primitives.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_BOUNDINGVOLUMEHIERARCHY_H
#define PBAT_GEOMETRY_BOUNDINGVOLUMEHIERARCHY_H

#include "KdTree.h"

#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <queue>
#include <stack>
#include <tbb/parallel_for.h>
#include <utility>
#include <vector>

namespace pbat {
namespace geometry {

/**
 * @brief CRTP base class for BVHs.
 *
 * @tparam TDerived Type of the child class (the concrete BVH implementation)
 * @tparam TBoundingVolume Type of bounding volumes used in the BVH tree
 * @tparam TPrimitive Type of primitives stored in the BVH
 * @tparam Dims Embedding dimensionality
 */
template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
class BoundingVolumeHierarchy
{
  public:
    using DerivedType           = TDerived;        ///< Actual type
    using BoundingVolumeType    = TBoundingVolume; ///< Type of bounding volumes
    using PrimitiveType         = TPrimitive;      ///< Type of primitives
    static auto constexpr kDims = Dims;            ///< Embedding dimensionality

    template <class TDerived2, class TBoundingVolume2, class TPrimitive2, int Dims2>
    friend class BoundingVolumeHierarchy;

    BoundingVolumeHierarchy() = default;

    /**
     * @brief Construct the BVH from a set of primitives
     * @param nPrimitives Number of primitives
     * @param maxPointsInLeaf Maximum number of primitives in a leaf node
     */
    void Construct(Index nPrimitives, Index maxPointsInLeaf = 10);
    /**
     * @brief Returns the bounding volumes of this BVH
     * @return Bounding volumes
     */
    auto BoundingVolumes() const -> std::vector<BoundingVolumeType> const&
    {
        return mBoundingVolumes;
    }
    /**
     * @brief Returns the indices of the primitives contained in the bounding volume bvIdx
     * @param bvIdx Index of the bounding volume
     * @return Range of indices of the primitives
     */
    auto PrimitivesInBoundingVolume(Index bvIdx) const;
    /**
     * @brief Returns the indices of the primitives intersecting the bounding volume bv
     * @tparam FIntersectsBoundingVolume Callable with signature `bool pred(BoundingVolume const&)`
     * @tparam FIntersectsPrimitive Callable with signature `bool pred(Primitive const&)`
     * primitive `p` is intersected.
     * @param ibv Predicate pred(bv) evaluating to true if the bounding volume bv is intersected.
     * @param ip Predicate pred(p) evaluating to true if the primitive p is intersected.
     * @param reserve Estimated number of intersecting primitives to reserve in memory
     * @return
     */
    template <class FIntersectsBoundingVolume, class FIntersectsPrimitive>
    std::vector<Index> PrimitivesIntersecting(
        FIntersectsBoundingVolume&& ibv,
        FIntersectsPrimitive&& ip,
        std::size_t reserve = 50ULL) const;
    /**
     * @brief Obtains the k nearest neighbours (primitives of this BVH)
     * @tparam FDistanceToBoundingVolume Callable with signature `Scalar pred(BoundingVolume
     * const&)`
     * @tparam FDistanceToPrimitive Callable with signature `Scalar pred(Primitive const&)`
     * @param db Distance function d(b) between bounding volume b and user-owned shape
     * @param dp Distance function d(p) between primitive p and user-owned shape
     * @param K Number of nearest neighbours to query
     * @return Pair of vectors containing the indices of the nearest primitives and their
     * distances
     */
    template <class FDistanceToBoundingVolume, class FDistanceToPrimitive>
    auto
    NearestPrimitivesTo(FDistanceToBoundingVolume&& db, FDistanceToPrimitive&& dp, std::size_t K)
        const -> std::pair<std::vector<Index>, std::vector<Scalar>>;

    /**
     * @brief Update the bounding volumes of this BVH
     */
    void Update();

    // Static virtual functions (CRTP)

    /**
     * @brief Returns the primitive at index p
     * @note This function must be implemented by the derived class
     * @param p Index of the primitive
     * @return Primitive at index p
     */
    PrimitiveType Primitive(Index p) const
    {
        return static_cast<TDerived const*>(this)->Primitive(p);
    }
    /**
     * @brief Returns the location of the primitive
     * @note This function must be implemented by the derived class
     * @param primitive Primitive
     * @return Location of the primitive
     */
    auto PrimitiveLocation(PrimitiveType const& primitive) const
    {
        return static_cast<TDerived const*>(this)->PrimitiveLocation(primitive);
    }
    /**
     * @brief Returns the bounding volume of the primitives in the range [first, last)
     * @note This function must be implemented by the derived class
     * @tparam RPrimitiveIndices Range of primitive indices
     * @param primitiveIndexRange Range of primitive indices
     * @return Bounding volume of the primitives
     */
    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& primitiveIndexRange) const
    {
        return static_cast<TDerived const*>(this)->BoundingVolumeOf(primitiveIndexRange);
    }

  protected:
    /**
     * @brief Returns the indices of the primitives overlapping between this BVH and another BVH
     * @tparam TDerived2 Type of the other BVH
     * @tparam TBoundingVolume2 Type of bounding volumes of the other BVH
     * @tparam TPrimitive2 Type of primitives of the other BVH
     * @tparam FBoundingVolumesOverlap Callable with signature `bool pred(BoundingVolume const&,
     * BoundingVolume2 const&)`
     * @tparam FPrimitivesOverlap Callable with signature `bool pred(Primitive const&, Primitive2
     * const&)`
     * @tparam FPrimitivesAreAdjacent Callable with signature `bool pred(Primitive const&,
     * Primitive2 const&)`
     * @tparam Dims2 Embedding dimensionality of the other BVH
     * @param other Other BVH
     * @param bvo Predicate pred(bv1, bv2) evaluating to true if the bounding volumes bv1 and bv2
     * overlap
     * @param po Predicate pred(p1, p2) evaluating to true if the primitives p1 and p2 overlap
     * @param PrimitivesAreAdjacent Predicate pred(p1, p2) evaluating to true if the primitives p1
     * and p2 are adjacent
     * @param reserve Estimated number of overlapping primitives to reserve in memory
     * @return `2 x |# overlaps|` matrix of overlapping primitive indices
     *
     */
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

    std::vector<BoundingVolumeType> mBoundingVolumes; ///< Bounding volumes of the BVH
    KdTree<kDims> mKdTree; ///< K-d tree used to store the primitives and the BVH tree
};

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
inline void BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::Construct(
    Index nPrimitives,
    Index maxPointsInLeaf)
{
    Matrix<Dims, Eigen::Dynamic> P(Dims, nPrimitives);
    for (auto p = 0; p < P.cols(); ++p)
    {
        P.col(p) = PrimitiveLocation(Primitive(p));
    }
    mKdTree.Construct(P, maxPointsInLeaf);
    std::size_t const nBoundingVolumes = mKdTree.Nodes().size();
    mBoundingVolumes.resize(nBoundingVolumes);
    tbb::parallel_for(std::size_t{0ULL}, nBoundingVolumes, [this](std::size_t b) {
        mBoundingVolumes[b] = BoundingVolumeOf(mKdTree.PointsInNode(static_cast<Index>(b)));
    });
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
inline auto
BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::PrimitivesInBoundingVolume(
    Index bvIdx) const
{
    std::size_t const bvIdxStl = static_cast<std::size_t>(bvIdx);
    return mKdTree.PointsInNode(mKdTree.Nodes()[bvIdxStl]);
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
inline void BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::Update()
{
    auto const& nodes = mKdTree.Nodes();
    tbb::parallel_for(std::size_t{0ULL}, nodes.size(), [&](std::size_t bvIdx) {
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
            return false; // Cannot visit deeper than a leaf node
        }
        else
        {
            auto const bvIdxStl          = static_cast<std::size_t>(bvIdx);
            BoundingVolumeType const& bv = mBoundingVolumes[bvIdxStl];
            return ibv(bv); // Visit deeper if this bounding volume overlaps with the
                            // queried shape
        }
    });

    return intersectingPrimitives;
}

template <class TDerived, class TBoundingVolume, class TPrimitive, int Dims>
template <class FDistanceToBoundingVolume, class FDistanceToPrimitive>
inline auto
BoundingVolumeHierarchy<TDerived, TBoundingVolume, TPrimitive, Dims>::NearestPrimitivesTo(
    FDistanceToBoundingVolume&& db,
    FDistanceToPrimitive&& dp,
    std::size_t K) const -> std::pair<std::vector<Index>, std::vector<Scalar>>
{
    std::vector<Index> neighbours{};
    std::vector<Scalar> distances{};
    neighbours.reserve(K);
    distances.reserve(K);

    enum class EQueueItem { Volume, Primitive };
    struct QueueItem
    {
        EQueueItem type; // Indicates if this QueueItem holds a primitive or a volume
        Index idx;       // Index of the primitive, if this QueueItem holds a primitive, or index
                         // of the node, if this QueueItem holds a volume (recall that node_idx =
                         // bv_idx + 1)
        Scalar d;        // Distance from this QueueItem to p
    };
    auto const MakeVolumeQueueItem = [&](Index bvIdx) {
        auto const bvIdxStl          = static_cast<std::size_t>(bvIdx);
        BoundingVolumeType const& bv = mBoundingVolumes[bvIdxStl];
        Scalar const d               = db(bv);
        QueueItem const q{EQueueItem::Volume, bvIdx, d};
        return q;
    };
    auto const MakePrimitiveQueueItem = [&](Index pIdx) {
        PrimitiveType const& p = Primitive(pIdx);
        Scalar const d         = dp(p);
        QueueItem const q{EQueueItem::Primitive, pIdx, d};
        return q;
    };

    auto const Greater = [](QueueItem const& q1, QueueItem const& q2) {
        return q1.d > q2.d;
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
            distances.push_back(q.d);
        }

        if (neighbours.size() == K)
            break;
    }
    return {neighbours, distances};
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
    stack.push({0, 0}); // Root bounding volumes of *this and other
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
            auto const n1n = node1.n;
            auto const n2n = node2.n;
            if (n1n > n2n)
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

#endif // PBAT_GEOMETRY_BOUNDINGVOLUMEHIERARCHY_H
