#ifndef PBAT_GEOMETRY_NEARESTNEIGHBOURSEARCH_H
#define PBAT_GEOMETRY_NEARESTNEIGHBOURSEARCH_H

#include "pbat/Aliases.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/NAryTreeTraversal.h"
#include "pbat/common/Queue.h"
#include "pbat/common/Stack.h"

#include <algorithm>
#include <array>
#include <limits>
#include <queue>
#include <type_traits>

namespace pbat::geometry {

/**
 * @brief Find distance minimizing objects in branch and bound tree rooted in root
 *
 * @tparam FChild Callable with signature `template <auto c> Index(TIndex node)`
 * @tparam FIsLeaf Callable with signature `bool(TIndex node)`
 * @tparam FDistanceLowerBound Callable with signature `Scalar(TIndex node)`
 * @tparam FLeafSize Callable with signature `TIndex(TIndex node)`
 * @tparam FLeafObject Callable with signature `TIndex(TIndex node, TIndex i)`
 * @tparam FDistance Callable with signature `TScalar(TIndex o)`
 * @tparam FOnFound Callable with signature `void(Index o, Scalar d, Index k)`
 * @tparam N Max number of children per node
 * @tparam TScalar Type of the scalar distance
 * @tparam TIndex Type of the index
 * @tparam kStackDepth Maximum depth of the traversal's stack
 * @tparam kQueueSize Maximum size of the nearest neighbour queue
 * @param fChild Function to get child c of a node. Returns the child index or -1 if no child.
 * @param fIsLeaf Function to determine if a node is a leaf node
 * @param fLower Function to compute the lower bound of the distance to node
 * @param fLeafSize Function to get the number of leaf objects in a node
 * @param fLeafObject Function to get the i-th leaf object in a node
 * @param fDistance Function to compute the distance to object
 * @param fOnFound Function to call when a nearest neighbour is found
 * @param fUpper Upper bound of the distance to the nearest neighbour
 * @param eps Epsilon to consider objects at the same distance
 * @param root Index of the root node to start the search from
 */
template <
    class FChild,
    class FIsLeaf,
    class FDistanceLowerBound,
    class FLeafSize,
    class FLeafObject,
    class FDistance,
    class FOnFound,
    auto N           = 2,
    class TScalar    = Scalar,
    class TIndex     = Index,
    auto kStackDepth = 64,
    auto kQueueSize  = 8>
void NearestNeighbour(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FDistanceLowerBound fLower,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistance fDistance,
    FOnFound fOnFound,
    TScalar fUpper = std::numeric_limits<TScalar>::max(),
    TScalar eps    = TScalar(0),
    TIndex root    = 0);

/**
 * @brief Find the K distance minimizers in a branch and bound tree rooted in root
 *
 * @tparam FChild Callable with signature `template <auto c> Index(TIndex node)`
 * @tparam FIsLeaf Callable with signature `bool(TIndex node)`
 * @tparam FDistanceLowerBound Callable with signature `TScalar(TIndex node)`
 * @tparam FLeafSize Callable with signature `TIndex(TIndex node)`
 * @tparam FLeafObject Callable with signature `TIndex(TIndex node, TIndex i)`
 * @tparam FDistance Callable with signature `TScalar(TIndex o)`
 * @tparam FOnFound Callable with signature `void(TIndex o, TScalar d, TIndex k)`
 * @tparam N Max number of children per node
 * @tparam TScalar Type of the scalar distance
 * @tparam TIndex Type of the index
 * @param fChild Function to get child c of a node. Returns the child index or -1 if no child.
 * @param fIsLeaf Function to determine if a node is a leaf node
 * @param fLower Function to compute the lower bound of the distance to node
 * @param fLeafSize Function to get the number of leaf objects in a node
 * @param fLeafObject Function to get the i-th leaf object in a node
 * @param fDistance Function to compute the distance to object
 * @param fOnFound Function to call when a nearest neighbour is found
 * @param K Number of nearest neighbours to find
 * @param fUpper Upper bound of the distance to the nearest neighbour
 * @param root Index of the root node to start the search from
 */
template <
    class FChild,
    class FIsLeaf,
    class FDistanceLowerBound,
    class FLeafSize,
    class FLeafObject,
    class FDistance,
    class FOnFound,
    auto N        = 2,
    class TScalar = Scalar,
    class TIndex  = Index>
void KNearestNeighbours(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FDistanceLowerBound fLower,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistance fDistance,
    FOnFound fOnFound,
    TIndex K,
    TScalar fUpper = std::numeric_limits<TScalar>::max(),
    TIndex root    = 0);

template <
    class FChild,
    class FIsLeaf,
    class FDistanceLowerBound,
    class FLeafSize,
    class FLeafObject,
    class FDistance,
    class FOnFound,
    auto N,
    class TScalar,
    class TIndex,
    auto kStackDepth,
    auto kQueueSize>
void NearestNeighbour(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FDistanceLowerBound fLower,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistance fDistance,
    FOnFound fOnFound,
    TScalar fUpper,
    TScalar eps,
    TIndex root)
{
    common::Stack<TIndex, kStackDepth> dfs{};
    common::Queue<TIndex, kQueueSize> nn{};
    if (fLower(root) <= fUpper)
        dfs.Push(root);

    do
    {
        TIndex node = dfs.Pop();
        if (not fIsLeaf(node))
        {
            // Recurse into children unless minimum cannot be found in their subdomain
            common::ForRange<0, N>([&]<auto i> {
                auto const child = fChild.template operator()<i>(node);
                if (child >= 0)
                    if (fLower(child) <= fUpper)
                        dfs.Push(child);
            });
        }
        else
        {
            auto nLeafObjects = fLeafSize(node);
            for (auto i = 0; i < nLeafObjects; ++i)
            {
                TIndex o      = fLeafObject(node, i);
                auto const d  = fDistance(o);
                auto const lo = fUpper - eps;
                auto const hi = fUpper + eps;
                if (d < lo)
                {
                    nn.Clear();
                    nn.Push(o);
                    fUpper = d;
                }
                else if (d <= hi and not nn.IsFull())
                {
                    nn.Push(o);
                }
            }
        }
    } while (not dfs.IsEmpty());
    TIndex k{0};
    while (not nn.IsEmpty())
    {
        TIndex o = nn.Top();
        nn.Pop();
        fOnFound(o, fUpper, k++);
    }
}

template <
    class FChild,
    class FIsLeaf,
    class FDistanceLowerBound,
    class FLeafSize,
    class FLeafObject,
    class FDistance,
    class FOnFound,
    auto N,
    class TScalar,
    class TIndex>
void KNearestNeighbours(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FDistanceLowerBound fLower,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistance fDistance,
    FOnFound fOnFound,
    TIndex K,
    TScalar fUpper,
    TIndex root)
{
    enum class EQueueItem { Node, Object };
    struct QueueItem
    {
        EQueueItem type; // Indicates if this QueueItem holds a primitive or a volume
        TIndex idx;      // Index of the primitive, if this QueueItem holds a primitive, or index
                         // of the node, if this QueueItem holds a volume (recall that node_idx =
                         // bv_idx + 1)
        TScalar d;       // Distance from this QueueItem to p
    };
    auto const fMakeNodeQueueItem = [&](TIndex node) {
        TScalar const d = fLower(node);
        QueueItem const q{EQueueItem::Node, node, d};
        return q;
    };
    auto const fMakeObjectQueueItem = [&](TIndex object) {
        TScalar const d = fDistance(object);
        QueueItem const q{EQueueItem::Object, object, d};
        return q;
    };
    auto const fGreater = [](QueueItem const& q1, QueueItem const& q2) {
        return q1.d > q2.d;
    };
    using Comparator    = decltype(fGreater);
    using Container     = std::vector<QueueItem>;
    using PriorityQueue = std::priority_queue<QueueItem, Container, Comparator>;
    Container allocation{};
    allocation.reserve(64 * N);
    PriorityQueue heap{fGreater, std::move(allocation)};
    heap.push(fMakeNodeQueueItem(root));
    TIndex k{0};
    while (not heap.empty())
    {
        QueueItem const q = heap.top();
        heap.pop();
        if (q.d > fUpper)
            break;
        if (q.type == EQueueItem::Node)
        {
            auto node = q.idx;
            if (fIsLeaf(node))
            {
                auto const nLeafObjects = fLeafSize(node);
                for (auto i = 0; i < nLeafObjects; ++i)
                {
                    TIndex const o = fLeafObject(node, i);
                    heap.push(fMakeObjectQueueItem(o));
                }
            }
            else
            {
                common::ForRange<0, N>([&]<auto i> {
                    auto const child = fChild.template operator()<i>(node);
                    if (child >= 0)
                        heap.push(fMakeNodeQueueItem(child));
                });
            }
        }
        else
        {
            fOnFound(q.idx, q.d, k++);
            if (k == K)
                break;
        }
    }
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_NEARESTNEIGHBOURSEARCH_H
