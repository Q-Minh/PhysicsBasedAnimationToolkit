/**
 * @file SpatialSearch.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Generic efficient spatial search query implementations
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_GEOMETRY_SPATIALSEARCH_H
#define PBAT_GEOMETRY_SPATIALSEARCH_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Heap.h"
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
 * @tparam FLeafSize Callable with signature `TIndex(TIndex node)`
 * @tparam FLeafObject Callable with signature `TIndex(TIndex node, TIndex i)`
 * @tparam FNodeOverlaps Callable with signature `bool(TIndex node)`
 * @tparam FObjectOverlaps Callable with signature `bool(TIndex o)`
 * @tparam FOnFound Callable with signature `void(Index o, Index k)`
 * @tparam N Max number of children per node
 * @tparam TScalar Type of the scalar distance
 * @tparam TIndex Type of the index
 * @tparam kStackDepth Maximum depth of the traversal's stack
 * @tparam kQueueSize Maximum size of the nearest neighbour queue
 * @param fChild Function to get child c of a node. Returns the child index or -1 if no child.
 * @param fIsLeaf Function to determine if a node is a leaf node
 * @param fLeafSize Function to get the number of leaf objects in a node
 * @param fLeafObject Function to get the i-th leaf object in a node
 * @param fNodeOverlaps Function to determine if a node overlaps with the query
 * @param fObjectOverlaps Function to determine if an object overlaps with the query
 * @param fOnFound Function to call when a nearest neighbour is found
 * @param root Index of the root node to start the search from
 */
template <
    class FChild,
    class FIsLeaf,
    class FLeafSize,
    class FLeafObject,
    class FNodeOverlap,
    class FObjectOverlap,
    class FOnFound,
    auto N           = 2,
    class TScalar    = Scalar,
    class TIndex     = Index,
    auto kStackDepth = 64>
PBAT_HOST_DEVICE void Overlaps(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FNodeOverlap fNodeOverlaps,
    FObjectOverlap fObjectOverlaps,
    FOnFound fOnFound,
    TIndex root = 0);

/**
 * @brief Find distance minimizing objects in branch and bound tree rooted in root
 *
 * @note Although it's possible to obtain nearest neighbour(s) using the KNearestNeighbours
 * function, we also expose this function which focuses on finding the distance minimizer. Both
 * functions are branch and bound tree traversal algorithms. This function exposes 2 variants of the
 * traversal:
 * 1. Pre-order traversal: This is the default traversal method. It visits the children of a node in
 *   the order they are stored in the branch and bound tree, i.e. natural order, or random order,
 * using depth-first search. This has the advantage of using fewer CPU-cycles per node visit, but
 * might require more visits than a guided traversal.
 * 2. Best-first search: This is an alternative traversal method. At each node, it sorts the
 * children by their lower bounds and visits them in that order. The sort costs non-trivially more
 * stack memory and CPU cycles, but might require fewer visits than a natural order depth-first
 * traversal.
 *
 * @note Another approach for tweaking the performance of the NN search is to provide a tight
 * initial upper bound fUpper, which will help prune most of the search space.
 *
 * @tparam FChild Callable with signature `template <auto c> Index(TIndex node)`
 * @tparam FIsLeaf Callable with signature `bool(TIndex node)`
 * @tparam FLeafSize Callable with signature `TIndex(TIndex node)`
 * @tparam FLeafObject Callable with signature `TIndex(TIndex node, TIndex i)`
 * @tparam FDistanceLowerBound Callable with signature `Scalar(TIndex node)`
 * @tparam FDistance Callable with signature `TScalar(TIndex o)`
 * @tparam FOnFound Callable with signature `void(Index o, Scalar d, Index k)`
 * @tparam N Max number of children per node
 * @tparam TScalar Type of the scalar distance
 * @tparam TIndex Type of the index
 * @tparam kStackDepth Maximum depth of the traversal's stack
 * @tparam kQueueSize Maximum size of the nearest neighbour queue
 * @param fChild Function to get child c of a node. Returns the child index or -1 if no child.
 * @param fIsLeaf Function to determine if a node is a leaf node
 * @param fLeafSize Function to get the number of leaf objects in a node
 * @param fLeafObject Function to get the i-th leaf object in a node
 * @param fLower Function to compute the lower bound of the distance to node
 * @param fDistance Function to compute the distance to object
 * @param fOnFound Function to call when a nearest neighbour is found
 * @param bUseBestFirstSearch Use best-first search instead of pre-order traversal
 * @param fUpper Upper bound of the distance to the nearest neighbour
 * @param eps Epsilon to consider objects at the same distance
 * @param root Index of the root node to start the search from
 */
template <
    class FChild,
    class FIsLeaf,
    class FLeafSize,
    class FLeafObject,
    class FDistanceLowerBound,
    class FDistance,
    class FOnFound,
    auto N           = 2,
    class TScalar    = Scalar,
    class TIndex     = Index,
    auto kStackDepth = 64,
    auto kQueueSize  = 8>
PBAT_HOST_DEVICE void NearestNeighbour(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistanceLowerBound fLower,
    FDistance fDistance,
    FOnFound fOnFound,
    bool bUseBestFirstSearch = false,
    TScalar fUpper           = std::numeric_limits<TScalar>::max(),
    TScalar eps              = TScalar(0),
    TIndex root              = 0);

/**
 * @brief Find the K distance minimizers in a branch and bound tree rooted in root
 *
 * @note This function is a branch and bound tree traversal algorithm, similar to NearestNeighbour.
 * However, we maintain a min-distance-heap of all visited nodes so far, whose space complexities
 * hould scale linearly w.r.t. the height of the tree, i.e. O(log(n)), where n is the number of
 * nodes in the tree. By always visiting nodes closest to the query first, we quickly restrict the
 * upper bound, pruning most of the search space. Each node visit costs approximately O(log(log(n)),
 * due to heap insertions and deletions, but the total number of visits is expected to be much
 * smaller than a generic depth-first traversal. However, the stack memory requirements are higher,
 * due to storing node indices, distances, and types (i.e. node or leaf object).
 *
 * @tparam FChild Callable with signature `template <auto c> Index(TIndex node)`
 * @tparam FIsLeaf Callable with signature `bool(TIndex node)`
 * @tparam FLeafSize Callable with signature `TIndex(TIndex node)`
 * @tparam FLeafObject Callable with signature `TIndex(TIndex node, TIndex i)`
 * @tparam FDistanceLowerBound Callable with signature `TScalar(TIndex node)`
 * @tparam FDistance Callable with signature `TScalar(TIndex o)`
 * @tparam FOnFound Callable with signature `void(TIndex o, TScalar d, TIndex k)`
 * @tparam N Max number of children per node
 * @tparam TScalar Type of the scalar distance
 * @tparam TIndex Type of the index
 * @param fChild Function to get child c of a node. Returns the child index or -1 if no child.
 * @param fIsLeaf Function to determine if a node is a leaf node
 * @param fLeafSize Function to get the number of leaf objects in a node
 * @param fLeafObject Function to get the i-th leaf object in a node
 * @param fLower Function to compute the lower bound of the distance to node
 * @param fDistance Function to compute the distance to object
 * @param fOnFound Function to call when a nearest neighbour is found
 * @param K Number of nearest neighbours to find
 * @param fUpper Upper bound of the distance to the nearest neighbour
 * @param root Index of the root node to start the search from
 */
template <
    class FChild,
    class FIsLeaf,
    class FLeafSize,
    class FLeafObject,
    class FDistanceLowerBound,
    class FDistance,
    class FOnFound,
    auto N             = 2,
    class TScalar      = Scalar,
    class TIndex       = Index,
    auto kHeapCapacity = 64>
PBAT_HOST_DEVICE void KNearestNeighbours(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistanceLowerBound fLower,
    FDistance fDistance,
    FOnFound fOnFound,
    TIndex K,
    TScalar fUpper = std::numeric_limits<TScalar>::max(),
    TIndex root    = 0);

template <
    class FChild,
    class FIsLeaf,
    class FLeafSize,
    class FLeafObject,
    class FNodeOverlap,
    class FObjectOverlap,
    class FOnFound,
    auto N,
    class TScalar,
    class TIndex,
    auto kStackDepth>
void Overlaps(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FNodeOverlap fNodeOverlaps,
    FObjectOverlap fObjectOverlaps,
    FOnFound fOnFound,
    TIndex root)
{
    Index k{0};
    auto fVisit = [&] PBAT_HOST_DEVICE(TIndex node) {
        if (not fNodeOverlaps(node))
            return false;
        if (fIsLeaf(node))
        {
            TIndex const nLeafObjects = fLeafSize(node);
            for (TIndex i = 0; i < nLeafObjects; ++i)
            {
                TIndex o = fLeafObject(node, i);
                if (fObjectOverlaps(o))
                    fOnFound(o, k++);
            }
        }
        return true;
    };
    // NOTE:
    // Instead of using a pre-order traversal, we visit children of the currently visited node
    // simultaneously. This is because the branch and bound tree might store each node's
    // children in contiguous memory, or at least with some notion of locality. Thus, the fLower
    // function will benefit from many cache hits.
    // using FVisit = decltype(fVisit);
    // common::TraverseNAryTreePseudoPreOrder<FVisit, FChild, TIndex, N, kStackDepth>(
    //     fVisit,
    //     fChild,
    //     root);
    common::Stack<TIndex, kStackDepth> stack{};
    if (not fVisit(root))
        return;
    stack.Push(root);
    while (not stack.IsEmpty())
    {
        TIndex const node = stack.Pop();
        common::ForRange<0, N>([&]<auto i> {
            TIndex const child = fChild.template operator()<i>(node);
            if (child >= 0)
                if (fVisit(child))
                    stack.Push(child);
        });
    }
}

template <
    class FChild,
    class FIsLeaf,
    class FLeafSize,
    class FLeafObject,
    class FDistanceLowerBound,
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
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistanceLowerBound fLower,
    FDistance fDistance,
    FOnFound fOnFound,
    bool bUseBestFirstSearch,
    TScalar fUpper,
    TScalar eps,
    TIndex root)
{
    common::Queue<TIndex, kQueueSize> nn{};
    auto fDoVisit = [&] PBAT_HOST_DEVICE(TIndex node, TScalar lower) {
        if (lower > fUpper)
            return false;
        if (fIsLeaf(node))
        {
            TIndex const nLeafObjects = fLeafSize(node);
            for (TIndex i = 0; i < nLeafObjects; ++i)
            {
                TIndex o         = fLeafObject(node, i);
                TScalar const d  = fDistance(o);
                TScalar const lo = fUpper - eps;
                TScalar const hi = fUpper + eps;
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
        return true;
    };
    auto fVisit = [&] PBAT_HOST_DEVICE(TIndex node) {
        return fDoVisit(node, fLower(node));
    };
    using FVisit = decltype(fVisit);
    // For the nearest neighbour search, it might be best not to blindly run a pre-order
    // traversal of the branch and bound tree. If requested, let's optimistically run a best-first
    // search. At each node, we will sort the children by their lower bounds and visit them in that
    // order.
    common::Stack<TIndex, kStackDepth> stack{};
    if (not fVisit(root))
        return;
    stack.Push(root);
    if (bUseBestFirstSearch)
    {
        std::array<TIndex, N> ordering{};
        std::array<TIndex, N> children{};
        std::array<TScalar, N> lowers{};
        while (not stack.IsEmpty())
        {
            TIndex const node = stack.Pop();
            common::ForRange<0, N>([&]<auto i> {
                TIndex const child   = fChild.template operator()<i>(node);
                ordering[i]          = i;
                children[i]          = child;
                bool const bHasChild = child >= 0;
                lowers[i] = bHasChild ? fLower(child) : std::numeric_limits<TScalar>::max();
            });
            if constexpr (N == 2)
            {
                if (lowers[0] > lowers[1])
                    std::swap(ordering[0], ordering[1]);
            }
            else
            {
                std::sort(ordering.begin(), ordering.end(), [&](TIndex i, TIndex j) {
                    return lowers[i] < lowers[j];
                });
            }
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
            // NOTE: I hope this loop gets unrolled by the compiler!!
            for (int j = N - 1; j >= 0; --j)
            {
                auto const i = ordering[j];
                if (children[i] >= 0)
                {
                    if (fDoVisit(children[i], lowers[i]))
                        stack.Push(children[i]);
                }
                else
                    break; // The ordering is such that all non-children are at the end, so exit as
                           // soon as we encounter non-child.
            }
#include "pbat/warning/Pop.h"
        }
    }
    else
    {
        // NOTE:
        // Instead of using a pre-order traversal, we visit children of the currently visited node
        // simultaneously. This is because the branch and bound tree might store each node's
        // children in contiguous memory, or at least with some notion of locality. Thus, the fLower
        // function will benefit from many cache hits.
        // common::TraverseNAryTreePseudoPreOrder<FVisit, FChild, TIndex, N, kStackDepth>(
        //     fVisit,
        //     fChild,
        //     root);
        while (not stack.IsEmpty())
        {
            TIndex const node = stack.Pop();
            common::ForRange<0, N>([&]<auto i> {
                TIndex const child = fChild.template operator()<i>(node);
                if (child >= 0)
                    if (fVisit(child))
                        stack.Push(child);
            });
        }
    }
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
    class FLeafSize,
    class FLeafObject,
    class FDistanceLowerBound,
    class FDistance,
    class FOnFound,
    auto N,
    class TScalar,
    class TIndex,
    auto kHeapCapacity>
void KNearestNeighbours(
    FChild fChild,
    FIsLeaf fIsLeaf,
    FLeafSize fLeafSize,
    FLeafObject fLeafObject,
    FDistanceLowerBound fLower,
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
    auto fMakeNodeQueueItem = [&](TIndex node) {
        return QueueItem{EQueueItem::Node, node, fLower(node)};
    };
    auto fMakeObjectQueueItem = [&](TIndex object) {
        return QueueItem{EQueueItem::Object, object, fDistance(object)};
    };
    auto fGreater = [](QueueItem const& q1, QueueItem const& q2) {
        return q1.d > q2.d;
    };
    using Comparator = decltype(fGreater);
    using Container  = std::vector<QueueItem>;
    using MinHeap    = common::Heap<QueueItem, Comparator, kHeapCapacity>;
    MinHeap heap{fGreater};
    heap.Push(fMakeNodeQueueItem(root));
    TIndex k{0};
    while (not heap.IsEmpty())
    {
        QueueItem const q = heap.Pop();
        if (q.d > fUpper)
            break;
        if (q.type == EQueueItem::Node)
        {
            TIndex const node = q.idx;
            if (fIsLeaf(node))
            {
                TIndex const nLeafObjects = fLeafSize(node);
                for (TIndex i = 0; i < nLeafObjects; ++i)
                {
                    TIndex const o = fLeafObject(node, i);
                    heap.Push(fMakeObjectQueueItem(o));
                }
            }
            else
            {
                common::ForRange<0, N>([&]<auto i> {
                    TIndex const child = fChild.template operator()<i>(node);
                    if (child >= 0)
                        heap.Push(fMakeNodeQueueItem(child));
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

#endif // PBAT_GEOMETRY_SPATIALSEARCH_H
