#ifndef PBAT_COMMON_NARYTREETRAVERSAL_H
#define PBAT_COMMON_NARYTREETRAVERSAL_H

#include "ConstexprFor.h"
#include "Stack.h"
#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"

namespace pbat::common {

/**
 * @brief Pre-order traversal over an n-ary tree starting from root
 * @tparam FVisit Function to visit a node
 * @tparam FChild Function to get the child of a node
 * @tparam TIndex Type of the index
 * @tparam N Number of children per node
 * @tparam kStackDepth Maximum depth of the traversal's stack
 * @param fVisit `bool(TIndex node)` function to visit a node. Returns true if node's sub-tree
 * should be visited.
 * @param fChild `template <TIndex c> TIndex(TIndex node)` function to get child c of a node.
 * Returns the child index or -1 if no child.
 * @param root Index of the root node to start the search from
 *
 * @note The traversal is deemed "pseudo" pre-order because each visited node's children are visited
 * in arbitrary order. The only guarantee is that a parent node is visited before its children. This
 * is due to compile-time loops not being able to guarantee the order of execution.
 */
template <class FVisit, class FChild, class TIndex = Index, auto N = 2, auto kStackDepth = 64>
PBAT_HOST_DEVICE void TraverseNAryTreePseudoPreOrder(FVisit fVisit, FChild fChild, TIndex root = 0);

/**
 * @brief Post-order traversal over an n-ary tree starting from root
 * @tparam FVisit Function to visit a node
 * @tparam FChild Function to get the child of a node
 * @tparam TIndex Type of the index
 * @tparam N Number of children per node
 * @tparam kStackDepth Maximum depth of the traversal's stack. The actual stack depth will be
 * N*kStackDepth.
 * @param fVisit `void(TIndex node)` function to visit a node.
 * @param fChild `template <TIndex c> TIndex(TIndex node)` function to get child c of a node.
 * Returns the child index or -1 if no child.
 * @param root Index of the root node to start the search from
 *
 * @note The traversal is deemed "pseudo" post-order because each visited node's children are
 * visited in arbitrary order. The only guarantee is that a parent node is visited after its
 * children. This is due to compile-time loops not being able to guarantee the order of execution.
 *
 * @note The visitor does not support sub-tree pruning, since visited nodes are always processed
 * after their sub-tree.
 */
template <class FVisit, class FChild, class TIndex = Index, auto N = 2, auto kStackDepth = 64>
PBAT_HOST_DEVICE void
TraverseNAryTreePseudoPostOrder(FVisit fVisit, FChild fChild, TIndex root = 0);

template <class FVisit, class FChild, class TIndex, auto N, auto kStackDepth>
PBAT_HOST_DEVICE void TraverseNAryTreePseudoPreOrder(FVisit fVisit, FChild fChild, TIndex root)
{
    Stack<TIndex, kStackDepth> dfs{};
    dfs.Push(root);
    while (not dfs.IsEmpty())
    {
        auto const node = dfs.Top();
        dfs.Pop();
        if (not fVisit(node))
            continue;

        ForRange<0, N>([&]<auto i> {
            auto const child = fChild.template operator()<N - i - 1>(node);
            if (child >= 0)
                dfs.Push(child);
        });
    }
}

template <class FVisit, class FChild, class TIndex, auto N, auto kStackDepth>
PBAT_HOST_DEVICE void TraverseNAryTreePseudoPostOrder(FVisit fVisit, FChild fChild, TIndex root)
{
    struct Visit
    {
        TIndex node;
        bool bChildrenVisited;
    };
    Stack<Visit, N * kStackDepth> dfs{};
    dfs.Push({root, false});
    while (not dfs.IsEmpty())
    {
        auto const visit = dfs.Top();
        dfs.Pop();
        if (visit.bChildrenVisited)
        {
            fVisit(visit.node);
        }
        else
        {
            dfs.Push({visit.node, true});
            ForRange<0, N>([&]<auto i> {
                auto const child = fChild.template operator()<N - i - 1>(visit.node);
                if (child >= 0)
                    dfs.Push({child, false});
            });
        }
    }
}

} // namespace pbat::common

#endif // PBAT_COMMON_NARYTREETRAVERSAL_H
