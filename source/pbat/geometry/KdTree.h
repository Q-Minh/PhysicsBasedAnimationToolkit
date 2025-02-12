/**
 * @file KdTree.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the KdTree class.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_KD_TREE_H
#define PBAT_GEOMETRY_KD_TREE_H

#include "AxisAlignedBoundingBox.h"

#include <algorithm>
#include <pbat/Aliases.h>
#include <pbat/profiling/Profiling.h>
#include <queue>
#include <ranges>
#include <stack>
#include <vector>

namespace pbat {
namespace geometry {

/**
 * @brief Node of a KDTree
 */
struct KdTreeNode
{
    Index lc{-1}; ///< Index of left child of this node in the KDTree. -1 if no left child.
    Index rc{-1}; ///< Index of right child of this node in the KDTree. -1 if no right child.

    Index begin{
        -1}; ///< Index to first point encapsulated in this node's AABB in the permutation list.
    std::size_t n{0}; ///< Number of points encapsulated in this node's AABB starting from begin and
                      ///< continuing in contiguous memory in the permutation list until begin + n.

    Index depth{0}; ///< Depth of this node in the KDTree.
    /**
     * @brief Returns true if this node has a left child, false otherwise
     * @return true if this node has a left child, false otherwise
     */
    bool HasLeftChild() const { return lc > -1; }
    /**
     * @brief Returns true if this node has a right child, false otherwise
     * @return true if this node has a right child, false otherwise
     */
    bool HasRightChild() const { return rc > -1; }
    /**
     * @brief Returns true if this node is a leaf node, false otherwise
     * @return true if this node is a leaf node, false otherwise
     */
    [[maybe_unused]] bool IsLeafNode() const
    {
        return (not HasLeftChild()) and (not HasRightChild());
    }
    /**
     * @brief Returns true if this node is an internal node, false otherwise
     * @return true if this node is an internal node, false otherwise
     */
    [[maybe_unused]] bool IsInternalNode() const { return HasLeftChild() or HasRightChild(); }
};

/**
 * @brief KDTree class
 * @tparam Dims Number of dimensions of the points' coordinate system in the k-D tree
 */
template <int Dims>
class KdTree
{
  public:
    KdTree() = default;

    /**
     * @brief Construct a k-D tree from a set of points
     * @tparam TDerivedP Eigen dense expression type
     * @param P Points to construct the k-D tree from
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerivedP>
    KdTree(Eigen::DenseBase<TDerivedP> const& P, std::size_t maxPointsInLeaf = 10);

  private:
    inline static auto const fStopDefault = [](auto, auto) {
        return false;
    };

  public:
    /**
     * @brief Breadth-first search over the k-D tree
     * @tparam FVisit Callable with signature `bool(Index, KdTreeNode const&)`
     * @tparam FStop Callable with signature `bool(Index, KdTreeNode const&)`
     * @param visit Callback invoked when visiting a node. Returns true if the node's children
     * should be visited, false otherwise
     * @param stop Callback invoked when visiting a node. Returns true if the search should stop,
     * false otherwise
     * @param root Index of the root node to start the search from
     */
    template <class FVisit, class FStop = decltype(fStopDefault)>
    void BreadthFirstSearch(FVisit visit, FStop stop = fStopDefault, Index root = 0) const;

    /**
     * @brief Depth-first search over the k-D tree
     * @tparam FVisit Callable with signature `bool(Index, KdTreeNode const&)`
     * @tparam FStop Callable with signature `bool(Index, KdTreeNode const&)`
     * @param visit Callback invoked when visiting a node. Returns true if the node's children
     * @param stop Callback invoked when visiting a node. Returns true if the search should stop,
     * @param root Index of the root node to start the search from
     */
    template <class FVisit, class FStop = decltype(fStopDefault)>
    void DepthFirstSearch(FVisit visit, FStop stop = fStopDefault, Index root = 0) const;

    /**
     * @brief Construct a k-D tree from a set of points
     * @tparam TDerivedP Eigen dense expression type
     * @param P Points to construct the k-D tree from
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerivedP>
    void Construct(Eigen::DenseBase<TDerivedP> const& P, std::size_t maxPointsInLeaf);

    /**
     * @brief Returns the nodes of the k-D tree
     * @return Nodes of the k-D tree
     */
    std::vector<KdTreeNode> const& Nodes() const { return mNodes; }
    /**
     * @brief Returns the permutation of the points in the k-D tree
     *
     * The permutation is such that mPermutation[i] gives the index of point i in the original point
     * set given to Construct().
     *
     * @return Permutation of the points in the k-D tree
     */
    std::vector<Index> const& Permutation() const { return mPermutation; }
    /**
     * @brief Returns the points in a node
     * @param nodeIdx Index of the node
     * @return Range of points in the node
     */
    auto PointsInNode(Index const nodeIdx) const;
    /**
     * @brief Returns the points in a node
     * @param node Reference to k-D tree node
     * @return Range of points in the node
     */
    auto PointsInNode(KdTreeNode const& node) const;

  protected:
    /**
     * @brief Recursive construct call that constructs a sub-tree of the full k-D tree
     * @tparam TDerivedP Eigen dense expression type
     * @param nodeIdx Index of the sub-tree root node
     * @param P Points to construct the k-D tree from
     * @param aabb Axis-aligned bounding box of the points contained in the sub-tree
     * @param begin Index of the first point in the permutation list contained in the sub-tree
     * @param n Number of points in the permutation list starting from begin contained in the
     * sub-tree
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerivedP>
    void DoConstruct(
        Index const nodeIdx,
        Eigen::DenseBase<TDerivedP> const& P,
        AxisAlignedBoundingBox<Dims> const& aabb,
        Index const begin,
        std::size_t const n,
        std::size_t maxPointsInLeaf);
    /**
     * @brief Adds a node to the k-D tree's node list
     *
     * @param begin Index of the first point in the permutation list contained in the sub-tree
     * rooted in the added node
     * @param n Number of points in the permutation list starting from begin contained in the
     * sub-tree rooted in the added node
     * @param depth Depth of the added node in the k-D tree
     * @return Index of the added node in the k-D tree's node list
     */
    Index AddNode(Index begin, std::size_t n, Index depth);

  private:
    std::vector<Index> mPermutation; ///< mPermutation[i] gives the index of point i in
                                     ///< the original points list given to
                                     ///< construct(std::vector<Vector3> const& points).
    std::vector<KdTreeNode> mNodes;  ///< KDTree nodes.
};

template <int Dims>
template <class TDerivedP>
inline KdTree<Dims>::KdTree(Eigen::DenseBase<TDerivedP> const& P, std::size_t maxPointsInLeaf)
{
    Construct(P, maxPointsInLeaf);
}

template <int Dims>
template <class FVisit, class FStop>
inline void KdTree<Dims>::BreadthFirstSearch(FVisit visit, FStop stop, Index root) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.KdTree.BreadthFirstSearch");
    std::queue<Index> bfs{};
    bfs.push(root);
    while (!bfs.empty())
    {
        auto const nodeIdx     = bfs.front();
        auto const nodeIdxStl  = static_cast<std::size_t>(nodeIdx);
        KdTreeNode const& node = mNodes[nodeIdxStl];
        bfs.pop();
        if (stop(nodeIdx, node))
            break;

        if (!visit(nodeIdx, node))
            continue;

        if (node.HasLeftChild())
            bfs.push(node.lc);

        if (node.HasRightChild())
            bfs.push(node.rc);
    }
}

template <int Dims>
template <class FVisit, class FStop>
inline void KdTree<Dims>::DepthFirstSearch(FVisit visit, FStop stop, Index root) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.KdTree.DepthFirstSearch");
    std::stack<Index> dfs{};
    dfs.push(root);
    while (!dfs.empty())
    {
        auto const nodeIdx    = dfs.top();
        auto const nodeIdxStl = static_cast<std::size_t>(nodeIdx);
        dfs.pop();
        KdTreeNode const& node = mNodes[nodeIdxStl];

        if (stop(nodeIdx, node))
            break;

        if (!visit(nodeIdx, node))
            continue;

        if (node.HasLeftChild())
            dfs.push(node.lc);

        if (node.HasRightChild())
            dfs.push(node.rc);
    }
}

template <int Dims>
template <class TDerivedP>
inline void
KdTree<Dims>::Construct(Eigen::DenseBase<TDerivedP> const& P, std::size_t maxPointsInLeaf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.KdTree.Construct");
    std::size_t const n = static_cast<std::size_t>(P.cols());
    mNodes.clear();
    mNodes.reserve(n);
    auto iota = std::views::iota(Index{0}, static_cast<Index>(n));
    mPermutation.assign(iota.begin(), iota.end());

    geometry::AxisAlignedBoundingBox<Dims> const aabb{P};
    auto const depth   = 0;
    auto const begin   = 0;
    auto const rootIdx = AddNode(begin, n, depth);
    DoConstruct(rootIdx, P, aabb, begin, n, maxPointsInLeaf);
}

template <int Dims>
template <class TDerivedP>
inline void KdTree<Dims>::DoConstruct(
    Index const nodeIdx,
    Eigen::DenseBase<TDerivedP> const& P,
    AxisAlignedBoundingBox<Dims> const& aabb,
    Index const begin,
    std::size_t const n,
    std::size_t maxPointsInLeaf)
{
    if (n <= maxPointsInLeaf)
        return;

    Eigen::Index dimension{};
    (aabb.max() - aabb.min()).maxCoeff(&dimension);

    std::size_t const halfnStl   = n / 2;
    Index const halfn            = static_cast<Index>(halfnStl);
    std::size_t const nodeIdxStl = static_cast<std::size_t>(nodeIdx);

    std::nth_element(
        mPermutation.begin() + begin,
        mPermutation.begin() + begin + halfn,
        mPermutation.begin() + begin + static_cast<Index>(n),
        [&](Index const lhs, Index const rhs) { return P(dimension, lhs) < P(dimension, rhs); });

    auto const depth      = mNodes[nodeIdxStl].depth;
    auto lnode            = AddNode(begin, halfnStl, depth + 1u);
    auto rnode            = AddNode(begin + halfn, n - halfnStl, depth + 1u);
    mNodes[nodeIdxStl].lc = lnode;
    mNodes[nodeIdxStl].rc = rnode;

    Scalar const split           = P(dimension, mPermutation[begin + halfnStl]);
    AxisAlignedBoundingBox laabb = aabb;
    laabb.max()(dimension)       = split;
    AxisAlignedBoundingBox raabb = aabb;
    raabb.min()(dimension)       = split;

    DoConstruct(mNodes[nodeIdxStl].lc, P, laabb, begin, halfnStl, maxPointsInLeaf);
    DoConstruct(mNodes[nodeIdxStl].rc, P, raabb, begin + halfn, n - halfnStl, maxPointsInLeaf);
}

template <int Dims>
inline auto KdTree<Dims>::PointsInNode(Index const nodeIdx) const
{
    auto const& node = mNodes[static_cast<std::size_t>(nodeIdx)];
    return PointsInNode(node);
}

template <int Dims>
inline auto KdTree<Dims>::PointsInNode(KdTreeNode const& node) const
{
    namespace vi = std::views;
    auto indrng  = mPermutation | vi::drop(node.begin) | vi::take(node.n);
    return indrng;
}

template <int Dims>
inline Index KdTree<Dims>::AddNode(Index begin, std::size_t n, Index depth)
{
    Index const nodeIdx = static_cast<Index>(mNodes.size());
    KdTreeNode node{};
    node.begin = begin;
    node.n     = n;
    node.depth = depth;
    mNodes.push_back(node);
    return nodeIdx;
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_KD_TREE_H