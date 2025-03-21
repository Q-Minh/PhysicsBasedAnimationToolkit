/**
 * @file KdTree.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the KdTree class.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_KDTREE_H
#define PBAT_GEOMETRY_KDTREE_H

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
    enum { kLeafNodeLeftChild = -2 };
    Index begin; ///< Index to first point encapsulated in this node's AABB in the permutation list.
    Index n;     ///< Number of points encapsulated in this node's AABB starting from begin and
                 ///< continuing in contiguous memory in the permutation list until begin + n.
    Index c{kLeafNodeLeftChild}; ///< Index of first (i.e. left) child. < 0 if no child. We set to
                                 ///< -2 so that Left() < 0 and Right() < 0 if leaf node.
    /**
     * @brief Returns true if this node is a leaf node, false otherwise
     * @return true if this node is a leaf node, false otherwise
     */
    [[maybe_unused]] bool IsLeafNode() const { return c == -2; }
    /**
     * @brief Returns true if this node is an internal node, false otherwise
     * @return true if this node is an internal node, false otherwise
     */
    [[maybe_unused]] bool IsInternalNode() const { return not IsLeafNode(); }
    /**
     * @brief Returns left child node
     * @return Index of left child node
     */
    [[maybe_unused]] auto Left() const { return c; }
    /**
     * @brief Returns right child node
     * @return Index of right child node
     */
    [[maybe_unused]] auto Right() const { return c + 1; }
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
    KdTree(Eigen::DenseBase<TDerivedP> const& P, Index maxPointsInLeaf = 8);
    /**
     * @brief Construct a k-D tree from a set of points
     * @tparam TDerivedP Eigen dense expression type
     * @param P Points to construct the k-D tree from
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerivedP>
    void Construct(Eigen::DenseBase<TDerivedP> const& P, Index maxPointsInLeaf);

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
    IndexVectorX const& Permutation() const { return mPermutation; }
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
        Index const n,
        Index maxPointsInLeaf);

  private:
    IndexVectorX mPermutation;      ///< mPermutation[i] gives the index of point i in
                                    ///< the original points list given to
                                    ///< construct(std::vector<Vector3> const& points).
    std::vector<KdTreeNode> mNodes; ///< KDTree nodes.
};

template <int Dims>
template <class TDerivedP>
inline KdTree<Dims>::KdTree(Eigen::DenseBase<TDerivedP> const& P, Index maxPointsInLeaf)
{
    Construct(P, maxPointsInLeaf);
}

template <int Dims>
template <class TDerivedP>
inline void KdTree<Dims>::Construct(Eigen::DenseBase<TDerivedP> const& P, Index maxPointsInLeaf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.KdTree.Construct");
    Index const n            = P.cols();
    auto const nMaxLeafNodes = (n / maxPointsInLeaf) + (n % maxPointsInLeaf != 0);
    mNodes.clear();
    // Our k-D tree is a full binary tree (i.e. # nodes = 2*leaves - 1). The estimated number of
    // leaf nodes is nMaxLeafNodes and should be a good upper bound.
    mNodes.reserve(static_cast<std::size_t>(2 * nMaxLeafNodes - 1));
    auto iota = std::views::iota(Index{0}, n);
    mPermutation.resize(n);
    std::copy(iota.begin(), iota.end(), mPermutation.data());

    geometry::AxisAlignedBoundingBox<Dims> const aabb{P};
    auto constexpr begin = 0;
    auto constexpr root  = 0;
    mNodes.emplace_back(begin, n, KdTreeNode::kLeafNodeLeftChild);
    DoConstruct(root, P, aabb, begin, n, maxPointsInLeaf);
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

        if (node.IsInternalNode())
        {
            bfs.push(node.Left());
            bfs.push(node.Right());
        }
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

        if (node.IsInternalNode())
        {
            dfs.push(node.Left());
            dfs.push(node.Right());
        }
    }
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
template <class TDerivedP>
inline void KdTree<Dims>::DoConstruct(
    Index const nodeIdx,
    Eigen::DenseBase<TDerivedP> const& P,
    AxisAlignedBoundingBox<Dims> const& aabb,
    Index const begin,
    Index const n,
    Index maxPointsInLeaf)
{
    if (n <= maxPointsInLeaf)
        return;

    Eigen::Index dimension{};
    (aabb.max() - aabb.min()).maxCoeff(&dimension);

    Index const halfn = n / 2;

    std::nth_element(
        mPermutation.data() + begin,
        mPermutation.data() + begin + halfn,
        mPermutation.data() + begin + n,
        [&](Index const lhs, Index const rhs) { return P(dimension, lhs) < P(dimension, rhs); });

    auto& node = mNodes[static_cast<std::size_t>(nodeIdx)];
    node.c     = static_cast<Index>(mNodes.size());
    mNodes.emplace_back(begin, halfn, KdTreeNode::kLeafNodeLeftChild);
    mNodes.emplace_back(begin + halfn, n - halfn, KdTreeNode::kLeafNodeLeftChild);

    Scalar const split           = P(dimension, mPermutation[begin + halfn]);
    AxisAlignedBoundingBox laabb = aabb;
    laabb.max()(dimension)       = split;
    AxisAlignedBoundingBox raabb = aabb;
    raabb.min()(dimension)       = split;

    // NOTE: mNodes might have been reallocated, which would invalidate the variable (a reference)
    // "node". But probably not, since we reserve enough memory. Should take the time to prove this.
    DoConstruct(node.Left(), P, laabb, begin, halfn, maxPointsInLeaf);
    DoConstruct(node.Right(), P, raabb, begin + halfn, n - halfn, maxPointsInLeaf);
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_KDTREE_H
