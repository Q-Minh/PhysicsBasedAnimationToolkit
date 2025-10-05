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
#include "pbat/common/NAryTreeTraversal.h"
#include "pbat/common/Stack.h"

#include <algorithm>
#include <pbat/Aliases.h>
#include <pbat/profiling/Profiling.h>
#include <ranges>
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
    [[maybe_unused]] bool IsLeaf() const { return c == -2; }
    /**
     * @brief Returns true if this node is an internal node, false otherwise
     * @return true if this node is an internal node, false otherwise
     */
    [[maybe_unused]] bool IsInternal() const { return not IsLeaf(); }
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
     *
     * Implements a top-down median splitting strategy to construct the k-D tree.
     * The median is chosen along the largest axis at each node of the tree, where the left child
     * will inherit the parent node's points less than the median and the right child will inherit
     * the rest. At each node, we use std::nth_element, i.e. introselect, which has
     * \f$ O(n) \f$ average complexity (and \f$ O(n) \f$ worst-case complexity in recent
     * compiler versions, i.e. [Microsoft STL](https://github.com/microsoft/STL/pull/5100)), to
     * partition the points in-place. The largest k-D tree occurs when `maxPointsInLeaf == 1`, and
     * the tree is always a full binary tree with \f$ 2n - 1 \f$ nodes.
     *
     * Thus, tree construction has \f$ O(n log(n)) \f$ average, and \f$ O(n log^2 n) \f$
     * worst-case time complexity.
     *
     * @note For fast construction, radix tree is potentially a better choice. However, our radix
     * tree does not support multiple points per leaf.
     *
     * @note We could parallelize construction quite easily in the future, since the node ranges are
     * non-overlapping.
     *
     * @tparam TDerivedP Eigen dense expression type
     * @param P Points to construct the k-D tree from
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerivedP>
    void Construct(Eigen::DenseBase<TDerivedP> const& P, Index maxPointsInLeaf);
    /**
     * @brief Depth-first search over the k-D tree
     * @tparam FVisit Callable with signature `bool(Index, KdTreeNode const&)`
     * @tparam FStop Callable with signature `bool(Index, KdTreeNode const&)`
     * @param visit Callback invoked when visiting a node. Returns true if the node's sub-tree
     * should be visited, false otherwise.
     * @param root Index of the root node to start the search from
     */
    template <class FVisit>
    void DepthFirstSearch(FVisit visit, Index root = 0) const;

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
    /**
     * @brief Returns the root node index
     * @return Root node index
     */
    Index constexpr Root() const { return 0; }

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
    // Our k-D tree is a full binary tree (i.e. # nodes = 2*leaves - 1). We try to estimate
    // the number of leaf nodes to reserve memory up-front and prevent any reallocation, but without
    // excessively using up memory. We estimate the number of nodes per leaf to be 80% of the
    // maximum number of points in a leaf node.
    Index const nPoints                      = P.cols();
    Scalar constexpr kEstimatedLeafOccupancy = 0.8;
    auto const nEstimatedNodesPerLeaf =
        (maxPointsInLeaf > 1) ?
            static_cast<Index>(
                std::ceil(kEstimatedLeafOccupancy * static_cast<Scalar>(maxPointsInLeaf))) :
            1;
    auto const nEstimatedLeafNodes =
        (nPoints / nEstimatedNodesPerLeaf) + (nPoints % nEstimatedNodesPerLeaf != 0);
    mNodes.clear();
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    mNodes.reserve(2 * nEstimatedLeafNodes - 1);
    auto iota = std::views::iota(Index{0}, nPoints);
    mPermutation.resize(nPoints);
    std::copy(iota.begin(), iota.end(), mPermutation.data());

    // Top-down construction of the k-D tree
    struct StackFrame
    {
        geometry::AxisAlignedBoundingBox<Dims> aabb;
        Index nodeIdx;
        Index begin;
        Index n;
    };
    common::Stack<StackFrame, 128> stack{};
    Index constexpr root = 0;
    stack.Push({geometry::AxisAlignedBoundingBox<Dims>{P}, root, 0, nPoints});
    mNodes.push_back({0, nPoints, KdTreeNode::kLeafNodeLeftChild});
    while (not stack.IsEmpty())
    {
        auto const [aabb, nodeIdx, begin, n] = stack.Pop();
        if (n <= maxPointsInLeaf)
            continue;

        // Find the dimension with the largest extent
        Eigen::Index dimension{};
        (aabb.max() - aabb.min()).maxCoeff(&dimension);
        // Partition the points along the dimension on left/right sides of median
        Index const halfn = n / 2;
        std::nth_element(
            mPermutation.data() + begin,
            mPermutation.data() + begin + halfn,
            mPermutation.data() + begin + n,
            [&](Index const lhs, Index const rhs) {
                return P(dimension, lhs) < P(dimension, rhs);
            });
        // Set the left (and implicitly right) child index of the current node
        mNodes[nodeIdx].c = static_cast<Index>(mNodes.size());
        // Split bounding box into child boxes
        Scalar const split           = P(dimension, mPermutation[begin + halfn]);
        AxisAlignedBoundingBox laabb = aabb;
        laabb.max()(dimension)       = split;
        AxisAlignedBoundingBox raabb = aabb;
        raabb.min()(dimension)       = split;
        // Store left/right node contiguously in memory
        mNodes.push_back({begin, halfn, KdTreeNode::kLeafNodeLeftChild});
        mNodes.push_back({begin + halfn, n - halfn, KdTreeNode::kLeafNodeLeftChild});
        // Schedule left and right sub-tree constructions
        stack.Push({laabb, mNodes[nodeIdx].Left(), begin, halfn});
        stack.Push({raabb, mNodes[nodeIdx].Right(), begin + halfn, n - halfn});
    }
#include "pbat/warning/Pop.h"
}

template <int Dims>
template <class FVisit>
inline void KdTree<Dims>::DepthFirstSearch(FVisit fVisit, Index root) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.KdTree.DepthFirstSearch");
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    common::TraverseNAryTreePseudoPreOrder(
        [&](Index node) { return fVisit(node, mNodes[node]); },
        [&]<auto c>(Index node) {
            if constexpr (c == 0)
                return mNodes[node].Left();
            else
                return mNodes[node].Right();
        },
        root);
#include "pbat/warning/Pop.h"
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

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_KDTREE_H
