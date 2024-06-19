#ifndef PBAT_GEOMETRY_KD_TREE_H
#define PBAT_GEOMETRY_KD_TREE_H

#include "AxisAlignedBoundingBox.h"

#include <pbat/Aliases.h>
#include <vector>

namespace pbat {
namespace geometry {

struct KdTreeNode
{
    std::int32_t left_child{
        -1}; ///< Index of left child of this node in the KDTree. -1 if no left child.
    std::int32_t right_child{
        -1}; ///< Index of right child of this node in the KDTree. -1 if no right child.

    std::int32_t begin{
        -1}; ///< Index to first point encapsulated in this node's AABB in the permutation list.
    std::int32_t n{
        0}; ///< Number of points encapsulated in this node's AABB starting from begin and
            ///< continuing in contiguous memory in the permutation list until begin + n.

    std::uint8_t depth{0};

    bool HasLeftChild() const { return left_child > -1; }
    bool HasRightChild() const { return right_child > -1; }
    bool IsLeafNode() const { return !HasLeftChild() && !HasRightChild(); }
    bool IsInternalNode() const { return HasLeftChild() || HasRightChild(); }
};

template <int Dims>
class KdTree
{
  public:
    KdTree() = default;

    template <class TDerivedP>
    KdTree(Eigen::DenseBase<TDerivedP> const& P, std::int16_t maxPointsInLeaf = 10);

    /**
     * @brief
     * @param visit Returns true if the node's children should be visited, false otherwise
     * @param should_stop Returns true if the search should stop
     * @param root
     */
    void BreadthFirstSearch(
        std::function<bool(Index /*node_index*/, KdTreeNode const& /*node*/)> const& visit,
        std::function<bool(Index /*node_index*/, KdTreeNode const& /*node*/)> const& stop = {},
        Index root = 0) const;

    /**
     * @brief
     * @param visit Returns true if the node's children should be visited, false otherwise
     * @param should_stop Returns true if the search should stop
     * @param root
     */
    void DepthFirstSearch(
        std::function<bool(Index /*node_index*/, KdTreeNode const& /*node*/)> const& visit,
        std::function<bool(Index /*node_index*/, KdTreeNode const& /*node*/)> const& stop = {},
        Index root = 0) const;

    template <class TDerivedP>
    void Construct(Eigen::DenseBase<TDerivedP> const& P, std::size_t maxPointsInLeaf);

    std::vector<KdTreeNode> const& nodes() const { return nodes_; }

    std::vector<Index> const& permutation() const { return permutation_; }

    std::vector<Index> PointsInNode(Index const nodeIdx) const;

    std::vector<Index> PointsInNode(KdTreeNode const& nodeIdx) const;

  protected:
    template <class TDerivedP>
    void DoConstruct(
        Index const node_idx,
        Eigen::DenseBase<TDerivedP> const& points,
        AxisAlignedBoundingBox<Dims> const& aabb,
        Index const begin,
        Index const n,
        std::size_t maxPointsInLeaf);

    Index AddNode(Index begin, Index n, Index depth);

  private:
    std::vector<Index> permutation_; ///< permutation_[i] gives the index of point i in
                                     ///< the original points list given to
                                     ///< construct(std::vector<Vector3> const& points).
    std::vector<KdTreeNode> nodes_;  ///< KDTree nodes.
};

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_KD_TREE_H