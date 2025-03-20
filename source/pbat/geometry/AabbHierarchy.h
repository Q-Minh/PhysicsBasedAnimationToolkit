#ifndef PBAT_GEOMETRY_AABBHIERARCHY_H
#define PBAT_GEOMETRY_AABBHIERARCHY_H

#include "KdTree.h"
#include "pbat/Aliases.h"
#include "pbat/common/NAryTreeTraversal.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>

namespace pbat::geometry {

/**
 * @brief Bounding volume hierarchy over axis-aligned bounding boxes.
 *
 * This BVH does not store the AABBs themselves, only the tree topology and the AABBs of the tree
 * nodes. The user is responsible for storing the objects and their AABBs. Doing so allows this BVH
 * implementation to support arbitrary object types. When objects move, the user should update the
 * AABBs and call AabbHierarchy::Update() to recompute the tree node AABBs.
 *
 * @note This BVH implementation relies on a static k-D tree, thus tree topology cannot be
 * modified after construction. In other words, dynamic insertion/deletion of objects
 * is only supported by reconstructing the tree from scratch.
 *
 * @tparam kDims Number of spatial dimensions
 */
template <auto kDims>
class AabbHierarchy
{
  public:
    AabbHierarchy() = default;
    /**
     * @brief Construct an Aabb Hierarchy from an input AABB matrix LU
     *
     * @tparam TDerived Type of the input matrix
     * @param B 2*kDims x |# objects| matrix of object AABBs, such that for an object o,
     * B.col(o).head<kDims>() is the lower bound and B.col(o).tail<kDims>() is the upper bound.
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerived>
    AabbHierarchy(Eigen::DenseBase<TDerived> const& B, Index maxPointsInLeaf = 10);
    /**
     * @brief Construct an Aabb Hierarchy from an input AABB matrix LU
     *
     * @tparam TDerived Type of the input matrix
     * @param B 2*kDims x |# objects| matrix of object AABBs, such that for an object o,
     * B.col(o).head<kDims>() is the lower bound and B.col(o).tail<kDims>() is the upper bound.
     */
    template <class TDerived>
    void Construct(Eigen::DenseBase<TDerived> const& B, Index maxPointsInLeaf = 10);
    /**
     * @brief Recomputes k-D tree node AABBs given the object AABBs
     *
     * A sequential post-order traversal of the k-D tree is performed, i.e. bottom up nodal AABB
     * computation, leading to \f$ O(n) \f$ time complexity. The traversal should be have some
     * respectable cache-efficiency at the tree-level, since nodes and their 2 children are stored
     * contiguously in memory. However, objects stored in leaves are generally spread out in memory.
     *
     * @tparam TDerivedB Type of the input matrix
     * @param B 2*kDims x |# objects| matrix of object AABBs, such that for an object o,
     * B.col(o).head<kDims>() is the lower bound and B.col(o).tail<kDims>() is the upper bound.
     */
    template <class TDerivedB>
    void Update(Eigen::DenseBase<TDerivedB> const& B);

  private:
    Matrix<2 * kDims, Eigen::Dynamic>
        LU;             ///< 2*kDims x |# k-D tree nodes| matrix of AABBs, such that
                        ///< for a node node, LU.col(node).head<kDims>() is the lower
                        ///< bound and LU.col(node).tail<kDims>() is the upper bound.
    KdTree<kDims> tree; ///< KdTree over the AABBs
};

template <auto kDims>
template <class TDerived>
inline AabbHierarchy<kDims>::AabbHierarchy(
    Eigen::DenseBase<TDerived> const& B,
    Index maxPointsInLeaf)
    : LU(), tree()
{
    Construct(B, maxPointsInLeaf);
}

template <auto kDims>
template <class TDerived>
inline void
AabbHierarchy<kDims>::Construct(Eigen::DenseBase<TDerived> const& B, Index maxPointsInLeaf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbHierarchy.Construct");
    Matrix<kDims, Eigen::Dynamic> P =
        0.5 * (B.template topRows<kDims>() + B.template bottomRows<kDims>());
    tree.Construct(P, maxPointsInLeaf);
    auto const nNodes = static_cast<Index>(tree.Nodes().size());
    LU.resize(2 * kDims, nNodes);
    LU.topRows<kDims>().setConstant(std::numeric_limits<Scalar>::max());
    LU.bottomRows<kDims>().setConstant(std::numeric_limits<Scalar>::lowest());
    ComputeNodeAabbs(B);
}

template <auto kDims>
template <class TDerivedB>
inline void AabbHierarchy<kDims>::Update(Eigen::DenseBase<TDerivedB> const& B)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbHierarchy.Update");
    KdTreeNode const* nodes  = tree.Nodes().data();
    IndexVectorX const& perm = tree.Permutation();
    common::TraverseNAryTreePseudoPostOrder(
        [&](Index n) {
            KdTreeNode const& node = nodes[n];
            if (node.IsLeaf())
            {
                auto inds               = perm(Eigen::seqN(node.begin, node.n));
                LU.col(n).head<kDims>() = B(Eigen::placeholders::all, inds).rowwise().minCoeff();
                LU.col(n).tail<kDims>() = B(Eigen::placeholders::all, inds).rowwise().maxCoeff();
            }
            else
            {
                auto const fUpdate = [&](Index c) {
                    auto cmin = LU.col(c).head<kDims>();
                    auto cmax = LU.col(c).tail<kDims>();
                    auto nmin = LU.col(n).head<kDims>();
                    auto nmax = LU.col(n).tail<kDims>();
                    nmin      = nmin.cwiseMin(cmin);
                    nmax      = nmax.cwiseMax(cmax);
                };
                if (node.lc)
                    fUpdate(node.lc);
                if (node.rc)
                    fUpdate(node.rc);
            }
        },
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return nodes[n].lc;
            else
                return nodes[n].rc;
        });
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_AABBHIERARCHY_H
