/**
 * @file AabbKdTreeHierarchy.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief BVH over axis-aligned bounding boxes using a k-D tree
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_GEOMETRY_AABBKDTREEHIERARCHY_H
#define PBAT_GEOMETRY_AABBKDTREEHIERARCHY_H

#include "KdTree.h"
#include "SpatialSearch.h"
#include "pbat/Aliases.h"
#include "pbat/common/NAryTreeTraversal.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>

namespace pbat::geometry {

/**
 * @brief Axis-aligned k-D tree hierarchy of axis-aligned bounding boxes.
 *
 * This BVH does not store the AABBs themselves, only the tree topology and the AABBs of the tree
 * nodes. The user is responsible for storing the objects and their AABBs. Doing so allows this BVH
 * implementation to support arbitrary object types. When objects move, the user should update the
 * AABBs and call AabbKdTreeHierarchy::Update() to recompute the tree node AABBs.
 *
 * @note This BVH implementation relies on a static k-D tree, thus tree topology cannot be
 * modified after construction. In other words, dynamic insertion/deletion of objects
 * is only supported by reconstructing the tree from scratch.
 *
 * @tparam kDims Number of spatial dimensions
 */
template <auto kDims>
class AabbKdTreeHierarchy
{
  public:
    static auto constexpr kDims = kDims;         ///< Number of spatial dimensions
    using SelfType = AabbKdTreeHierarchy<kDims>; ///< Type of this template instantiation

    AabbKdTreeHierarchy() = default;
    /**
     * @brief Construct an Aabb Hierarchy from an input AABB matrices L, U
     *
     * @tparam TDerivedL Type of the lower bound matrix
     * @tparam TDerivedU Type of the upper bound matrix
     * @param L kDims x |# objects| matrix of object AABB lower bounds, such that for an object o,
     * L.col(o) is the lower bound.
     * @param U kDims x |# objects| matrix of object AABB upper bounds, such that for an object o,
     * U.col(o) is the upper bound.
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerivedL, class TDerivedU>
    AabbKdTreeHierarchy(
        Eigen::DenseBase<TDerivedL> const& L,
        Eigen::DenseBase<TDerivedU> const& U,
        Index maxPointsInLeaf = 8);
    /**
     * @brief Construct an Aabb Hierarchy from an input AABB matrix B
     *
     * @tparam TDerivedL Type of the lower bound matrix
     * @tparam TDerivedU Type of the upper bound matrix
     * @param L kDims x |# objects| matrix of object AABB lower bounds, such that for an object o,
     * L.col(o) is the lower bound.
     * @param U kDims x |# objects| matrix of object AABB upper bounds, such that for an object o,
     * U.col(o) is the upper bound.
     * @param maxPointsInLeaf Maximum number of points in a leaf node
     */
    template <class TDerivedL, class TDerivedU>
    void Construct(
        Eigen::DenseBase<TDerivedL> const& L,
        Eigen::DenseBase<TDerivedU> const& U,
        Index maxPointsInLeaf = 8);
    /**
     * @brief Recomputes k-D tree node AABBs given the object AABBs
     *
     * A sequential post-order traversal of the k-D tree is performed, i.e. bottom up nodal AABB
     * computation, leading to \f$ O(n) \f$ time complexity. The traversal should be have some
     * respectable cache-efficiency at the tree-level, since nodes and their 2 children are stored
     * contiguously in memory. However, objects stored in leaves are generally spread out in memory.
     *
     * @tparam TDerivedL Type of the lower bound matrix
     * @tparam TDerivedU Type of the upper bound matrix
     * @param L kDims x |# objects| matrix of object AABB lower bounds, such that for an object o,
     * L.col(o) is the lower bound.
     * @param U kDims x |# objects| matrix of object AABB upper bounds, such that for an object o,
     * U.col(o) is the upper bound.
     */
    template <class TDerivedL, class TDerivedU>
    void Update(Eigen::DenseBase<TDerivedL> const& L, Eigen::DenseBase<TDerivedU> const& U);
    /**
     * @brief Find all objects that overlap with some user-defined query
     *
     * @tparam FNodeOverlaps Function with signature `template <class TDerivedL, class TDerivedU>
     * bool(TDerivedL const& L, TDerivedU const& U)`
     * @tparam FObjectOverlaps Function with signature `bool(Index o)`
     * @tparam FOnOverlap Function with signature `void(Index n, Index o)`
     * @param fNodeOverlaps Function to determine if a node overlaps with the query
     * @param fObjectOverlaps Function to determine if an object overlaps with the query
     * @param fOnOverlap Function to process an overlap
     */
    template <class FNodeOverlaps, class FObjectOverlaps, class FOnOverlap>
    void
    Overlaps(FNodeOverlaps fNodeOverlaps, FObjectOverlaps fObjectOverlaps, FOnOverlap fOnOverlap)
        const;
    /**
     * @brief Find the nearest neighbour to some user-defined query. If there are multiple nearest
     * neighbours, we may return a certain number > 1 of them.
     *
     * @tparam FDistanceToNode Function with signature `template <class TDerivedL, class TDerivedU>
     * Scalar(TDerivedL const& L, TDerivedU const& U)`
     * @tparam FDistanceToObject Function with signature `Scalar(Index o)`
     * @tparam FOnNearestNeighbour Function with signature `void(Index o, Scalar d, Index k)`
     * @param fDistanceToNode Function to compute the distance to a node
     * @param fDistanceToObject Function to compute the distance to an object
     * @param fOnNearestNeighbour Function to process a nearest neighbour
     * @param radius Maximum distance to search for nearest neighbours
     * @param eps Maximum distance error
     */
    template <class FDistanceToNode, class FDistanceToObject, class FOnNearestNeighbour>
    void NearestNeighbour(
        FDistanceToNode fDistanceToNode,
        FDistanceToObject fDistanceToObject,
        FOnNearestNeighbour fOnNearestNeighbour,
        Scalar radius = std::numeric_limits<Scalar>::max(),
        Scalar eps    = Scalar(0)) const;
    /**
     * @brief Find the K nearest neighbours to some user-defined query.
     *
     * @tparam FDistanceToNode Function with signature `template <class TDerivedL, class TDerivedU>
     * Scalar(TDerivedL const& L, TDerivedU const& U)`
     * @tparam FDistanceToObject Function with signature `Scalar(Index o)`
     * @tparam FOnNearestNeighbour Function with signature `void(Index o, Scalar d, Index k)`
     * @param fDistanceToNode Function to compute the distance to a node
     * @param fDistanceToObject Function to compute the distance to an object
     * @param fOnNearestNeighbour Function to process a nearest neighbour
     * @param K Number of nearest neighbours to find
     * @param radius Maximum distance to search for nearest neighbours
     */
    template <class FDistanceToNode, class FDistanceToObject, class FOnNearestNeighbour>
    void KNearestNeighbours(
        FDistanceToNode fDistanceToNode,
        FDistanceToObject fDistanceToObject,
        FOnNearestNeighbour fOnNearestNeighbour,
        Index K,
        Scalar radius = std::numeric_limits<Scalar>::max()) const;
    /**
     * @brief Find all object pairs that overlap
     *
     * @tparam FObjectsOverlap Function with signature `bool(Index o1, Index o2)`
     * @tparam FOnSelfOverlap Function with signature `void(Index o1, Index o2, Index k)`
     * @param fObjectsOverlap Function to determine if 2 objects from this tree overlap
     * @param fOnSelfOverlap Function to process an overlap
     */
    template <class FObjectsOverlap, class FOnSelfOverlap>
    void SelfOverlaps(FObjectsOverlap fObjectsOverlap, FOnSelfOverlap fOnSelfOverlap) const;
    /**
     * @brief Find all object pairs that overlap with another hierarchy
     *
     * @tparam FObjectsOverlap Callable with signature `bool(Index o1, Index o2)`
     * @tparam FOnOverlap Callable with signature `void(Index o1, Index o2, Index k)`
     * @param rhs Other hierarchy to compare against
     * @param fObjectsOverlap Function to determine if 2 objects (o1,o2) overlap, where o1 is an
     * object from this tree and o2 is an object from the rhs tree
     * @param fOnOverlap Function to process an overlap (o1,o2) where o1 is an object from this tree
     * and o2 is an object from the rhs tree
     */
    template <class FObjectsOverlap, class FOnOverlap>
    void
    Overlaps(SelfType const& rhs, FObjectsOverlap fObjectsOverlap, FOnOverlap fOnOverlap) const;

    /**
     * @brief Get the internal node bounding boxes
     * @return `2*kDims x |# internal nodes|` matrix of AABBs, such that for an internal node node,
     * IB.col(node).head<kDims>() is the lower bound and IB.col(node).tail<kDims>() is the upper
     * bound.
     */
    auto InternalNodeBoundingBoxes() const -> Matrix<2 * kDims, Eigen::Dynamic> const&
    {
        return IB;
    }
    /**
     * @brief Get the underlying k-D tree
     * @return The k-D tree hierarchy
     */
    auto Tree() const -> KdTree<kDims> const& { return tree; }

  private:
    Matrix<2 * kDims, Eigen::Dynamic>
        IB;             ///< 2*kDims x |# k-D tree nodes| matrix of AABBs, such that
                        ///< for a node node, IB.col(node).head<kDims>() is the lower
                        ///< bound and IB.col(node).tail<kDims>() is the upper bound.
    KdTree<kDims> tree; ///< KdTree over the AABBs
};

template <auto kDims>
template <class TDerivedL, class TDerivedU>
inline AabbKdTreeHierarchy<kDims>::AabbKdTreeHierarchy(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U,
    Index maxPointsInLeaf)
{
    Construct(L, U, maxPointsInLeaf);
}

template <auto kDims>
template <class TDerivedL, class TDerivedU>
inline void AabbKdTreeHierarchy<kDims>::Construct(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U,
    Index maxPointsInLeaf)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbKdTreeHierarchy.Construct");
    Matrix<kDims, Eigen::Dynamic> P = 0.5 * (L.derived() + U.derived());
    tree.Construct(P, maxPointsInLeaf);
    auto const nNodes = static_cast<Index>(tree.Nodes().size());
    IB.resize(2 * kDims, nNodes);
    IB.template topRows<kDims>().setConstant(std::numeric_limits<Scalar>::max());
    IB.template bottomRows<kDims>().setConstant(std::numeric_limits<Scalar>::lowest());
}

template <auto kDims>
template <class TDerivedL, class TDerivedU>
inline void AabbKdTreeHierarchy<kDims>::Update(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbKdTreeHierarchy.Update");
    KdTreeNode const* nodes  = tree.Nodes().data();
    IndexVectorX const& perm = tree.Permutation();
    common::TraverseNAryTreePseudoPostOrder(
        [&](Index n) {
            KdTreeNode const& node = nodes[n];
            if (node.IsLeaf())
            {
                auto inds               = perm(Eigen::seqN(node.begin, node.n));
                IB.col(n).head<kDims>() = L(Eigen::placeholders::all, inds).rowwise().minCoeff();
                IB.col(n).tail<kDims>() = U(Eigen::placeholders::all, inds).rowwise().maxCoeff();
            }
            else
            {
                // Our k-D tree's internal nodes always have both children.
                auto nbox          = IB.col(n);
                auto lbox          = IB.col(node.Left());
                auto rbox          = IB.col(node.Right());
                nbox.head<kDims>() = lbox.head<kDims>().cwiseMin(rbox.head<kDims>());
                nbox.tail<kDims>() = lbox.tail<kDims>().cwiseMax(rbox.tail<kDims>());
            }
        },
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return nodes[n].Left();
            else
                return nodes[n].Right();
        });
}

template <auto kDims>
template <class FNodeOverlaps, class FObjectOverlaps, class FOnOverlap>
inline void AabbKdTreeHierarchy<kDims>::Overlaps(
    FNodeOverlaps fNodeOverlaps,
    FObjectOverlaps fObjectOverlaps,
    FOnOverlap fOnOverlap) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbKdTreeHierarchy.Overlaps");
    KdTreeNode const* nodes  = tree.Nodes().data();
    IndexVectorX const& perm = tree.Permutation();
    geometry::Overlaps(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return nodes[n].Left();
            else
                return nodes[n].Right();
        },
        [&](Index n) { return nodes[n].IsLeaf(); },
        [&](Index n) { return nodes[n].n; },
        [&](Index n, Index i) { return perm(nodes[n].begin + i); },
        [&](Index n) {
            auto L          = IB.col(n).head<kDims>();
            auto U          = IB.col(n).tail<kDims>();
            using TDerivedL = decltype(L);
            using TDerivedU = decltype(U);
            return fNodeOverlaps.template operator()<TDerivedL, TDerivedU>(L, U);
        },
        fObjectOverlaps,
        fOnOverlap);
}

template <auto kDims>
template <class FDistanceToNode, class FDistanceToObject, class FOnNearestNeighbour>
inline void AabbKdTreeHierarchy<kDims>::NearestNeighbour(
    FDistanceToNode fDistanceToNode,
    FDistanceToObject fDistanceToObject,
    FOnNearestNeighbour fOnNearestNeighbour,
    Scalar radius,
    Scalar eps) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbKdTreeHierarchy.NearestNeighbour");
    KdTreeNode const* nodes  = tree.Nodes().data();
    IndexVectorX const& perm = tree.Permutation();
    geometry::NearestNeighbour(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return nodes[n].Left();
            else
                return nodes[n].Right();
        },
        [&](Index n) { return nodes[n].IsLeaf(); },
        [&](Index n) { return nodes[n].n; },
        [&](Index n, Index i) { return perm(nodes[n].begin + i); },
        [&](Index n) {
            auto L          = IB.col(n).head<kDims>();
            auto U          = IB.col(n).tail<kDims>();
            using TDerivedL = decltype(L);
            using TDerivedU = decltype(U);
            return fDistanceToNode.template operator()<TDerivedL, TDerivedU>(L, U);
        },
        fDistanceToObject,
        fOnNearestNeighbour,
        false /*bUseBestFirstSearch*/,
        radius,
        eps);
}

template <auto kDims>
template <class FDistanceToNode, class FDistanceToObject, class FOnNearestNeighbour>
inline void AabbKdTreeHierarchy<kDims>::KNearestNeighbours(
    FDistanceToNode fDistanceToNode,
    FDistanceToObject fDistanceToObject,
    FOnNearestNeighbour fOnNearestNeighbour,
    Index K,
    Scalar radius) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbKdTreeHierarchy.KNearestNeighbours");
    KdTreeNode const* nodes  = tree.Nodes().data();
    IndexVectorX const& perm = tree.Permutation();
    geometry::KNearestNeighbours(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return nodes[n].Left();
            else
                return nodes[n].Right();
        },
        [&](Index n) { return nodes[n].IsLeaf(); },
        [&](Index n) { return nodes[n].n; },
        [&](Index n, Index i) { return perm(nodes[n].begin + i); },
        [&](Index n) {
            auto L          = IB.col(n).head<kDims>();
            auto U          = IB.col(n).tail<kDims>();
            using TDerivedL = decltype(L);
            using TDerivedU = decltype(U);
            return fDistanceToNode.template operator()<TDerivedL, TDerivedU>(L, U);
        },
        fDistanceToObject,
        fOnNearestNeighbour,
        K,
        radius);
}

template <auto kDims>
template <class FObjectsOverlap, class FOnSelfOverlap>
inline void AabbKdTreeHierarchy<kDims>::SelfOverlaps(
    FObjectsOverlap fObjectsOverlap,
    FOnSelfOverlap fOnSelfOverlap) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbKdTreeHierarchy.SelfOverlaps");
    KdTreeNode const* nodes  = tree.Nodes().data();
    IndexVectorX const& perm = tree.Permutation();
    geometry::SelfOverlaps(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return nodes[n].Left();
            else
                return nodes[n].Right();
        },
        [&](Index n) { return nodes[n].IsLeaf(); },
        [&](Index n) { return nodes[n].n; },
        [&](Index n, Index i) { return perm(nodes[n].begin + i); },
        [&](Index n1, Index n2) {
            auto L1 = IB.col(n1).head<kDims>();
            auto U1 = IB.col(n1).tail<kDims>();
            auto L2 = IB.col(n2).head<kDims>();
            auto U2 = IB.col(n2).tail<kDims>();
            return geometry::OverlapQueries::AxisAlignedBoundingBoxes(L1, U1, L2, U2);
        },
        fObjectsOverlap,
        fOnSelfOverlap);
}

template <auto kDims>
template <class FObjectsOverlap, class FOnOverlap>
inline void AabbKdTreeHierarchy<kDims>::Overlaps(
    SelfType const& rhs,
    FObjectsOverlap fObjectsOverlap,
    FOnOverlap fOnOverlap) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbKdTreeHierarchy.Overlaps");
    // This tree will be the left-hand side tree
    KdTreeNode const* nodes  = tree.Nodes().data();
    IndexVectorX const& perm = tree.Permutation();
    auto const fChildLhs     = [&]<auto c>(Index n) -> Index {
        if constexpr (c == 0)
            return nodes[n].Left();
        else
            return nodes[n].Right();
    };
    auto const fIsLeafLhs = [&](Index n) {
        return nodes[n].IsLeaf();
    };
    auto const fLeafSizeLhs = []([[maybe_unused]] Index n) {
        return nodes[n].n;
    };
    auto const fLeafObjectLhs = [&](Index n, [[maybe_unused]] Index i) {
        return perm(nodes[n].begin + i);
    };
    // This tree will be the right-hand side tree
    KdTreeNode const* nodesRhs  = rhs.tree.Nodes().data();
    IndexVectorX const& permRhs = rhs.tree.Permutation();
    auto const fChildRhs        = [&]<auto c>(Index n) -> Index {
        if constexpr (c == 0)
            return nodesRhs[n].Left();
        else
            return nodesRhs[n].Right();
    };
    auto const fIsLeafRhs = [&](Index n) {
        return nodesRhs[n].IsLeaf();
    };
    auto const fLeafSizeRhs = []([[maybe_unused]] Index n) {
        return nodesRhs[n].n;
    };
    auto const fLeafObjectRhs = [&](Index n, [[maybe_unused]] Index i) {
        return permRhs(nodesRhs[n].begin + i);
    };
    // Register overlaps
    geometry::Overlaps(
        fChildLhs,
        fIsLeafLhs,
        fLeafSizeLhs,
        fLeafObjectLhs,
        fChildRhs,
        fIsLeafRhs,
        fLeafSizeRhs,
        fLeafObjectRhs,
        [&](Index n1, Index n2) {
            auto L1 = IB.col(n1).head<kDims>();
            auto U1 = IB.col(n1).tail<kDims>();
            auto L2 = rhs.IB.col(n2).head<kDims>();
            auto U2 = rhs.IB.col(n2).tail<kDims>();
            return geometry::OverlapQueries::AxisAlignedBoundingBoxes(L1, U1, L2, U2);
        },
        fObjectsOverlap,
        fOnOverlap);
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_AABBKDTREEHIERARCHY_H
