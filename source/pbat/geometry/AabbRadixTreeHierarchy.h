#ifndef PBAT_GEOMETRY_AABBRADIXTREEHIERARCHY_H
#define PBAT_GEOMETRY_AABBRADIXTREEHIERARCHY_H

#include "AxisAlignedBoundingBox.h"
#include "SpatialSearch.h"
#include "pbat/Aliases.h"
#include "pbat/common/BinaryRadixTree.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/NAryTreeTraversal.h"
#include "pbat/common/Permute.h"
#include "pbat/geometry/Morton.h"
#include "pbat/profiling/Profiling.h"

#include <cpp-sort/sorters/ska_sorter.h>
#include <numeric>

namespace pbat::geometry {

template <auto kDims>
class AabbRadixTreeHierarchy
{
  public:
    using IndexType = Index; ///< Type of the indices

    AabbRadixTreeHierarchy() = default;
    /**
     * @brief Construct an AabbRadixTreeHierarchy from an input AABB matrix B
     *
     * @tparam TDerived Type of the input matrix
     * @param B 2*kDims x |# objects| matrix of object AABBs, such that for an object o,
     * B.col(o).head<kDims>() is the lower bound and B.col(o).tail<kDims>() is the upper bound.
     */
    template <class TDerived>
    AabbRadixTreeHierarchy(Eigen::DenseBase<TDerived> const& B);
    /**
     * @brief Construct an AabbRadixTreeHierarchy from an input AABB matrix B
     *
     * Construction has \f$ O(n log n) \f$ average time complexity due to morton code sorting.
     *
     * @tparam TDerived Type of the input matrix
     * @param B 2*kDims x |# objects| matrix of object AABBs, such that for an object o,
     * B.col(o).head<kDims>() is the lower bound and B.col(o).tail<kDims>() is the upper bound.
     */
    template <class TDerived>
    void Construct(Eigen::DenseBase<TDerived> const& B);
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

  protected:
    /**
     * @brief Compute Morton codes for the AABBs
     *
     * @tparam TDerivedB Type of the input matrix
     * @param B 2*kDims x |# objects| matrix of object AABBs, such that for an object o,
     * B.col(o).head<kDims>() is the lower bound and B.col(o).tail<kDims>() is the upper bound.
     */
    template <class TDerivedB>
    void ComputeMortonCodes(Eigen::DenseBase<TDerivedB> const& B);
    /**
     * @brief Sort the Morton codes
     */
    void SortMortonCodes();

  private:
    Eigen::Vector<MortonCodeType, Eigen::Dynamic> codes; ///< Morton codes of the AABBs
    Eigen::Vector<IndexType, Eigen::Dynamic> inds;       ///< |# codes| sorted ordering
    Matrix<2 * kDims, Eigen::Dynamic>
        IB; ///< 2*kDims x |# internal nodes| matrix of AABBs, such that
            ///< for a node node, IB.col(node).head<kDims>() is the lower
            ///< bound and IB.col(node).tail<kDims>() is the upper bound.
    common::BinaryRadixTree<IndexType> tree; ///< KdTree over the AABBs
};

template <auto kDims>
template <class TDerived>
inline AabbRadixTreeHierarchy<kDims>::AabbRadixTreeHierarchy(Eigen::DenseBase<TDerived> const& B)
{
    Construct(B.derived());
}

template <auto kDims>
template <class TDerived>
inline void AabbRadixTreeHierarchy<kDims>::Construct(Eigen::DenseBase<TDerived> const& B)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.Construct");
    auto const n = B.cols();
    codes.resize(n);
    inds.resize(n);
    ComputeMortonCodes(B);
    SortMortonCodes();
    tree.Construct(codes);
    IB.resize(2 * kDims, tree.InternalNodeCount());
    Update(B);
}

template <auto kDims>
template <class TDerivedB>
inline void AabbRadixTreeHierarchy<kDims>::Update(Eigen::DenseBase<TDerivedB> const& B)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.Update");
    IB.template topRows<kDims>().setConstant(std::numeric_limits<Scalar>::max());
    IB.template bottomRows<kDims>().setConstant(std::numeric_limits<Scalar>::lowest());
    common::TraverseNAryTreePseudoPostOrder(
        [&](Index n) {
            if (tree.IsLeaf(n))
            {
                auto i             = inds(tree.CodeIndex(n));
                auto nbox          = B.col(i);
                auto pbox          = IB.col(tree.Parent(n));
                pbox.head<kDims>() = pbox.head<kDims>().cwiseMin(nbox.head<kDims>());
                pbox.tail<kDims>() = pbox.tail<kDims>().cwiseMax(nbox.tail<kDims>());
            }
            else
            {
                auto nbox = IB.col(n);
                auto pbox = IB.col(
                    tree.Parent(n)); // tree.Parent(tree.Root()) == tree.Root(), so this is safe
                pbox.head<kDims>() = pbox.head<kDims>().cwiseMin(nbox.head<kDims>());
                pbox.tail<kDims>() = pbox.tail<kDims>().cwiseMax(nbox.tail<kDims>());
            }
        },
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return tree.Left(n);
            else
                return tree.Right(n);
        });
}

template <auto kDims>
template <class FNodeOverlaps, class FObjectOverlaps, class FOnOverlap>
inline void AabbRadixTreeHierarchy<kDims>::Overlaps(
    FNodeOverlaps fNodeOverlaps,
    FObjectOverlaps fObjectOverlaps,
    FOnOverlap fOnOverlap) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.Overlaps");
    common::TraverseNAryTreePseudoPostOrder(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return tree.Left(n);
            else
                return tree.Right(n);
        },
        [&](Index n) { return tree.IsLeaf(n); },
        []([[maybe_unused]] Index n) { return 1; },
        [&](Index n, [[maybe_unused]] Index i) { return inds(tree.CodeIndex(n)); },
        [&](Index n) {
            if (tree.IsLeaf(n))
                return true; // Radix tree leaf nodes correspond to individual objects
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
inline void AabbRadixTreeHierarchy<kDims>::NearestNeighbour(
    FDistanceToNode fDistanceToNode,
    FDistanceToObject fDistanceToObject,
    FOnNearestNeighbour fOnNearestNeighbour,
    Scalar radius,
    Scalar eps) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.NearestNeighbour");
    geometry::NearestNeighbour(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return tree.Left(n);
            else
                return tree.Right(n);
        },
        [&](Index n) { return tree.IsLeaf(n); },
        []([[maybe_unused]] Index n) { return 1; },
        [&](Index n, [[maybe_unused]] Index i) { return inds(tree.CodeIndex(n)); },
        [&](Index n) {
            auto L           = IB.col(n).head<kDims>();
            auto U           = IB.col(n).tail<kDims>();
            using TDerivedL  = decltype(L);
            using TDerivedU  = decltype(U);
            using ScalarType = std::invoke_result_t<FDistanceToNode, TDerivedL, TDerivedU>;
            if (tree.IsLeaf(n))
                return ScalarType(0); // Radix tree leaf nodes correspond to individual objects
            return fDistanceToNode.template operator()<TDerivedL, TDerivedU>(L, U);
        },
        fDistanceToObject,
        fOnNearestNeighbour,
        radius,
        eps);
}

template <auto kDims>
template <class FDistanceToNode, class FDistanceToObject, class FOnNearestNeighbour>
inline void AabbRadixTreeHierarchy<kDims>::KNearestNeighbours(
    FDistanceToNode fDistanceToNode,
    FDistanceToObject fDistanceToObject,
    FOnNearestNeighbour fOnNearestNeighbour,
    Index K,
    Scalar radius) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.KNearestNeighbours");
    geometry::KNearestNeighbours(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return tree.Left(n);
            else
                return tree.Right(n);
        },
        [&](Index n) { return tree.IsLeaf(n); },
        []([[maybe_unused]] Index n) { return 1; },
        [&](Index n, [[maybe_unused]] Index i) { return inds(tree.CodeIndex(n)); },
        [&](Index n) {
            auto L           = IB.col(n).head<kDims>();
            auto U           = IB.col(n).tail<kDims>();
            using TDerivedL  = decltype(L);
            using TDerivedU  = decltype(U);
            using ScalarType = std::invoke_result_t<FDistanceToNode, TDerivedL, TDerivedU>;
            if (tree.IsLeaf(n))
                return ScalarType(0); // Radix tree leaf nodes correspond to individual objects
            return fDistanceToNode.template operator()<TDerivedL, TDerivedU>(L, U);
        },
        fDistanceToObject,
        fOnNearestNeighbour,
        K,
        radius);
}

template <auto kDims>
template <class TDerivedB>
inline void AabbRadixTreeHierarchy<kDims>::ComputeMortonCodes(Eigen::DenseBase<TDerivedB> const& B)
{
    Vector<kDims> min     = B.template topRows<kDims>().rowwise().minCoeff();
    auto max              = B.template bottomRows<kDims>().rowwise().maxCoeff();
    Vector<kDims> extents = max - min;
    for (Index i = 0; i < B.cols(); ++i)
    {
        Vector<kDims> centroid =
            0.5 * (B.col(i).template head<kDims>() + B.col(i).template tail<kDims>());
        common::ForRange<0, kDims>(
            [&]<auto d>() { centroid(d) = (centroid(d) - min(d)) / extents(d); });
        codes(i) = Morton3D(centroid);
    }
}

template <auto kDims>
inline void AabbRadixTreeHierarchy<kDims>::SortMortonCodes()
{
    std::iota(inds.begin(), inds.end(), 0);
    cppsort::ska_sort(inds.begin(), inds.end(), [&](IndexType i) { return codes(i); });
    common::Permute(codes.begin(), codes.end(), inds.begin());
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_AABBRADIXTREEHIERARCHY_H
