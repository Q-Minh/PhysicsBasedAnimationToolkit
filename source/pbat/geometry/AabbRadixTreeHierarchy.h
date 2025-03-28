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
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/profiling/Profiling.h"

#include <cpp-sort/sorters/ska_sorter.h>
#include <numeric>

namespace pbat::geometry {

/**
 * @brief Axis-aligned radix tree hierarchy of axis-aligned bounding boxes.
 *
 * This BVH does not store the AABBs themselves, only the tree topology and the AABBs of the tree
 * nodes. The user is responsible for storing the objects and their AABBs. Doing so allows this BVH
 * implementation to support arbitrary object types.
 *
 * @note This BVH implementation relies on a binary radix tree, thus tree topology should be rebuilt
 * from scratch whenever updates are required. However, this aabb radix tree construction has
 * average time complexity of \f$ O(n) \f$ and worst-case time complexity of \f$ O(n log n) \f$.
 *
 * @tparam kDims Number of spatial dimensions
 */
template <auto kDims>
class AabbRadixTreeHierarchy
{
  public:
    using IndexType             = Index;            ///< Type of the indices
    static auto constexpr kDims = kDims;            ///< Number of spatial dimensions
    using SelfType = AabbRadixTreeHierarchy<kDims>; ///< Type of this template instantiation

    AabbRadixTreeHierarchy() = default;
    /**
     * @brief Construct an AabbRadixTreeHierarchy from an input AABB matrix B
     *
     * @tparam TDerived Type of the input matrix
     * @param L kDims x |# objects| matrix of object AABB lower bounds, such that for an object o,
     * L.col(o).head<kDims>() is the lower bound.
     * @param U kDims x |# objects| matrix of object AABB upper bounds, such that for an object o,
     * U.col(o).head<kDims>() is the upper bound.
     */
    template <class TDerivedL, class TDerivedU>
    AabbRadixTreeHierarchy(
        Eigen::DenseBase<TDerivedL> const& L,
        Eigen::DenseBase<TDerivedU> const& U);
    /**
     * @brief Construct an AabbRadixTreeHierarchy from an input AABB matrix B
     *
     * Construction has \f$ O(n log n) \f$ average time complexity due to morton code sorting.
     *
     * @tparam TDerived Type of the input matrix
     * @param L kDims x |# objects| matrix of object AABB lower bounds, such that for an object o,
     * L.col(o).head<kDims>() is the lower bound.
     * @param U kDims x |# objects| matrix of object AABB upper bounds, such that for an object o,
     * U.col(o).head<kDims>() is the upper bound.
     */
    template <class TDerivedL, class TDerivedU>
    void Construct(Eigen::DenseBase<TDerivedL> const& L, Eigen::DenseBase<TDerivedU> const& U);
    /**
     * @brief Recomputes k-D tree node AABBs given the object AABBs
     *
     * A sequential post-order traversal of the k-D tree is performed, i.e. bottom up nodal AABB
     * computation, leading to \f$ O(n) \f$ time complexity.
     *
     * @tparam TDerivedB Type of the input matrix
     * @param L kDims x |# objects| matrix of object AABB lower bounds, such that for an object o,
     * L.col(o).head<kDims>() is the lower bound.
     * @param U kDims x |# objects| matrix of object AABB upper bounds, such that for an object o,
     * U.col(o).head<kDims>() is the upper bound.
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
     * @param fObjectsOverlap Function to determine if 2 objects (o1,o2) overlap, where o1 is an
     * object from this tree and o2 is an object from the rhs tree
     * @param fOnOverlap Function to process an overlap (o1,o2) where o1 is an object from this tree
     * and o2 is an object from the rhs tree
     * @param rhs Other hierarchy to compare against
     */
    template <class FObjectsOverlap, class FOnOverlap>
    void
    Overlaps(FObjectsOverlap fObjectsOverlap, FOnOverlap fOnOverlap, SelfType const& rhs) const;

    /**
     * @brief Get the internal node bounding boxes
     * @return `2*kDims x |# internal nodes|` matrix of AABBs, such that for an internal node node,
     * IB.col(node).head<kDims>() is the lower bound and IB.col(node).tail<kDims>() is the upper
     * bound.
     */
    auto InternalNodeBoundingBoxes() const { return IB; }
    /**
     * @brief Get the underlying tree hierarchy
     * @return Binary radix tree
     */
    auto Tree() const -> common::BinaryRadixTree<IndexType> const& { return tree; }

  protected:
    /**
     * @brief Compute Morton codes for the AABBs
     *
     * @tparam TDerivedB Type of the input matrix
     * @param L kDims x |# objects| matrix of object AABB lower bounds, such that for an object o,
     * L.col(o).head<kDims>() is the lower bound.
     * @param U kDims x |# objects| matrix of object AABB upper bounds, such that for an object o,
     * U.col(o).head<kDims>() is the upper bound.
     */
    template <class TDerivedL, class TDerivedU>
    void
    ComputeMortonCodes(Eigen::DenseBase<TDerivedL> const& L, Eigen::DenseBase<TDerivedU> const& U);
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
template <class TDerivedL, class TDerivedU>
inline AabbRadixTreeHierarchy<kDims>::AabbRadixTreeHierarchy(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U)
{
    Construct(L.derived(), U.derived());
}

template <auto kDims>
template <class TDerivedL, class TDerivedU>
inline void AabbRadixTreeHierarchy<kDims>::Construct(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.Construct");
    auto const n = L.cols();
    codes.resize(n);
    inds.resize(n);
    ComputeMortonCodes(L.derived(), U.derived());
    SortMortonCodes();
    tree.Construct(codes);
    IB.resize(2 * kDims, tree.InternalNodeCount());
}

template <auto kDims>
template <class TDerivedL, class TDerivedU>
inline void AabbRadixTreeHierarchy<kDims>::Update(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.Update");
    IB.template topRows<kDims>().setConstant(std::numeric_limits<Scalar>::max());
    IB.template bottomRows<kDims>().setConstant(std::numeric_limits<Scalar>::lowest());
    common::TraverseNAryTreePseudoPostOrder(
        [&](Index n) {
            // n is guaranteed to be internal node, due to our fChild functor
            auto const lc = tree.Left(n);
            auto const rc = tree.Right(n);
            Vector<kDims> LL, LU, RL, RU;
            if (tree.IsLeaf(lc))
            {
                auto i = inds(tree.CodeIndex(lc));
                LL     = L.col(i).head<kDims>();
                LU     = U.col(i).head<kDims>();
            }
            else
            {
                LL = IB.col(lc).head<kDims>();
                LU = IB.col(lc).tail<kDims>();
            }
            if (tree.IsLeaf(rc))
            {
                auto i = inds(tree.CodeIndex(rc));
                RL     = L.col(i).head<kDims>();
                RU     = U.col(i).head<kDims>();
            }
            else
            {
                RL = IB.col(rc).head<kDims>();
                RU = IB.col(rc).tail<kDims>();
            }
            IB.col(n).head<kDims>() = LL.cwiseMin(RL);
            IB.col(n).tail<kDims>() = LU.cwiseMax(RU);
        },
        // fChild functor that only returns non-leaf children
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return tree.IsLeaf(tree.Left(n)) ? -1 : tree.Left(n);
            else
                return tree.IsLeaf(tree.Right(n)) ? -1 : tree.Right(n);
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
    geometry::Overlaps(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return tree.IsLeaf(n) ? -1 : tree.Left(n);
            else
                return tree.IsLeaf(n) ? -1 : tree.Right(n);
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
                return tree.IsLeaf(n) ? -1 : tree.Left(n);
            else
                return tree.IsLeaf(n) ? -1 : tree.Right(n);
        },
        [&](Index n) { return tree.IsLeaf(n); },
        []([[maybe_unused]] Index n) { return 1; },
        [&](Index n, [[maybe_unused]] Index i) { return inds(tree.CodeIndex(n)); },
        [&](Index n) {
            using TDerivedL  = decltype(IB.col(n).head<kDims>());
            using TDerivedU  = decltype(IB.col(n).tail<kDims>());
            using ScalarType = std::invoke_result_t<FDistanceToNode, TDerivedL, TDerivedU>;
            if (tree.IsLeaf(n))
                return ScalarType(0); // Radix tree leaf nodes correspond to individual objects
            auto L = IB.col(n).head<kDims>();
            auto U = IB.col(n).tail<kDims>();
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
                return tree.IsLeaf(n) ? -1 : tree.Left(n);
            else
                return tree.IsLeaf(n) ? -1 : tree.Right(n);
        },
        [&](Index n) { return tree.IsLeaf(n); },
        []([[maybe_unused]] Index n) { return 1; },
        [&](Index n, [[maybe_unused]] Index i) { return inds(tree.CodeIndex(n)); },
        [&](Index n) {
            using TDerivedL  = decltype(IB.col(n).head<kDims>());
            using TDerivedU  = decltype(IB.col(n).tail<kDims>());
            using ScalarType = std::invoke_result_t<FDistanceToNode, TDerivedL, TDerivedU>;
            if (tree.IsLeaf(n))
                return ScalarType(0); // Radix tree leaf nodes correspond to individual objects
            auto L = IB.col(n).head<kDims>();
            auto U = IB.col(n).tail<kDims>();
            return fDistanceToNode.template operator()<TDerivedL, TDerivedU>(L, U);
        },
        fDistanceToObject,
        fOnNearestNeighbour,
        K,
        radius);
}

template <auto kDims>
template <class FObjectsOverlap, class FOnSelfOverlap>
inline void AabbRadixTreeHierarchy<kDims>::SelfOverlaps(
    FObjectsOverlap fObjectsOverlap,
    FOnSelfOverlap fOnSelfOverlap) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.SelfOverlaps");
    geometry::SelfOverlaps(
        [&]<auto c>(Index n) -> Index {
            if constexpr (c == 0)
                return tree.IsLeaf(n) ? -1 : tree.Left(n);
            else
                return tree.IsLeaf(n) ? -1 : tree.Right(n);
        },
        [&](Index n) { return tree.IsLeaf(n); },
        []([[maybe_unused]] Index n) { return 1; },
        [&](Index n, [[maybe_unused]] Index i) { return inds(tree.CodeIndex(n)); },
        [&](Index n1, Index n2) {
            if (tree.IsLeaf(n1) or tree.IsLeaf(n2))
                return true; // Radix tree leaf nodes correspond to individual objects
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
inline void AabbRadixTreeHierarchy<kDims>::Overlaps(
    FObjectsOverlap fObjectsOverlap,
    FOnOverlap fOnOverlap,
    SelfType const& rhs) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.Overlaps");
    // This tree will be the left-hand side tree
    auto const fChildLhs = [&]<auto c>(Index n) -> Index {
        if constexpr (c == 0)
            return tree.IsLeaf(n) ? -1 : tree.Left(n);
        else
            return tree.IsLeaf(n) ? -1 : tree.Right(n);
    };
    auto const fIsLeafLhs = [&](Index n) {
        return tree.IsLeaf(n);
    };
    auto const fLeafSizeLhs = []([[maybe_unused]] Index n) {
        return 1;
    };
    auto const fLeafObjectLhs = [&](Index n, [[maybe_unused]] Index i) {
        return inds(tree.CodeIndex(n));
    };
    // This tree will be the right-hand side tree
    auto const fChildRhs = [&]<auto c>(Index n) -> Index {
        if constexpr (c == 0)
            return rhs.tree.IsLeaf(n) ? -1 : rhs.tree.Left(n);
        else
            return rhs.tree.IsLeaf(n) ? -1 : rhs.tree.Right(n);
    };
    auto const fIsLeafRhs = [&](Index n) {
        return rhs.tree.IsLeaf(n);
    };
    auto const fLeafSizeRhs = []([[maybe_unused]] Index n) {
        return 1;
    };
    auto const fLeafObjectRhs = [&](Index n, [[maybe_unused]] Index i) {
        return rhs.inds(rhs.tree.CodeIndex(n));
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
            if (tree.IsLeaf(n1) or rhs.tree.IsLeaf(n2))
                return true; // Radix tree leaf nodes correspond to individual objects
            auto L1 = IB.col(n1).head<kDims>();
            auto U1 = IB.col(n1).tail<kDims>();
            auto L2 = rhs.IB.col(n2).head<kDims>();
            auto U2 = rhs.IB.col(n2).tail<kDims>();
            return geometry::OverlapQueries::AxisAlignedBoundingBoxes(L1, U1, L2, U2);
        },
        fObjectsOverlap,
        fOnOverlap);
}

template <auto kDims>
template <class TDerivedL, class TDerivedU>
inline void AabbRadixTreeHierarchy<kDims>::ComputeMortonCodes(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.ComputeMortonCodes");
    Vector<kDims> min     = L.rowwise().minCoeff();
    Vector<kDims> max     = U.rowwise().maxCoeff();
    Vector<kDims> extents = max - min;
    auto C                = 0.5 * (L.derived() + U.derived());
    for (Index i = 0; i < L.cols(); ++i)
    {
        Vector<kDims> center;
        common::ForRange<0, kDims>([&]<auto d>() { center(d) = (C(d, i) - min(d)) / extents(d); });
        codes(i) = Morton3D(center);
    }
}

template <auto kDims>
inline void AabbRadixTreeHierarchy<kDims>::SortMortonCodes()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.AabbRadixTreeHierarchy.SortMortonCodes");
    // NOTE:
    // It would potentially be more efficient not to re-sort from scratch every time, due to
    // this std::iota call. If we kept the inds ordering between calls, we would benefit from
    // significant performance speedups of certain sorting algorithms on nearly sorted input.
    // However, this would mean that in the call to ComputeMortonCodes, we would have to
    // recompute the Morton codes in the same order as the previous call, which will not
    // exploit cache locality of iterating through B's columns. This is a trade-off that
    // should be considered if performance is critical.
    std::iota(inds.begin(), inds.end(), 0);
    cppsort::ska_sort(inds.begin(), inds.end(), [&](IndexType i) { return codes(i); });
    common::Permute(codes.begin(), codes.end(), inds.begin());
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_AABBRADIXTREEHIERARCHY_H
