/**
 * @file BinaryRadixTree.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Radix Tree implementation
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_COMMON_BINARYRADIXTREE_H
#define PBAT_COMMON_BINARYRADIXTREE_H

#include "pbat/Aliases.h"
#include "pbat/common/Stack.h"

#include <algorithm>
#include <bit>

namespace pbat::common {

/**
 * @brief Binary radix tree implementation
 *
 * @tparam TIndex Type of the indices to use for tree topology
 */
template <class TIndex = Index>
class BinaryRadixTree
{
  public:
    BinaryRadixTree() = default;
    /**
     * @brief Construct a new Binary Radix Tree object
     *
     * @tparam TDerived Type of the Eigen expression for codes
     * @param codes Sorted list of integral codes
     * @param bStoreParent Store parent indices/relationships if true
     */
    template <class TDerived>
    BinaryRadixTree(Eigen::DenseBase<TDerived> const& codes, bool bStoreParent = false);

    /**
     * @brief Construct a Radix Tree from a sorted list of unsigned integral codes
     *
     * Implementation has average linear time complexity \f$ O(n) \f$, where \f$ n \f$ is the number
     * of codes/leaves. We do not prove the worst case time complexity, but it is most probably the
     * same as the average case.
     *
     * We visit \f$ n-1 \f$ internal nodes of the tree during construction.
     * At each internal node \f$ i \f$, a \f$ O(\log(n)) \f$ binary search is performed over the
     * range of codes covered by that internal node. Assuming we always split ranges in the middle,
     * the work done at each level k of the tree is \f$ 2^k \log(\frac{n}{2^k}) \f$.
     * The total work is thus
     * \f[
     * \sum_{k=0}^{\log(n)} 2^k \log(\frac{n}{2^k}) = \sum_{k=0}^{\log(n)} 2^k \log(n) -
     * \sum_{k=0}^{\log(n)} 2^k \log(2^k)
     * \f]
     *
     * The first term is a constant \f$ \log(n) \f$ times the sum of a geometric series, which is
     * \f$ 2^{\log(n)+1} - 1 = 2n - 1 \f$, leading to
     * \f[
     * \log(n) (2n - 1)
     * \f]
     *
     * The second term can be written as the series
     * \f[
     * S = 0 \cdot 2^0 + 1 \cdot 2^1 + 2 \cdot 2^2 + \ldots + \log(n) \cdot 2^{\log(n)}
     * \f]
     *
     * We can rewrite this as
     * \f[
     * 2S - S = -2^1 - 2^2 - 2^3 + \ldots - 2^{\log(n)} + \log(n) 2^{\log(n)+1}
     * \f]
     *
     * The negative terms form a similar geometric series
     * \f[
     * -\sum_{k=0}^{\log(n)} 2^k = -(2n - 1)
     * \f]
     * while the positive term can be simplified to
     * \f[
     * \log(n) 2^{\log(n)+1} = \log(n) 2n
     * \f]
     *
     * Subtracting the 2nd term from the first leads to
     * \f[
     * \log(n) (2n - 1) - \log(n) 2n + 2n - 1 = 2n - \log(n) - 1
     * \f]
     *
     * Thus, the time complexity is \f$ O(n) \f$.
     *
     * @tparam TDerived Type of the Eigen expression for codes
     * @param codes Sorted list of integral codes
     * @param bStoreParent Store parent indices/relationships if true
     */
    template <class TDerived>
    void Construct(Eigen::DenseBase<TDerived> const& codes, bool bStoreParent = false);
    /**
     * @brief Get the left child of internal node `i`
     *
     * @param i Index of the internal node
     * @return Index of the left child.
     * @pre `IsLeaf(i)` is false
     */
    TIndex Left(TIndex i) const { return mChild(0, i); }
    /**
     * @brief Get the right child of internal node `i`
     *
     * @param i Index of the internal node
     * @return Index of the right child
     * @pre `IsLeaf(i)` is false
     */
    TIndex Right(TIndex i) const { return mChild(1, i); }
    /**
     * @brief Get the parent of a node
     *
     * @param i Index of the node
     * @return Index of the parent
     */
    TIndex Parent(TIndex i) const { return mParent(i); }
    /**
     * @brief Get the number of internal nodes
     *
     * @return Number of internal nodes
     */
    TIndex InternalNodeCount() const { return mChild.cols(); }
    /**
     * @brief Get the number of leaf nodes
     *
     * @return Number of leaf nodes
     */
    TIndex LeafCount() const { return mParent.size() - InternalNodeCount(); }
    /**
     * @brief Check if a node is a leaf
     *
     * @param i Index of the node
     * @return true if the node is a leaf, false otherwise
     */
    bool IsLeaf(TIndex i) const { return i >= InternalNodeCount(); }
    /**
     * @brief Get the root of the tree
     *
     * @return Index of the root node
     */
    constexpr TIndex Root() const { return 0; }
    /**
     * @brief Get the index of the code associated with a leaf node
     *
     * @param leaf Index of the leaf node
     * @return Index of the code
     */
    TIndex CodeIndex(TIndex leaf) const { return leaf - InternalNodeCount(); }
    /**
     * @brief Get the index of the leaf node associated with a code
     *
     * @param codeIdx Index of the code
     * @return Index of the leaf node
     */
    TIndex LeafIndex(TIndex codeIdx) const { return codeIdx + InternalNodeCount(); }
    /**
     * @brief Get the children array of the tree
     *
     * @return `2 x |# internal nodes|` matrix `c`, s.t. `c(0,i)` -> left child of internal node
     * `i`, `c(1,i)` -> right child of internal node `i`
     */
    auto Children() const { return mChild; }
    /**
     * @brief Left child array of the tree
     *
     * @return `|# internal nodes|` vector `l`, s.t. `l(i)` -> left child of internal node `i`
     */
    auto Left() const { return mChild.row(0); }
    /**
     * @brief Right child array of the tree
     *
     * @return `|# internal nodes|` vector `r`, s.t. `r(i)` -> right child of internal node `i`
     */
    auto Right() const { return mChild.row(1); }
    /**
     * @brief Get the parent array of the tree
     *
     * @note The root node is its own parent, i.e. `p(0) == 0` is true
     * @return `|# internal + leaf nodes|` vector `p`, s.t. `p(i)` -> parent of node `i`
     */
    auto Parent() const { return mParent; }
    /**
     * @brief Check if the tree has parent relationships
     *
     * @return true if the tree has parent relationships, false otherwise
     */
    bool HasParentRelationship() const { return mParent.size() > 0; }

  private:
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
        mChild; ///< `2 x |# internal nodes|` matrix, s.t. `mChild(0,i)` ->
                ///< left child of internal node `i`, `mChild(1,i)` ->
                ///< right child of internal node `i`
    Eigen::Vector<TIndex, Eigen::Dynamic> mParent; ///< `|# internal + leaf nodes|` vector, s.t.
                                                   ///< `mParent(i)` -> parent of node `i`
};

template <class TIndex>
template <class TDerived>
inline BinaryRadixTree<TIndex>::BinaryRadixTree(
    Eigen::DenseBase<TDerived> const& codes,
    bool bStoreParent)
{
    Construct(codes.derived(), bStoreParent);
}

template <class TIndex>
template <class TDerived>
inline void
BinaryRadixTree<TIndex>::Construct(Eigen::DenseBase<TDerived> const& codes, bool bStoreParent)
{
    using CodeType = typename TDerived::Scalar;
    static_assert(
        std::is_integral_v<CodeType> and std::is_unsigned_v<CodeType> and
            not std::is_same_v<CodeType, bool>,
        "Codes must be of unsigned integral type");
    auto constexpr nBits = sizeof(CodeType) * 8;
    static_assert(nBits <= 64, "CodeType must have at most 64 bits");

    std::uint64_t constexpr msb = 0b1ULL << (nBits - 1);
    TIndex const nLeaves        = codes.size();
    TIndex const nInternal      = nLeaves - 1;
    mChild.resize(2, nInternal);
    mParent.resize(bStoreParent * (nLeaves + nInternal));

    struct Node
    {
        TIndex begin;
        TIndex end;
    };
    auto const fCommonPrefixLength = [](CodeType ci, CodeType cj) {
        return std::countl_zero(ci ^ cj);
    };
    pbat::common::Stack<Node, 64> stack{};
    stack.Push({0, nLeaves - 1});
    // Top-down construction over internal nodes
    while (not stack.IsEmpty())
    {
        Node const node = stack.Pop();
        // Compute range [first, last] of codes covered by the node.
        // If the node is a left child, its range is reversed (i.e. end -> begin).
        // If the node is a right child, its range is not reversed (i.e. begin -> end).
        bool const bReversed = node.begin > node.end;
        auto const first     = (not bReversed) * node.begin + bReversed * node.end;
        auto const last      = (not bReversed) * node.end + bReversed * node.begin;
        // Find the split position
        auto const cfirst  = codes(first);
        auto const clast   = codes(last);
        auto const mask    = msb >> fCommonPrefixLength(cfirst, clast);
        auto const begin   = codes.begin() + first;
        auto const end     = codes.begin() + last + 1;
        auto const upper   = std::upper_bound(begin, end, cfirst, [&](CodeType ci, CodeType cj) {
            return (mask & ci) < (mask & cj);
        });
        TIndex const split = first + std::distance(begin, upper);
        // The left and right child ranges are split as [first, split-1] and [split,last],
        // respectively.
        TIndex lc = split - 1;
        TIndex rc = split;
        // Ranges of size 1 indicate leaf nodes. We offset leaf indices by the number of internal
        // nodes.
        bool const bIsLeftLeaf  = (lc == first);
        bool const bIsRightLeaf = (rc == last);
        lc += bIsLeftLeaf * nInternal;
        rc += bIsRightLeaf * nInternal;
        // Set parent-child relationships
        mChild(0, node.begin) = lc;
        mChild(1, node.begin) = rc;
        // Only recurse into internal nodes
        if (not bIsLeftLeaf)
            stack.Push({lc, first});
        if (not bIsRightLeaf)
            stack.Push({rc, last});
    }
    // Store parent relationships if requested
    if (bStoreParent)
    {
        mParent(0) = 0;
        for (auto i = 0; i < nInternal; ++i)
        {
            mParent(Left(i))  = i;
            mParent(Right(i)) = i;
        }
    }
}

} // namespace pbat::common

#endif // PBAT_COMMON_BINARYRADIXTREE_H
