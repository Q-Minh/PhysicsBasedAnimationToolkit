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
     */
    template <class TDerived>
    BinaryRadixTree(Eigen::DenseBase<TDerived> const& codes);

    /**
     * @brief Construct a Radix Tree from a sorted list of integral codes
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
     */
    template <class TDerived>
    void Construct(Eigen::DenseBase<TDerived> const& codes);
    /**
     * @brief Get the left child of an internal node
     *
     * @param i Index of the internal node
     * @return Index of the left child. If the node is a leaf, the index is offset by # internal
     * nodes.
     */
    TIndex Left(TIndex i) const { return mChild(0, i); }
    /**
     * @brief Get the right child of an internal node
     *
     * @param i Index of the internal node
     * @return Index of the right child. If the node is a leaf, the index is offset by # internal
     * nodes.
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

  private:
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
        mChild; ///< 2x|# internal nodes| matrix, s.t. mChild(0,i) ->
                ///< left child of internal node i, mChild(1,i) ->
                ///< right child of internal node i
    Eigen::Vector<TIndex, Eigen::Dynamic> mParent; ///< |# internal +leaf nodes| vector, s.t.
                                                   ///< mParent(i) -> parent of node i
};

template <class TIndex>
template <class TDerived>
inline BinaryRadixTree<TIndex>::BinaryRadixTree(Eigen::DenseBase<TDerived> const& codes)
{
    Construct(codes.derived());
}

template <class TIndex>
template <class TDerived>
inline void BinaryRadixTree<TIndex>::Construct(Eigen::DenseBase<TDerived> const& codes)
{
    using CodeType = typename TDerived::Scalar;
    static_assert(
        std::is_integral_v<CodeType> and not std::is_same_v<CodeType, bool>,
        "Codes must be integral");

    TIndex const nLeaves   = codes.size();
    TIndex const nInternal = nLeaves - 1;
    mChild.resize(2, nInternal);
    mParent.resize(nLeaves + nInternal);

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
    // Loop over internal nodes
    while (not stack.IsEmpty())
    {
        Node const node = stack.Pop();
        // Compute range [first, last] of codes covered by the node.
        // If the node is a left child, its range is reversed (i.e. end -> begin).
        // If the node is a right child, its range is not reversed (i.e. begin -> end).
        bool bReversed = node.begin > node.end;
        auto first     = (not bReversed) * node.begin + bReversed * node.end;
        auto last      = (not bReversed) * node.end + bReversed * node.begin;
        // Find the split position
        TIndex split = first;
        auto cfirst  = codes(first);
        auto clast   = codes(last);
        // If first and last codes are the same, all the codes in between are the same
        // and we can split the node right down the middle without binary search.
        if (cfirst == clast)
        {
            split += ((last - first + 1) >> 1);
        }
        // Otherwise, we perform a binary search to find the split position.
        else
        {
            CodeType const mask =
                ~CodeType(0) /*bitwise ones*/ >> fCommonPrefixLength(cfirst, clast);
            auto begin = codes.begin() + first;
            auto end   = codes.begin() + last + 1;
            split += std::distance(
                begin,
                std::upper_bound(begin, end, cfirst, [&](CodeType ci, CodeType cj) {
                    return (mask & ci) < (mask & cj);
                }));
        }
        // The left and right child ranges are split as [first, split-1] and [split,last],
        // respectively.
        TIndex lc = split - 1;
        TIndex rc = split;
        // Ranges of size 1 indicate leaf nodes. We offset leaf indices by the number of internal
        // nodes.
        bool bIsLeftLeaf  = (lc == first);
        bool bIsRightLeaf = (rc == last);
        lc += bIsLeftLeaf * nInternal;
        rc += bIsRightLeaf * nInternal;
        // Set parent-child relationships
        mChild(0, node.begin) = lc;
        mChild(1, node.begin) = rc;
        mParent(lc)           = node.begin;
        mParent(rc)           = node.begin;
        // Only recurse into internal nodes
        if (not bIsLeftLeaf)
            stack.Push({lc, first});
        if (not bIsRightLeaf)
            stack.Push({rc, last});
    }
    mParent(0) = 0;
}

} // namespace pbat::common

#endif // PBAT_COMMON_BINARYRADIXTREE_H
