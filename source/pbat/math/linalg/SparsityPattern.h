#ifndef PBAT_MATH_LINALG_SPARSITYPATTERN_H
#define PBAT_MATH_LINALG_SPARSITYPATTERN_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/profiling/Profiling.h"

#include <exception>
#include <fmt/core.h>
#include <range/v3/action/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/unique.hpp>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace pbat {
namespace math {
namespace linalg {

/**
 * @brief Sparsity pattern precomputer to accelerate sparse matrix assembly.
 * @tparam TScalar Scalar type for the matrix values
 * @tparam TIndex Index type for the matrix indices
 * @tparam StorageOptions Storage options for the Eigen sparse matrix
 */
template <typename TIndex = Index, Eigen::StorageOptions Options = Eigen::ColMajor>
class SparsityPattern
{
  public:
    using IndexType = TIndex; ///< Index type for the matrix indices
    using InnerPatternType =
        Eigen::SparseMatrix<std::uint8_t, Options, TIndex>; ///< Type of the inner pattern

    PBAT_API SparsityPattern() = default;

    template <
        common::CContiguousIndexRange TRowIndexRange,
        common::CContiguousIndexRange TColIndexRange>
    SparsityPattern(
        IndexType nRows,
        IndexType nCols,
        TRowIndexRange&& rowIndices,
        TColIndexRange&& colIndices);

    template <
        common::CContiguousIndexRange TRowIndexRange,
        common::CContiguousIndexRange TColIndexRange>
    void Compute(
        IndexType nRows,
        IndexType nCols,
        TRowIndexRange&& rowIndices,
        TColIndexRange&& colIndices);

    template <common::CArithmeticRange TNonZeroRange>
    auto ToMatrix(TNonZeroRange&& nonZeros) const
        -> Eigen::SparseMatrix<std::ranges::range_value_t<TNonZeroRange>, Options, TIndex>;

    template <common::CArithmeticRange TNonZeroRange, class TDerived>
    void To(TNonZeroRange&& nonZeros, Eigen::SparseCompressedBase<TDerived>& Ain) const;

    template <common::CArithmeticRange TNonZeroRange, class TDerived>
    void AddTo(TNonZeroRange&& nonZeros, Eigen::SparseCompressedBase<TDerived>& Ain) const;

    PBAT_API bool IsEmpty() const { return mNonZeroIndexToValueIndex.empty(); }

    template <common::CFloatingPoint TScalar>
    auto Pattern() const -> Eigen::SparseMatrix<TScalar, Options, TIndex>;

  private:
    std::vector<IndexType>
        mNonZeroIndexToValueIndex;  ///< `|# non zeros|` map from non-zero index to value index
    InnerPatternType mInnerPattern; ///< Inner pattern of the matrix
};

template <typename TIndex, Eigen::StorageOptions Options>
template <
    common::CContiguousIndexRange TRowIndexRange,
    common::CContiguousIndexRange TColIndexRange>
SparsityPattern<TIndex, Options>::SparsityPattern(
    IndexType nRows,
    IndexType nCols,
    TRowIndexRange&& rowIndices,
    TColIndexRange&& colIndices)
{
    Compute(
        nRows,
        nCols,
        std::forward<TRowIndexRange>(rowIndices),
        std::forward<TColIndexRange>(colIndices));
}

template <typename TIndex, Eigen::StorageOptions Options>
template <
    common::CContiguousIndexRange TRowIndexRange,
    common::CContiguousIndexRange TColIndexRange>
inline void SparsityPattern<TIndex, Options>::Compute(
    IndexType nRows,
    IndexType nCols,
    TRowIndexRange&& rowIndices,
    TColIndexRange&& colIndices)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.math.linalg.SparsityPattern.Compute");
    namespace srng        = std::ranges;
    namespace sviews      = std::views;
    namespace rng         = ranges;
    namespace views       = rng::views;
    using IndexVectorType = Eigen::Vector<IndexType, Eigen::Dynamic>;

    auto const nRowIndices = srng::size(rowIndices);
    auto const nColIndices = srng::size(colIndices);
    if (nRowIndices != nColIndices)
    {
        std::string const what = fmt::format(
            "Expected same number of row indices and column indices, but got {} row indices "
            "and {} column indices",
            nRowIndices,
            nColIndices);
        throw std::invalid_argument(what);
    }
    auto const [rowMin, rowMax] = srng::minmax_element(rowIndices);
    auto const [colMin, colMax] = srng::minmax_element(colIndices);
    bool const bRowsInBounds    = (*rowMin >= 0) && (*rowMax < nRows);
    bool const bColsInBounds    = (*colMin >= 0) && (*colMax < nCols);
    if (not(bRowsInBounds and bColsInBounds))
    {
        std::string const what = fmt::format(
            "Out of bounds min (row,col)=({},{}) and max (row,col)=({},{}), while "
            "(nRows,nCols)=({},{})",
            *rowMin,
            *colMin,
            *rowMax,
            *colMax,
            nRows,
            nCols);
        throw std::invalid_argument(what);
    }

    auto const nnz = nRowIndices;
    mNonZeroIndexToValueIndex.resize(nnz, IndexType{0});

    auto rows = srng::data(rowIndices);
    auto cols = srng::data(colIndices);

    bool constexpr bIsRowMajor = (Options == Eigen::RowMajor);
    auto const indexPair       = [&](Index s) {
        if constexpr (bIsRowMajor)
            return std::make_pair(rows[s], cols[s]);
        else
            return std::make_pair(cols[s], rows[s]);
    };
    auto const less = [&](IndexType lhs, IndexType rhs) {
        return indexPair(lhs) < indexPair(rhs);
    };
    auto const equal = [&](IndexType lhs, IndexType rhs) {
        return indexPair(lhs) == indexPair(rhs);
    };
    auto const numNonZeroIndices    = static_cast<IndexType>(nnz);
    auto const sortedNonZeroIndices = views::iota(IndexType{0}, numNonZeroIndices) |
                                      rng::to<std::vector>() | rng::actions::sort(less);
    auto const sortedUniqueNonZeroIndices =
        sortedNonZeroIndices | views::unique(equal) | rng::to<std::vector>();

    Eigen::Vector<IndexType, Eigen::Dynamic> innerSizes;
    innerSizes.setZero(bIsRowMajor ? nRows : nCols);
    if constexpr (bIsRowMajor)
    {
        for (auto u : sortedUniqueNonZeroIndices)
            ++innerSizes[rows[u]];
    }
    else
    {
        for (auto u : sortedUniqueNonZeroIndices)
            ++innerSizes[cols[u]];
    }

    for (auto k = 0, u = 0; k < numNonZeroIndices; ++k)
    {
        auto const s            = sortedNonZeroIndices[static_cast<std::size_t>(k)];
        auto const uu           = sortedUniqueNonZeroIndices[static_cast<std::size_t>(u)];
        bool const bIsSameEntry = equal(s, uu);
        mNonZeroIndexToValueIndex[static_cast<std::size_t>(s)] =
            static_cast<IndexType>(bIsSameEntry ? u : ++u);
    }
    mInnerPattern.resize(nRows, nCols);
    mInnerPattern.reserve(innerSizes);
    for (auto s : sortedUniqueNonZeroIndices)
        mInnerPattern.insert(rows[s], cols[s]) = 0;
    mInnerPattern.makeCompressed();
}

template <typename TIndex, Eigen::StorageOptions Options>
template <common::CArithmeticRange TNonZeroRange>
inline auto SparsityPattern<TIndex, Options>::ToMatrix(TNonZeroRange&& nonZeros) const
    -> Eigen::SparseMatrix<std::ranges::range_value_t<TNonZeroRange>, Options, TIndex>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.math.linalg.SparsityPattern.ToMatrix");
    using ScalarType = std::ranges::range_value_t<TNonZeroRange>;
    auto Acpy        = Pattern<ScalarType>();
    AddTo(std::forward<TNonZeroRange>(nonZeros), Acpy);
    return Acpy;
}

template <typename TIndex, Eigen::StorageOptions Options>
template <common::CArithmeticRange TNonZeroRange, class TDerived>
inline void SparsityPattern<TIndex, Options>::AddTo(
    TNonZeroRange&& nonZeros,
    Eigen::SparseCompressedBase<TDerived>& Ain) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.math.linalg.SparsityPattern.AddTo");
    static_assert(
        std::is_same_v<Scalar, std::ranges::range_value_t<TNonZeroRange>>,
        "Only Scalar non-zero values are accepted");

    namespace rng  = std::ranges;
    auto const nnz = rng::size(nonZeros);
    if (nnz != mNonZeroIndexToValueIndex.size())
    {
        std::string const what =
            fmt::format("Expected {} non zeros, got {}", mNonZeroIndexToValueIndex.size(), nnz);
        throw std::invalid_argument(what);
    }

    Scalar* values = Ain.valuePtr();
    for (auto k = 0ULL; k < nnz; ++k)
        values[mNonZeroIndexToValueIndex[k]] += nonZeros[k];
}

template <typename TIndex, Eigen::StorageOptions Options>
template <common::CArithmeticRange TNonZeroRange, class TDerived>
inline void SparsityPattern<TIndex, Options>::To(
    TNonZeroRange&& nonZeros,
    Eigen::SparseCompressedBase<TDerived>& Ain) const
{
    Ain.coeffs().setZero();
    AddTo(std::forward<TNonZeroRange>(nonZeros), Ain);
}

template <typename TIndex, Eigen::StorageOptions Options>
template <common::CFloatingPoint TScalar>
inline auto SparsityPattern<TIndex, Options>::Pattern() const
    -> Eigen::SparseMatrix<TScalar, Options, TIndex>
{
    return mInnerPattern.template cast<TScalar>();
}

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_SPARSITYPATTERN_H
