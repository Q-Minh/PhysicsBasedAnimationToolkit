#ifndef PBAT_MATH_LINALG_SPARSITY_PATTERN_H
#define PBAT_MATH_LINALG_SPARSITY_PATTERN_H

#include "PhysicsBasedAnimationToolkitExport.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Concepts.h>
#include <pbat/profiling/Profiling.h>
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

class SparsityPattern
{
  public:
    PBAT_API SparsityPattern() = default;

    template <
        common::CContiguousIndexRange TRowIndexRange,
        common::CContiguousIndexRange TColIndexRange>
    SparsityPattern(
        Index nRows,
        Index nCols,
        TRowIndexRange&& rowIndices,
        TColIndexRange&& colIndices);

    template <
        common::CContiguousIndexRange TRowIndexRange,
        common::CContiguousIndexRange TColIndexRange>
    void
    Compute(Index nRows, Index nCols, TRowIndexRange&& rowIndices, TColIndexRange&& colIndices);

    template <common::CArithmeticRange TNonZeroRange>
    CSCMatrix ToMatrix(TNonZeroRange&& nonZeros) const;

    PBAT_API bool IsEmpty() const;

  private:
    std::vector<Index> ij; ///< Maps (triplet/duplicate) non-zero index k to its corresponding index
                           ///< into the unique non-zero list
    CSCMatrix A;           ///< Sparsity pattern + unique non-zeros
};

template <
    common::CContiguousIndexRange TRowIndexRange,
    common::CContiguousIndexRange TColIndexRange>
SparsityPattern::SparsityPattern(
    Index nRows,
    Index nCols,
    TRowIndexRange&& rowIndices,
    TColIndexRange&& colIndices)
{
    Compute(
        nRows,
        nCols,
        std::forward<TRowIndexRange>(rowIndices),
        std::forward<TColIndexRange>(colIndices));
}

template <
    common::CContiguousIndexRange TRowIndexRange,
    common::CContiguousIndexRange TColIndexRange>
inline void SparsityPattern::Compute(
    Index nRows,
    Index nCols,
    TRowIndexRange&& rowIndices,
    TColIndexRange&& colIndices)
{
    PBAT_PROFILE_NAMED_SCOPE("math::linalg::SparsityPattern::Compute");
    namespace srng   = std::ranges;
    namespace sviews = std::views;
    namespace rng    = ranges;
    namespace views  = rng::views;

    A = CSCMatrix(nRows, nCols);
    ij.resize(srng::size(rowIndices), Index{0});

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
            *rowMax,
            nRows,
            nCols);
        throw std::invalid_argument(what);
    }

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
    auto rows = srng::data(rowIndices);
    auto cols = srng::data(colIndices);

    auto const indexPair = [&](Index s) {
        return std::make_pair(cols[s], rows[s]);
    };
    auto const less = [&](Index lhs, Index rhs) {
        return indexPair(lhs) < indexPair(rhs);
    };
    auto const equal = [&](Index lhs, Index rhs) {
        return indexPair(lhs) == indexPair(rhs);
    };
    auto const numNonZeroIndices = static_cast<Index>(nRowIndices);
    // NOTE: Bottleneck is the sort
    auto const sortedNonZeroIndices = views::iota(Index{0}, numNonZeroIndices) |
                                      rng::to<std::vector>() | rng::actions::sort(less);
    auto const sortedUniqueNonZeroIndices =
        sortedNonZeroIndices | views::unique(equal) | rng::to<std::vector>();

    IndexVectorX cc = IndexVectorX::Zero(nCols);
    for (auto u : sortedUniqueNonZeroIndices)
        ++cc[cols[u]];

    for (auto k = 0, u = 0; k < numNonZeroIndices; ++k)
    {
        // NOTE: Yes, the casting is just absurd here... otherwise code
        // doesn't compile with agressive warnings on signed/unsigned mismatch
        auto const s                    = sortedNonZeroIndices[static_cast<std::size_t>(k)];
        auto const uu                   = sortedUniqueNonZeroIndices[static_cast<std::size_t>(u)];
        bool const bIsSameEntry         = equal(s, uu);
        ij[static_cast<std::size_t>(s)] = static_cast<Index>(bIsSameEntry ? u : ++u);
    }
    A.reserve(cc);
    for (auto s : sortedUniqueNonZeroIndices)
        A.insert(rows[s], cols[s]) = 0.;
    A.makeCompressed();
}

template <common::CArithmeticRange TNonZeroRange>
CSCMatrix SparsityPattern::ToMatrix(TNonZeroRange&& nonZeros) const
{
    PBAT_PROFILE_NAMED_SCOPE("math::linalg::SparsityPattern::ToMatrix");
    static_assert(
        std::is_same_v<Scalar, std::ranges::range_value_t<TNonZeroRange>>,
        "Only Scalar non-zero values are accepted");

    namespace rng  = std::ranges;
    auto const nnz = rng::size(nonZeros);
    if (nnz != ij.size())
    {
        std::string const what = fmt::format("Expected {} non zeros, got {}", ij.size(), nnz);
        throw std::invalid_argument(what);
    }

    CSCMatrix Acpy{A};
    Scalar* values = Acpy.valuePtr();
    for (auto k = 0ULL; k < nnz; ++k)
        values[ij[k]] += nonZeros[k];
    return Acpy;
}

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_SPARSITY_PATTERN_H