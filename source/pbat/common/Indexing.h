/**
 * @file Indexing.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_INDEXING_H
#define PBAT_COMMON_INDEXING_H

#include "Concepts.h"
#include "Eigen.h"
#include "pbat/Aliases.h"

#include <concepts>
#include <numeric>
#include <random>
#include <ranges>

namespace pbat {
namespace common {

/**
 * @brief Cumulative sum of a range of integers
 *
 * @tparam R Integer range type
 * @tparam TIndex Type of the integers
 * @param sizes Range of integers
 * @return Eigen::Vector<TIndex, Eigen::Dynamic> Cumulative sum of the range
 */
template <CIndexRange R, std::integral TIndex = std::ranges::range_value_t<R>>
Eigen::Vector<TIndex, Eigen::Dynamic> CumSum(R&& sizes)
{
    namespace rng         = std::ranges;
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    IndexVectorType cs(rng::size(sizes) + 1);
    auto bi    = cs.data();
    *bi++      = Index{0};
    auto begin = rng::begin(sizes);
    auto end   = rng::end(sizes);
    std::partial_sum(begin, end, bi);
    return cs;
}

/**
 * @brief Counts the number of occurrences of each integer in a contiguous range
 *
 * @tparam TIndex Integer type of counts
 * @param begin Range begin
 * @param end Range end (exclusive)
 * @param ncounts Upper bound on values in range
 * @return Eigen::Vector<TIndex, Eigen::Dynamic> Counts of each integer in the range
 */
template <std::integral TIndex>
Eigen::Vector<TIndex, Eigen::Dynamic> Counts(auto begin, auto end, TIndex ncounts)
{
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    IndexVectorType counts(ncounts);
    counts.setZero();
    for (auto it = begin; it != end; ++it)
        ++counts(*it);
    return counts;
}

/**
 * @brief Randomly shuffle a range of integers
 *
 * @tparam TIndex Integer type of the range
 * @param begin Start of the range (inclusive)
 * @param end End of the range (exclusive)
 * @return Eigen::Vector<TIndex, Eigen::Dynamic> Shuffled range of integers
 */
template <std::integral TIndex>
Eigen::Vector<TIndex, Eigen::Dynamic> Shuffle(TIndex begin, TIndex end)
{
    auto iota = std::views::iota(begin, end);
    Eigen::Vector<TIndex, Eigen::Dynamic> inds(end - begin);
    std::copy(std::ranges::begin(iota), std::ranges::end(iota), inds.begin());
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::ranges::shuffle(inds, gen);
    return inds;
}

/**
 * @brief Filters a range of integers based on a predicate function
 *
 * @tparam TIndexB Type of the beginning index
 * @tparam TIndexE Type of the ending index
 * @tparam Func Predicate function type (TIndex -> bool)
 * @tparam TIndex Common type of the indices
 * @param begin Start of the range (inclusive)
 * @param end End of the range (exclusive)
 * @param f Predicate function to filter the range
 * @return Eigen::Vector<TIndex, Eigen::Dynamic> Filtered range of integers
 */
template <
    std::integral TIndexB,
    std::integral TIndexE,
    class Func,
    class TIndex = std::common_type_t<TIndexB, TIndexE>>
Eigen::Vector<TIndex, Eigen::Dynamic> Filter(TIndexB begin, TIndexE end, Func&& f)
{
    auto filteredView = std::views::iota(static_cast<TIndex>(begin), static_cast<TIndex>(end)) |
                        std::views::filter(f);
    std::vector<TIndex> filtered{};
    filtered.reserve(static_cast<std::size_t>(end - begin));
    std::ranges::copy(filteredView, std::back_inserter(filtered));
    return ToEigen(filtered);
}

/**
 * @brief Repeats elements of a vector according to a repetition vector
 *
 * Similar to [numpy.repeat](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html)
 *
 * @tparam TDerivedX Eigen dense expression of the input vector
 * @tparam TDerivedR Eigen dense expression of the repetition vector
 * @tparam TScalar Scalar type of the input vector
 * @tparam TIndex Integer type of the repetition vector
 * @param x Values to repeat
 * @param r Repetition vector
 * @return Eigen::Vector<TScalar, Eigen::Dynamic> Vector with repeated elements
 */
template <
    class TDerivedX,
    class TDerivedR,
    class TScalar        = typename TDerivedX::Scalar,
    std::integral TIndex = typename TDerivedR::Scalar>
Eigen::Vector<TScalar, Eigen::Dynamic>
Repeat(Eigen::DenseBase<TDerivedX> const& x, Eigen::DenseBase<TDerivedR> const& r)
{
    using VectorType = Eigen::Vector<TScalar, Eigen::Dynamic>;
    VectorType y(r.sum());
    for (Index i = 0, k = 0; i < r.size(); ++i)
    {
        auto ri                  = r(i);
        y.segment(k, ri).array() = x(i);
        k += ri;
    }
    return y;
}

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_INDEXING_H
