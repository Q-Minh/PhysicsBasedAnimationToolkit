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

#include <array>
#include <concepts>
#include <numeric>
#include <random>
#include <ranges>
#include <utility>

namespace pbat {
namespace common {

/**
 * @brief Cumulative sum of a range of integers
 *
 * @tparam R Integer range type
 * @tparam TIndex Type of the integers
 * @param sizes Range of integers
 * @return Cumulative sum of the range
 */
template <CIndexRange R, std::integral TIndex = std::ranges::range_value_t<R>>
auto CumSum(R&& sizes) -> Eigen::Vector<TIndex, Eigen::Dynamic>
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
 * @brief Compute an exclusive prefix sum from a compile-time parameter pack of values.
 *
 * Given N values `s0, s1, ..., s_{N-1}`, returns a `std::array<T, N+1>` containing
 * `{0, s0, s0+s1, ..., s0+s1+...+s_{N-1}}`.
 *
 * @tparam T      Arithmetic result type (deduced from the common type of the arguments).
 * @tparam Sizes  Pack of arithmetic types, all convertible to T.
 * @param sizes   The N values to prefix-sum.
 * @return `std::array<T, sizeof...(Sizes) + 1>` with the exclusive prefix sums.
 *
 * Example:
 * @code
 *   auto p = ExclusivePrefixSum(3, 5, 2); // -> std::array<int,4>{0, 3, 8, 10}
 * @endcode
 */
template <class... Sizes>
    requires(sizeof...(Sizes) > 0 and (std::is_arithmetic_v<std::decay_t<Sizes>> and ...))
constexpr auto ExclusivePrefixSum(Sizes... sizes)
{
    using T                 = std::common_type_t<std::decay_t<Sizes>...>;
    std::size_t constexpr N = sizeof...(Sizes);
    std::array<T, N + 1> prefix{};
    T const values[N] = {static_cast<T>(sizes)...};
    prefix[0]         = T{0};
    for (std::size_t i = 0; i < N; ++i)
        prefix[i + 1] = prefix[i] + values[i];
    return prefix;
}

/**
 * @brief Overload that writes the exclusive prefix sum into a pre-existing array (or
 * array-like container with `operator[]`).
 *
 * @tparam TArray  Container type supporting `operator[]` (e.g. `std::array<T, N+1>`).
 * @tparam Sizes   Pack of arithmetic types.
 * @param[out] out The output array; must have at least `sizeof...(Sizes) + 1` elements.
 * @param sizes    The N values to prefix-sum.
 */
template <class TArray, class... Sizes>
    requires(sizeof...(Sizes) > 0 and (std::is_arithmetic_v<std::decay_t<Sizes>> and ...))
constexpr void ExclusivePrefixSum(TArray& out, Sizes... sizes)
{
    using T                 = std::common_type_t<std::decay_t<Sizes>...>;
    std::size_t constexpr N = sizeof...(Sizes);
    T const values[N]       = {static_cast<T>(sizes)...};
    out[0]                  = T{0};
    for (std::size_t i = 0; i < N; ++i)
        out[i + 1] = out[i] + values[i];
}

/**
 * @brief Counts the number of occurrences of each integer in a contiguous range
 *
 * @tparam TIndex Integer type of counts
 * @param begin Range begin
 * @param end Range end (exclusive)
 * @param ncounts Upper bound on values in range
 * @return Counts of each integer in the range
 */
template <std::integral TIndex>
auto Counts(auto begin, auto end, TIndex ncounts) -> Eigen::Vector<TIndex, Eigen::Dynamic>
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
 * @return Shuffled range of integers
 */
template <std::integral TIndex>
auto Shuffle(TIndex begin, TIndex end) -> Eigen::Vector<TIndex, Eigen::Dynamic>
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
 * @return Filtered range of integers
 */
template <
    std::integral TIndexB,
    std::integral TIndexE,
    class Func,
    class TIndex = std::common_type_t<TIndexB, TIndexE>>
auto Filter(TIndexB begin, TIndexE end, Func&& f) -> Eigen::Vector<TIndex, Eigen::Dynamic>
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
 * @return Vector with repeated elements
 */
template <
    class TDerivedX,
    class TDerivedR,
    class TScalar        = typename TDerivedX::Scalar,
    std::integral TIndex = typename TDerivedR::Scalar>
auto Repeat(Eigen::DenseBase<TDerivedX> const& x, Eigen::DenseBase<TDerivedR> const& r)
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
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
