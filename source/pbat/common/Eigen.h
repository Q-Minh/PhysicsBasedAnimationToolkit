/**
 * @file Eigen.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Eigen adaptors for ranges
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_EIGEN_H
#define PBAT_COMMON_EIGEN_H

#include "Concepts.h"

#include <pbat/Aliases.h>
#include <ranges>

namespace pbat {
namespace common {

/**
 * @brief Map a range of scalars to an eigen vector of such scalars
 * @tparam R Range type
 * @param r Range
 * @return Eigen vector adaptor
 */
template <CContiguousArithmeticRange R>
auto ToEigen(R&& r)
    -> Eigen::Map<Eigen::Vector<std::ranges::range_value_t<R>, Eigen::Dynamic> const>
{
    namespace rng = std::ranges;
    return Eigen::Map<Eigen::Vector<std::ranges::range_value_t<R>, Eigen::Dynamic> const>(
        rng::data(r),
        static_cast<Eigen::Index>(rng::size(r)));
}

/**
 * @brief Map a range of scalar matrices to an eigen vector of such scalars
 * @tparam R Range type
 * @param r Range
 * @return Eigen matrix adaptor
 */
template <CContiguousArithmeticMatrixRange R>
auto ToEigen(R&& r) -> Eigen::Map<Eigen::Matrix<
    typename std::ranges::range_value_t<R>::Scalar,
    Eigen::Dynamic,
    Eigen::Dynamic> const>
{
    namespace rng   = std::ranges;
    using ValueType = rng::range_value_t<R>;
    auto rows       = ValueType::RowsAtCompileTime;
    auto cols       = ValueType::ColsAtCompileTime;
    if constexpr (ValueType::Flags & Eigen::RowMajorBit)
        std::swap(rows, cols);

    return Eigen::Map<
        Eigen::Matrix<typename ValueType::Scalar, Eigen::Dynamic, Eigen::Dynamic> const>(
        static_cast<Scalar const*>(std::addressof(rng::data(r)[0][0])),
        rows,
        static_cast<Eigen::Index>(rng::size(r) * cols));
}

namespace detail {

template <std::ranges::random_access_range R>
struct Slice
{
    Slice(R&& r) : r(std::forward<R>(r)) {}

    Index size() const { return static_cast<Index>(std::ranges::size(r)); }
    Index operator[](Index i) const { return static_cast<Index>(r[i]); }

    std::remove_cvref_t<R> r;
};

} // namespace detail

/**
 * @brief Slice view over a range for Eigen advanced indexing
 *
 * @tparam R Range type
 * @param r Range
 * @return Type with size(), operator[] and operator() for Eigen advanced indexing
 */
template <std::ranges::random_access_range R>
auto Slice(R&& r)
{
    return detail::Slice<R>(std::forward<R>(r));
}

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_EIGEN_H
