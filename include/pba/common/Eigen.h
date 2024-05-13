#ifndef PBA_COMMON_EIGEN_H
#define PBA_COMMON_EIGEN_H

#include "Concepts.h"
#include "pba/aliases.h"

#include <ranges>

namespace pba {
namespace common {

template <ContiguousArithmeticRange R>
Eigen::Map<Eigen::Vector<std::ranges::range_value_t<R>, Eigen::Dynamic> const> ToEigen(R&& r)
{
    namespace rng = std::ranges;
    return Eigen::Map<Eigen::Vector<std::ranges::range_value_t<R>, Eigen::Dynamic> const>(
        rng::data(r),
        static_cast<Eigen::Index>(rng::size(r)));
}

template <ContiguousArithmeticMatrixRange R>
Eigen::Map<Eigen::Matrix<
    typename std::ranges::range_value_t<R>::Scalar,
    Eigen::Dynamic,
    Eigen::Dynamic> const>
ToEigen(R&& r)
{
    namespace rng   = std::ranges;
    using ValueType = std::ranges::range_value_t<R>;
    auto Rows       = ValueType::RowsAtCompileTime;
    auto Cols       = ValueType::ColsAtCompileTime;
    if (ValueType::Flags & Eigen::RowMajorBit)
        std::swap(Rows, Cols);

    return Eigen::Map<
        Eigen::Matrix<typename ValueType::Scalar, Eigen::Dynamic, Eigen::Dynamic> const>(
        static_cast<Scalar const*>(std::addressof(rng::data(r)[0][0])),
        Rows,
        static_cast<Eigen::Index>(rng::size(r) * Cols));
}

} // namespace common
} // namespace pba

#endif // PBA_COMMON_EIGEN_H