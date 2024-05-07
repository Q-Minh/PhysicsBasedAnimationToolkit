#ifndef PBA_COMMON_EIGEN_H
#define PBA_COMMON_EIGEN_H

#include "Concepts.h"
#include "pba/aliases.h"

#include <ranges>

namespace pba {
namespace common {

template <ContiguousArithmeticRange R>
Eigen::Map<std::conditional_t<
    std::is_const_v<R>,
    Eigen::Vector<RangeValueType<R>, Eigen::Dynamic> const,
    Eigen::Vector<RangeValueType<R>, Eigen::Dynamic>>>
ToEigen(R&& rng);

template <ContiguousArithmeticRange R>
Eigen::Map<std::conditional_t<
    std::is_const_v<R>,
    Eigen::Vector<RangeValueType<R>, Eigen::Dynamic> const,
    Eigen::Vector<RangeValueType<R>, Eigen::Dynamic>>>
ToEigen(R&& r)
{
    namespace rng = std::ranges;
    if constexpr (std::is_const_v<R>)
    {
        return Eigen::Map<Eigen::Vector<RangeValueType<R>, Eigen::Dynamic> const>(
            rng::data(r),
            static_cast<Eigen::Index>(rng::size(r)));
    }
    else
    {
        return Eigen::Map<Eigen::Vector<RangeValueType<R>, Eigen::Dynamic>>(
            rng::data(r),
            static_cast<Eigen::Index>(rng::size(r)));
    }
}

template <ContiguousMatrixRange R>
Eigen::Map<std::conditional_t<
    std::is_const_v<R>,
    Eigen::Matrix<typename RangeValueType<R>::Scalar, Eigen::Dynamic, Eigen::Dynamic> const,
    Eigen::Matrix<typename RangeValueType<R>::Scalar, Eigen::Dynamic, Eigen::Dynamic>>>
ToEigen(R&& r)
{
    namespace rng = std::ranges;
    auto Rows     = RangeValueType<R>::RowsAtCompileTime;
    auto Cols     = RangeValueType<R>::ColsAtCompileTime;
    if (RangeValueType<R>::Flags & Eigen::RowMajorBit)
        std::swap(Rows, Cols);

    if constexpr (std::is_const_v<R>)
    {
        return Eigen::Map<
            Eigen::
                Matrix<typename RangeValueType<R>::Scalar, Eigen::Dynamic, Eigen::Dynamic> const>(
            static_cast<Scalar const*>(rng::data(r)[0][0]),
            Rows,
            static_cast<Eigen::Index>(rng::size(r) * Cols));
    }
    else
    {
        return Eigen::Map<
            Eigen::Matrix<typename RangeValueType<R>::Scalar, Eigen::Dynamic, Eigen::Dynamic>>(
            static_cast<Scalar*>(rng::data(r)[0][0]),
            Rows,
            static_cast<Eigen::Index>(rng::size(r) * Cols));
    }
}

} // namespace common
} // namespace pba

#endif // PBA_COMMON_EIGEN_H