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
    VectorX>>
ToEigen(R&& rng);

template <ContiguousArithmeticRange R>
Eigen::Map<std::conditional_t<
    std::is_const_v<R>,
    Eigen::Vector<RangeValueType<R>, Eigen::Dynamic> const,
    VectorX>>
ToEigen(R&& r)
{
    namespace rng = std::ranges;
    if constexpr (std::is_const_v<R>)
    {
        return Eigen::Map<VectorX const>(rng::data(r), static_cast<Eigen::Index>(rng::size(r)));
    }
    else
    {
        return Eigen::Map<VectorX>(rng::data(r), static_cast<Eigen::Index>(rng::size(r)));
    }
}

} // namespace common
} // namespace pba

#endif // PBA_COMMON_EIGEN_H