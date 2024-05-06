#ifndef PBA_CORE_MATH_CONCEPTS_H
#define PBA_CORE_MATH_CONCEPTS_H

#include "pba/aliases.h"
#include "pba/common/Concepts.h"

#include <concepts>

namespace pba {
namespace math {

template <class T>
concept PolynomialBasis = requires(T t)
{
    requires std::integral<decltype(T::Dims)>;
    requires std::integral<decltype(T::Order)>;
    requires std::integral<decltype(T::Size)>;
    {
        t.eval(Vector<T::Dims>{})
    } -> std::same_as<Vector<T::Size>>;
    {
        t.derivatives(Vector<T::Dims>{})
    } -> std::same_as<Matrix<T::Dims, T::Size>>;
    {
        t.antiderivatives(Vector<T::Dims>{})
    } -> std::same_as<Matrix<T::Size, T::Dims>>;
};

template <class Q>
concept QuadratureRule = requires(Q q)
{
    requires std::integral<decltype(Q::Dims)>;
    requires common::ContiguousArithmeticRange<decltype(q.points)>;
    requires common::ContiguousArithmeticRange<decltype(q.weights)>;
    { q.points.size() } -> std::integral;
    { q.weights.size() } -> std::integral;
};

template <class Q>
concept FixedPointQuadratureRule = requires(Q q)
{
    requires QuadratureRule<Q>;
    requires std::integral<decltype(Q::npoints)>;
    { q.points.size() / q.weights.size() == Q::Dims };
    { q.weights.size() == Q::npoints };
};

template <class Q>
concept PolynomialQuadratureRule = requires(Q q)
{
    requires QuadratureRule<Q>;
    requires std::integral<decltype(Q::Order)>;
};

} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_CONCEPTS_H