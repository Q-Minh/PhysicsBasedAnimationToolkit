#ifndef PBA_CORE_FEM_CONCEPTS_H
#define PBA_CORE_FEM_CONCEPTS_H

#include "pba/aliases.h"
#include "pba/common/Concepts.h"

#include <concepts>

namespace pba {
namespace fem {

template <class T>
concept Element = requires(T t)
{
    typename T::AffineBase;
    {
        T::Order
    } -> std::convertible_to<int>;
    {
        T::Dims
    } -> std::convertible_to<int>;
    {
        T::Nodes
    } -> std::convertible_to<int>;
    {
        T::Coordinates
    } -> common::ContiguousIndexRange;
    {
        t.N(Vector<T::Dims>{})
    } -> std::same_as<Vector<T::Nodes>>;
    {
        t.GradN(Vector<T::Dims>{})
    } -> std::same_as<Matrix<T::Nodes, T::Dims>>;
    {
        t.Jacobian(Vector<T::Dims>{}, Matrix<T::Dims, T::Nodes>{})
    } -> std::same_as<Matrix<T::Dims, T::Dims>>;
};

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_CONCEPTS_H