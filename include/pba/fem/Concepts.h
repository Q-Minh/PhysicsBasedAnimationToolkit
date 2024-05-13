#ifndef PBA_CORE_FEM_CONCEPTS_H
#define PBA_CORE_FEM_CONCEPTS_H

#include "pba/aliases.h"
#include "pba/common/Concepts.h"

#include <concepts>

namespace pba {
namespace fem {

template <class T>
concept CElement = requires(T t)
{
    typename T::AffineBaseType;
    {
        T::kOrder
    } -> std::convertible_to<int>;
    {
        T::kDims
    } -> std::convertible_to<int>;
    {
        T::kNodes
    } -> std::convertible_to<int>;
    {
        T::kCoordinates
    } -> common::ContiguousIndexRange;
    {
        t.N(Vector<T::kDims>{})
    } -> std::same_as<Vector<T::kNodes>>;
    {
        t.GradN(Vector<T::kDims>{})
    } -> std::same_as<Matrix<T::kNodes, T::kDims>>;
    {
        t.Jacobian(Vector<T::kDims>{}, Matrix<T::kDims, T::kNodes>{})
    } -> std::same_as<Matrix<T::kDims, T::kDims>>;
};

template <class M>
concept CMesh = requires(M m)
{
    {
        M::ElementType
    } -> CElement;
    {
        M::kDims
    } -> std::convertible_to<int>;
    {
        m.X
    } -> std::same_as<MatrixX>;
    {
        m.E
    } -> std::same_as<IndexMatrixX>;
};

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_CONCEPTS_H