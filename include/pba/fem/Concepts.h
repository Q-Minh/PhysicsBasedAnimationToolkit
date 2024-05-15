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
        T::bHasConstantJacobian
    } -> std::convertible_to<bool>;
    // Should be valid for argument > 1 as well, but we don't check that.
    typename T::template QuadratureType<1>;
    {
        T::kOrder
    } -> std::convertible_to<int>;
    {
        T::kDims
    } -> std::convertible_to<int>;
    {
        T::kNodes
    } -> std::convertible_to<int>;
    requires common::CContiguousIndexRange<decltype(T::Coordinates)>;
    requires common::CContiguousIndexRange<decltype(T::Vertices)>;
    {
        t.N(Vector<T::kDims>{})
    } -> std::convertible_to<Vector<T::kNodes>>;
    {
        t.GradN(Vector<T::kDims>{})
    } -> std::convertible_to<Matrix<T::kNodes, T::kDims>>;
};

template <class M>
concept CMesh = requires(M m)
{
    requires CElement<typename M::ElementType>;
    {
        M::kDims
    } -> std::convertible_to<int>;
    {
        m.X
    } -> std::convertible_to<MatrixX>;
    {
        m.E
    } -> std::convertible_to<IndexMatrixX>;
};

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_CONCEPTS_H