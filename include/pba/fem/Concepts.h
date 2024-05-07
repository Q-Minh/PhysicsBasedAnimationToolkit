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
    requires std::integral<decltype(std::declval<T>().Order)>;
    requires std::integral<decltype(T::Dims)>;
    requires std::integral<decltype(T::Nodes)>;
    requires std::integral<decltype(T::Vertices)>;
    requires common::ContiguousIndexRange<decltype(std::declval<T>().Coordinates)>;
    {
        t.N(Vector<T::Dims>{})
    } -> std::same_as<Vector<T::Nodes>>;
    {
        t.GradN(Vector<T::Dims>{})
    } -> std::same_as<Matrix<T::Nodes, T::Dims>>;
    // TODO: Also add constraint for an AffineMap(...) member function
};

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_CONCEPTS_H