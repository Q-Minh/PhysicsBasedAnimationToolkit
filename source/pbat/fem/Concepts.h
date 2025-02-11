/**
 * @file Concepts.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_CONCEPTS_H
#define PBAT_FEM_CONCEPTS_H

#include <concepts>
#include <pbat/Aliases.h>
#include <pbat/common/Concepts.h>

namespace pbat {
namespace fem {

/**
 * @brief Reference finite element
 *
 * Example type `TElement` satisfying concept CElement
 * ```cpp
 * template <int Order>
 * struct TElement
 * {
 *     using AffineBaseType = TElement<1>;
 *
 *     static int constexpr kOrder;
 *     static int constexpr kDims;
 *     static int constexpr kNodes;
 *     static std::array<int, kNodes * kDims> constexpr Coordinates;
 *     static std::array<int, AffineBaseType::kNodes> constexpr Vertices;
 *
 *     static bool constexpr bHasConstantJacobian;
 *
 *     template <int PolynomialOrder>
 *     using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;
 *
 *     template <class TDerived, class TScalar = typename TDerived::Scalar>
 *     static Eigen::Vector<TScalar, kNodes>
 *     N(Eigen::DenseBase<TDerived> const& X);
 *
 *     static Matrix<kNodes, kDims> GradN(Vector<kDims> const& X);
 * };
 * ```
 *
 * @note
 * - Divide `Coordinates` by `kOrder` to obtain actual coordinates in the reference element
 * - `Vertices` are indices into nodes `[0,kNodes-1]` revealing vertices of the element
 * - `bHasConstantJacobian` is `true` if the Jacobian is constant over the element
 * - `QuadratureType` proposes a quadrature rule suitable for integrating over the element
 * - `N` computes the nodal shape functions evaluated at the given reference point `X`
 * - `GradN` computes the gradient of the nodal shape functions evaluated at the given reference
 * point `X`
 *
 * @tparam T Element type
 */
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

/**
 * @brief Finite element mesh
 *
 * @tparam M Mesh type
 */
template <class M>
concept CMesh = requires(M m)
{
    requires CElement<typename M::ElementType>;
    {
        M::kDims
    } -> std::convertible_to<int>;
    {
        M::kOrder
    } -> std::convertible_to<int>;
    {
        m.X
    } -> std::convertible_to<MatrixX>;
    {
        m.E
    } -> std::convertible_to<IndexMatrixX>;
};

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_CONCEPTS_H