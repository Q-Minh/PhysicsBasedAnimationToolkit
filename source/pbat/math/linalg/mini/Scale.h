#ifndef PBAT_MATH_LINALG_MINI_SCALE_H
#define PBAT_MATH_LINALG_MINI_SCALE_H

#include "Api.h"
#include "Assign.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix>
class Scale
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Scale<NestedType>;

    static auto constexpr kRows     = NestedType::kRows;
    static auto constexpr kCols     = NestedType::kCols;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE Scale(ScalarType _k, NestedType const& _A) : k(_k), A(_A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return k * A(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    ScalarType k;
    NestedType const& A;
};

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator*(typename std::remove_cvref_t<TMatrix>::ScalarType k, TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    return Scale<MatrixType>(k, std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator*(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    return k * std::forward<TMatrix>(A);
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator*=(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    MultiplyAssign(std::forward<TMatrix>(A), k);
    return A;
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator/(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    using ScalarType = typename MatrixType::ScalarType;
    return Scale<MatrixType>(ScalarType(1. / k), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator/=(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    DivideAssign(std::forward<TMatrix>(A), k);
    return A;
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator-(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    using ScalarType = typename MatrixType::ScalarType;
    return Scale<MatrixType>(ScalarType(-1.), std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_SCALE_H