#ifndef PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H
#define PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H

#include "Api.h"
#include "Concepts.h"
#include "Norm.h"
#include "Scale.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TMatrix>
class Square
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Square<NestedType>;

    static auto constexpr kRows     = NestedType::kRows;
    static auto constexpr kCols     = NestedType::kCols;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE Square(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return A(i, j) * A(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& A;
};

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Squared(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    return Square<MatrixType>(std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Normalized(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    return std::forward<TMatrix>(A) / Norm(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H