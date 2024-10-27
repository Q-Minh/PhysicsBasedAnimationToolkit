#ifndef PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H
#define PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H

#include "Concepts.h"
#include "SubMatrix.h"
#include "Transpose.h"
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
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = Square<NestedType>;

    static auto constexpr RowsAtCompileTime = NestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = NestedType::ColsAtCompileTime;
    static bool constexpr IsRowMajor        = NestedType::IsRowMajor;

    PBAT_HOST_DEVICE Square(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(i, j) * A(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }
    PBAT_HOST_DEVICE auto operator[](auto i) const { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i) const
    {
        return Slice<1, ColsAtCompileTime>(i, 0);
    }
    PBAT_HOST_DEVICE ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

  private:
    NestedType const& A;
};

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator-(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    using ScalarType = typename MatrixType::Scalar;
    return Scale<MatrixType>(ScalarType(-1.), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Squared(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return Square<MatrixType>(std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Normalized(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    static_assert(
        (MatrixType::RowsAtCompileTime == 1) or (MatrixType::ColsAtCompileTime == 1),
        "Only vectors can be normalized");
    return std::forward<TMatrix>(A) / Norm(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H