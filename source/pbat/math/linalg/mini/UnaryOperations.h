#ifndef PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H
#define PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H

#include "Api.h"
#include "Concepts.h"
#include "Norm.h"
#include "Scale.h"
#include "pbat/HostDevice.h"

#include <algorithm>
#include <cmath>
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

template <CMatrix TLhsMatrix>
class Reciprocal
{
  public:
    using LhsNestedType = TLhsMatrix;
    using ScalarType    = typename LhsNestedType::ScalarType;
    using SelfType      = Reciprocal<LhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = LhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE Reciprocal(LhsNestedType const& A) : mA(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return ScalarType(1) / mA(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& mA;
};

template <CMatrix TMatrix>
class Absolute
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Absolute<NestedType>;

    static auto constexpr kRows     = NestedType::kRows;
    static auto constexpr kCols     = NestedType::kCols;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE Absolute(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        using namespace std;
        return abs(A(i, j));
    }

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
PBAT_HOST_DEVICE auto Abs(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    return Absolute<MatrixType>(std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Normalized(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    return std::forward<TMatrix>(A) / Norm(std::forward<TMatrix>(A));
}

template <CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE auto Min(TMatrix const& A)
{
    using IntegerType = std::remove_const_t<decltype(TMatrix::kRows)>;
    auto minimum      = [&]<IntegerType... K>(std::integer_sequence<IntegerType, K...>) {
        return std::min({A(K)...});
    };
    return minimum(std::make_integer_sequence<IntegerType, TMatrix::kRows * TMatrix::kCols>());
}

template <CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE auto Max(TMatrix const& A)
{
    using IntegerType = std::remove_const_t<decltype(TMatrix::kRows)>;
    auto maximum      = [&]<IntegerType... K>(std::integer_sequence<IntegerType, K...>) {
        return std::max({A(K)...});
    };
    return maximum(std::make_integer_sequence<IntegerType, TMatrix::kRows * TMatrix::kCols>());
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator/(typename std::remove_cvref_t<TMatrix>::ScalarType k, TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    return k * Reciprocal<MatrixType>(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_UNARY_OPERATIONS_H