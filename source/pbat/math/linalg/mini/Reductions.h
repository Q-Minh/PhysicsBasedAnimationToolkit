#ifndef PBAT_MATH_LINALG_MINI_REDUCTIONS_H
#define PBAT_MATH_LINALG_MINI_REDUCTIONS_H

#include "Api.h"
#include "Concepts.h"
#include "Product.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TMatrix>
class ConstDiagonal
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = ConstDiagonal<NestedType>;

    static auto constexpr kRows     = NestedType::kRows;
    static auto constexpr kCols     = 1;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE ConstDiagonal(NestedType const& A) : mA(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, [[maybe_unused]] auto j) const
    {
        return mA(i, i);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return mA(i, i); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& mA;
};

template <CMatrix TMatrix>
class Diagonal
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Diagonal<NestedType>;

    static auto constexpr kRows     = NestedType::kRows;
    static auto constexpr kCols     = 1;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE Diagonal(NestedType& A) : mA(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, [[maybe_unused]] auto j) const
    {
        return mA(i, i);
    }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i, [[maybe_unused]] auto j) { return mA(i, i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return mA(i, i); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i) { return mA(i, i); }
    PBAT_HOST_DEVICE ScalarType& operator[](auto i) { return (*this)(i); }

    PBAT_HOST_DEVICE void SetConstant(auto k) { AssignScalar(*this, k); }

    PBAT_MINI_READ_WRITE_API(SelfType)

  private:
    NestedType& mA;
};

template <CMatrix TMatrix>
PBAT_HOST_DEVICE auto Diag(TMatrix const& A)
{
    return ConstDiagonal<TMatrix>(A);
}

template <CMatrix TMatrix>
PBAT_HOST_DEVICE auto Diag(TMatrix& A)
{
    return Diagonal<TMatrix>(A);
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Trace(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(
        MatrixType::kRows == MatrixType::kCols,
        "Cannot compute trace of non-square matrix");
    auto sum = [&]<auto... I>(std::index_sequence<I...>) {
        return (std::forward<TMatrix>(A)(I, I) + ...);
    };
    return sum(std::make_index_sequence<MatrixType::kRows>{});
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto Dot(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return Trace(std::forward<TLhsMatrix>(A).Transpose() * std::forward<TRhsMatrix>(B));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_REDUCTIONS_H