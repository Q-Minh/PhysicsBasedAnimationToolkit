#ifndef PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H
#define PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H

#include "Api.h"
#include "Concepts.h"
#include "Scale.h"
#include "pbat/HostDevice.h"

#include <cmath>
#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Sum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Sum<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = RhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE Sum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix sum dimensions");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return A(i, j) + B(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Subtraction
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Subtraction<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = RhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE Subtraction(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix sum dimensions");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return A(i, j) - B(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Minimum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Minimum<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = RhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE Minimum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix minimum dimensions");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        using namespace std;
        return min(A(i, j), B(i, j));
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Maximum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Maximum<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = RhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE Maximum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix maximum dimensions");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        using namespace std;
        return max(A(i, j), B(i, j));
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator+(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return Sum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator+=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    AddAssign(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
    return A;
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator-(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return Subtraction<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator-=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    SubtractAssign(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
    return A;
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto Min(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return Minimum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto Max(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return Maximum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H