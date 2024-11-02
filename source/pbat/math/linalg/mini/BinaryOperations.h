#ifndef PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H
#define PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H

#include "Api.h"
#include "Concepts.h"
#include "Scale.h"
#include "pbat/HostDevice.h"

#include <cmath>
#include <functional>
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

template <CMatrix TLhsMatrix>
class SumScalar
{
  public:
    using LhsNestedType = TLhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = SumScalar<LhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = LhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE SumScalar(LhsNestedType const& A, ScalarType k) : mA(A), mK(k) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return mA(i, j) + mK; }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& mA;
    ScalarType mK;
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

template <CMatrix TLhsMatrix>
class SubtractionScalar
{
  public:
    using LhsNestedType = TLhsMatrix;
    using ScalarType    = typename LhsNestedType::ScalarType;
    using SelfType      = SubtractionScalar<LhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = LhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE SubtractionScalar(LhsNestedType const& A, ScalarType k) : mA(A), mK(k) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return mA(i, j) - mK; }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& mA;
    ScalarType mK;
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

template <CMatrix TMatrix, class Compare>
class MatrixScalarPredicate
{
  public:
    using CompareType = Compare;
    using NestedType  = TMatrix;
    using ScalarType  = typename NestedType::ScalarType;
    using SelfType    = MatrixScalarPredicate<NestedType, CompareType>;

    static auto constexpr kRows     = NestedType::kRows;
    static auto constexpr kCols     = NestedType::kCols;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE MatrixScalarPredicate(NestedType const& A, ScalarType k, CompareType comp)
        : mA(A), mK(k), mComparator(comp)
    {
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return mComparator(mA(i, j), mK);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& mA;
    ScalarType mK;
    CompareType mComparator;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix, class Compare>
class MatrixMatrixPredicate
{
  public:
    using CompareType   = Compare;
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;
    using ScalarType    = typename LhsNestedType::ScalarType;
    using SelfType      = MatrixMatrixPredicate<LhsNestedType, RhsNestedType, CompareType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = LhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE
    MatrixMatrixPredicate(LhsNestedType const& A, RhsNestedType const& B, CompareType comp)
        : mA(A), mB(B), mComparator(comp)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "A and B must have same dimensions");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return mComparator(mA(i, j), mB(i, j));
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& mA;
    RhsNestedType const& mB;
    CompareType mComparator;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator+(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    if constexpr (std::is_arithmetic_v<RhsMatrixType>)
    {
        return SumScalar<LhsMatrixType>(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
    }
    else
    {
        return Sum<LhsMatrixType, RhsMatrixType>(
            std::forward<TLhsMatrix>(A),
            std::forward<TRhsMatrix>(B));
    }
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator+=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    if constexpr (std::is_arithmetic_v<RhsMatrixType>)
    {
        AddAssignScalar(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
    }
    else
    {
        AddAssign(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
    }
    return A;
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator-(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    if constexpr (std::is_arithmetic_v<RhsMatrixType>)
    {
        return SubtractionScalar<LhsMatrixType>(
            std::forward<TLhsMatrix>(A),
            std::forward<TRhsMatrix>(B));
    }
    else
    {
        return Subtraction<LhsMatrixType, RhsMatrixType>(
            std::forward<TLhsMatrix>(A),
            std::forward<TRhsMatrix>(B));
    }
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator-=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    if constexpr (std::is_arithmetic_v<RhsMatrixType>)
    {
        SubtractAssignScalar(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
    }
    else
    {
        SubtractAssign(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
    }
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

#define PBAT_MINI_DEFINE_MATRIX_SCALAR_PREDICATE(Operator, Comparator)               \
    template <CMatrix TMatrix>                                                       \
    PBAT_HOST_DEVICE auto Operator(TMatrix const& A, typename TMatrix::ScalarType k) \
    {                                                                                \
        using ScalarType  = typename TMatrix::ScalarType;                            \
        using CompareType = Comparator<ScalarType>;                                  \
        return MatrixScalarPredicate<TMatrix, CompareType>(A, k, CompareType{});     \
    }

#define PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(Operator, Comparator)                          \
    template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>                                           \
    PBAT_HOST_DEVICE auto Operator(TLhsMatrix const& A, TRhsMatrix const& B)                    \
    {                                                                                           \
        using ScalarType  = typename TLhsMatrix::ScalarType;                                    \
        using CompareType = Comparator<ScalarType>;                                             \
        return MatrixMatrixPredicate<TLhsMatrix, TRhsMatrix, CompareType>(A, B, CompareType{}); \
    }

PBAT_MINI_DEFINE_MATRIX_SCALAR_PREDICATE(operator<, std::less)
PBAT_MINI_DEFINE_MATRIX_SCALAR_PREDICATE(operator>, std::greater)
PBAT_MINI_DEFINE_MATRIX_SCALAR_PREDICATE(operator==, std::equal_to)
PBAT_MINI_DEFINE_MATRIX_SCALAR_PREDICATE(operator!=, std::not_equal_to)
PBAT_MINI_DEFINE_MATRIX_SCALAR_PREDICATE(operator<=, std::less_equal)
PBAT_MINI_DEFINE_MATRIX_SCALAR_PREDICATE(operator>=, std::greater_equal)

PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator<, std::less)
PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator>, std::greater)
PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator==, std::equal_to)
PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator!=, std::not_equal_to)
PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator<=, std::less_equal)
PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator>=, std::greater_equal)

PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator&&, std::logical_and)
PBAT_MINI_DEFINE_MATRIX_MATRIX_PREDICATE(operator||, std::logical_or)

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H