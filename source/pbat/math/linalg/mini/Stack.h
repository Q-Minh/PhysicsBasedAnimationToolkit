#ifndef PBAT_MATH_LINALG_MINI_STACK_H
#define PBAT_MATH_LINALG_MINI_STACK_H

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class HorizontalStack
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;
    using ScalarType    = typename LhsNestedType::ScalarType;
    using SelfType      = HorizontalStack<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = LhsNestedType::kCols + RhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE HorizontalStack(LhsNestedType const& lhs, RhsNestedType const& rhs)
        : mLhs(lhs), mRhs(rhs)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows,
            "lhs and rhs must have same rows");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return (j < LhsNestedType::kCols) ? mLhs(i, j) : mRhs(i, j - LhsNestedType::kCols);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& mLhs;
    RhsNestedType const& mRhs;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class VerticalStack
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;
    using ScalarType    = typename LhsNestedType::ScalarType;
    using SelfType      = VerticalStack<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows + RhsNestedType::kRows;
    static auto constexpr kCols     = LhsNestedType::kCols;
    static bool constexpr bRowMajor = LhsNestedType::bRowMajor;

    PBAT_HOST_DEVICE VerticalStack(LhsNestedType const& lhs, RhsNestedType const& rhs)
        : mLhs(lhs), mRhs(rhs)
    {
        static_assert(
            LhsNestedType::kCols == RhsNestedType::kCols,
            "lhs and rhs must have same columns");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return (i < LhsNestedType::kRows) ? mLhs(i, j) : mRhs(i - LhsNestedType::kRows, j);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& mLhs;
    RhsNestedType const& mRhs;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto HStack(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    PBAT_MINI_CHECK_CMATRIX(LhsMatrixType);
    PBAT_MINI_CHECK_CMATRIX(RhsMatrixType);
    return HorizontalStack<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto VStack(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    PBAT_MINI_CHECK_CMATRIX(LhsMatrixType);
    PBAT_MINI_CHECK_CMATRIX(RhsMatrixType);
    return VerticalStack<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_STACK_H