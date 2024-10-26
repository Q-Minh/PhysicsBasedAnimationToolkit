#ifndef PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H
#define PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H

#include "Concepts.h"
#include "Scale.h"
#include "SubMatrix.h"
#include "Transpose.h"
#include "pbat/HostDevice.h"

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

    using Scalar   = typename LhsNestedType::Scalar;
    using SelfType = Sum<LhsNestedType, RhsNestedType>;

    static auto constexpr RowsAtCompileTime = LhsNestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = RhsNestedType::ColsAtCompileTime;

    PBAT_HOST_DEVICE Sum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::RowsAtCompileTime == RhsNestedType::RowsAtCompileTime and
                LhsNestedType::ColsAtCompileTime == RhsNestedType::ColsAtCompileTime,
            "Invalid matrix sum dimensions");
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(i, j) + B(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }

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
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Minimum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using Scalar   = typename LhsNestedType::Scalar;
    using SelfType = Minimum<LhsNestedType, RhsNestedType>;

    static auto constexpr RowsAtCompileTime = LhsNestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = RhsNestedType::ColsAtCompileTime;

    PBAT_HOST_DEVICE Minimum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::RowsAtCompileTime == RhsNestedType::RowsAtCompileTime and
                LhsNestedType::ColsAtCompileTime == RhsNestedType::ColsAtCompileTime,
            "Invalid matrix minimum dimensions");
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return min(A(i, j), B(i, j)); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }

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
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Maximum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using Scalar   = typename LhsNestedType::Scalar;
    using SelfType = Maximum<LhsNestedType, RhsNestedType>;

    static auto constexpr RowsAtCompileTime = LhsNestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = RhsNestedType::ColsAtCompileTime;

    PBAT_HOST_DEVICE Maximum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::RowsAtCompileTime == RhsNestedType::RowsAtCompileTime and
                LhsNestedType::ColsAtCompileTime == RhsNestedType::ColsAtCompileTime,
            "Invalid matrix maximum dimensions");
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return max(A(i, j), B(i, j)); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }

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
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator+(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    return Sum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator+=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        ((std::forward<TLhsMatrix>(A)(I, j) += std::forward<TRhsMatrix>(B)(I, j)), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>()), ...);
    };
    fCols(std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>());
    return A;
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator-(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    using NegatedMatrixType = Scale<RhsMatrixType>;
    NegatedMatrixType negB  = -std::forward<TRhsMatrix>(B);
    return Sum<LhsMatrixType, NegatedMatrixType>(std::forward<TLhsMatrix>(A), negB);
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto operator-=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    static_assert(
        LhsMatrixType::RowsAtCompileTime == RhsMatrixType::RowsAtCompileTime and
            LhsMatrixType::ColsAtCompileTime == RhsMatrixType::ColsAtCompileTime,
        "A and B must have same dimensions");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        ((std::forward<TLhsMatrix>(A)(I, j) -= std::forward<TRhsMatrix>(B)(I, j)), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>()), ...);
    };
    fCols(std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>());
    return A;
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto Min(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    return Minimum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto Max(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    return Maximum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_BINARY_OPERATIONS_H