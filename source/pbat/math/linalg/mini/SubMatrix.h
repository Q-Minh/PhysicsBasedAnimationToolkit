#ifndef PBAT_MATH_LINALG_MINI_SUBMATRIX_H
#define PBAT_MATH_LINALG_MINI_SUBMATRIX_H

#include "Concepts.h"
#include "Transpose.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TMatrix, int M, int N>
class ConstSubMatrix
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = ConstSubMatrix<NestedType, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;

    PBAT_HOST_DEVICE ConstSubMatrix(NestedType const& A, auto ib = 0, auto jb = 0)
        : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::RowsAtCompileTime >= M and NestedType::ColsAtCompileTime >= N and M > 0 and
                N > 0,
            "Invalid submatrix dimensions");
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(ib + i, jb + j); }

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
    PBAT_HOST_DEVICE ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
    }

  private:
    NestedType const& A;
    int ib, jb;
};

template <CMatrix TMatrix, int M, int N>
class SubMatrix
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = SubMatrix<NestedType, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;

    PBAT_HOST_DEVICE SubMatrix(NestedType& A, auto ib = 0, auto jb = 0) : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::RowsAtCompileTime >= M and NestedType::ColsAtCompileTime >= N and M > 0 and
                N > 0,
            "Invalid submatrix dimensions");
    }

    template <class /*CMatrix*/ TOtherMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TOtherMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TOtherMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        static_assert(
            OtherMatrixType::RowsAtCompileTime == RowsAtCompileTime and
                OtherMatrixType::ColsAtCompileTime == ColsAtCompileTime,
            "Invalid submatrix dimensions");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TOtherMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<RowsAtCompileTime>()), ...);
        };
        fCols(std::make_index_sequence<ColsAtCompileTime>());
        return *this;
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(ib + i, jb + j); }
    PBAT_HOST_DEVICE auto& operator()(auto i, auto j) { return A(ib + i, jb + j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }
    PBAT_HOST_DEVICE auto& operator()(auto i)
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }

    template <auto S, auto T>
    PBAT_HOST_DEVICE SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE SubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j)
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE SubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i)
    {
        return Slice<1, ColsAtCompileTime>(i, 0);
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

    PBAT_HOST_DEVICE TransposeView<SelfType> Transpose() { return TransposeView<SelfType>(*this); }
    PBAT_HOST_DEVICE ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
    }

  private:
    NestedType& A;
    int ib, jb;
};

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_SUBMATRIX_H