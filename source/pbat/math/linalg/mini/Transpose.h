#ifndef PBAT_MATH_LINALG_MINI_TRANSPOSE_H
#define PBAT_MATH_LINALG_MINI_TRANSPOSE_H

#include "Assign.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TMatrix>
class ConstTransposeView;

template <CMatrix TMatrix>
class TransposeView;

// NOTE:
// There is a cyclic dependency between transpose and submatrix views. This is because a submatrix
// should have a transpose, but you should also be able to get a submatrix from a transposed matrix.
// To break the dependency, we create a type TransposeSubMatrix (and its const version), which has
// the exact same implementation as SubMatrix (and its const version), specifically for
// (Const)TransposeView.

template <CMatrix TMatrix, int M, int N>
class ConstTransposeSubMatrix
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = ConstTransposeSubMatrix<NestedType, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;
    static bool constexpr IsRowMajor        = NestedType::IsRowMajor;

    PBAT_HOST_DEVICE ConstTransposeSubMatrix(NestedType const& A, auto ib = 0, auto jb = 0)
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
    PBAT_HOST_DEVICE auto operator[](auto i) const { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i) const
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
class TransposeSubMatrix
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = TransposeSubMatrix<NestedType, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;
    static bool constexpr IsRowMajor        = NestedType::IsRowMajor;

    PBAT_HOST_DEVICE TransposeSubMatrix(NestedType& A, auto ib = 0, auto jb = 0)
        : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::RowsAtCompileTime >= M and NestedType::ColsAtCompileTime >= N and M > 0 and
                N > 0,
            "Invalid submatrix dimensions");
    }

    template <class /*CMatrix*/ TOtherMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TOtherMatrix&& B)
    {
        Assign(*this, std::forward<TOtherMatrix>(B));
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
    PBAT_HOST_DEVICE auto operator[](auto i) const { return (*this)(i); }
    PBAT_HOST_DEVICE auto& operator[](auto i) { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE TransposeSubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return TransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE TransposeSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j)
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE TransposeSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i)
    {
        return Slice<1, ColsAtCompileTime>(i, 0);
    }

    template <auto S, auto T>
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i) const
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

template <CMatrix TMatrix>
class TransposeView
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = TransposeView<NestedType>;

    static auto constexpr RowsAtCompileTime = NestedType::ColsAtCompileTime;
    static auto constexpr ColsAtCompileTime = NestedType::RowsAtCompileTime;
    static bool constexpr IsRowMajor        = not NestedType::IsRowMajor;

    PBAT_HOST_DEVICE TransposeView(NestedType& A) : A(A) {}

    template <class TOtherMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TOtherMatrix&& B)
    {
        Assign(*this, std::forward<TOtherMatrix>(B));
        return *this;
    }

    PBAT_HOST_DEVICE void SetConstant(Scalar k)
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = k), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<RowsAtCompileTime>()), ...);
        };
        fCols(std::make_index_sequence<ColsAtCompileTime>());
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return A.Cols(); }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return A.Rows(); }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(j, i); }
    PBAT_HOST_DEVICE auto& operator()(auto i, auto j) { return A(j, i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }
    PBAT_HOST_DEVICE auto& operator()(auto i)
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }
    PBAT_HOST_DEVICE auto operator[](auto i) const { return (*this)(i); }
    PBAT_HOST_DEVICE auto& operator[](auto i) { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE TransposeSubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return TransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE TransposeSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j)
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE TransposeSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i)
    {
        return Slice<1, ColsAtCompileTime>(i, 0);
    }

    template <auto S, auto T>
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i) const
    {
        return Slice<1, ColsAtCompileTime>(i, 0);
    }

    PBAT_HOST_DEVICE NestedType const& Transpose() const { return A; }
    PBAT_HOST_DEVICE NestedType& Transpose() { return A; }

  private:
    NestedType& A;
};

template <CMatrix TMatrix>
class ConstTransposeView
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = ConstTransposeView<NestedType>;

    static auto constexpr RowsAtCompileTime = NestedType::ColsAtCompileTime;
    static auto constexpr ColsAtCompileTime = NestedType::RowsAtCompileTime;
    static bool constexpr IsRowMajor        = not NestedType::IsRowMajor;

    PBAT_HOST_DEVICE ConstTransposeView(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return A.Cols(); }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return A.Rows(); }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(j, i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }

    template <auto S, auto T>
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE ConstTransposeSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i) const
    {
        return Slice<1, ColsAtCompileTime>(i, 0);
    }

    PBAT_HOST_DEVICE NestedType const& Transpose() const { return A; }

  private:
    NestedType const& A;
};

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_TRANSPOSE_H