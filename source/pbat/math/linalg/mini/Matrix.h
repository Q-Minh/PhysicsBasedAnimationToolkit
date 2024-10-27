#ifndef PBAT_MATH_LINALG_MINI_MATRIX_CUH
#define PBAT_MATH_LINALG_MINI_MATRIX_CUH

#include "Concepts.h"
#include "SubMatrix.h"
#include "Transpose.h"
#include "pbat/HostDevice.h"

#include <array>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class TScalar, int M, int N>
class Ones
{
  public:
    using Scalar   = TScalar;
    using SelfType = Ones<Scalar, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;
    static bool constexpr IsRowMajor        = false;

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return Scalar{1}; }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const { return Scalar{1}; }

    template <auto S, auto T>
    PBAT_HOST_DEVICE Ones<Scalar, S, T> Slice(auto i, auto j) const
    {
        return Ones<Scalar, S, T>();
    }
    PBAT_HOST_DEVICE Ones<Scalar, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Ones<Scalar, RowsAtCompileTime, 1>();
    }
    PBAT_HOST_DEVICE Ones<Scalar, 1, ColsAtCompileTime> Row(auto i) const
    {
        return Ones<Scalar, 1, ColsAtCompileTime>();
    }
    PBAT_HOST_DEVICE ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }
};

template <class TScalar, int M, int N>
class Zeros
{
  public:
    using Scalar   = TScalar;
    using SelfType = Zeros<Scalar, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;
    static bool constexpr IsRowMajor        = false;

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return Scalar{0}; }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const { return Scalar{0}; }

    template <auto S, auto T>
    PBAT_HOST_DEVICE Zeros<Scalar, S, T> Slice(auto i, auto j) const
    {
        return Zeros<Scalar, S, T>();
    }
    PBAT_HOST_DEVICE Zeros<Scalar, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Zeros<Scalar, RowsAtCompileTime, 1>();
    }
    PBAT_HOST_DEVICE Zeros<Scalar, 1, ColsAtCompileTime> Row(auto i) const
    {
        return Zeros<Scalar, 1, ColsAtCompileTime>();
    }
    PBAT_HOST_DEVICE ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }
};

template <class TScalar, int M, int N>
class Identity
{
  public:
    using Scalar   = TScalar;
    using SelfType = Identity<Scalar, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;
    static bool constexpr IsRowMajor        = false;

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return static_cast<Scalar>(i == j); }

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
};

template <CMatrix TMatrix>
class Diagonal
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = Diagonal<NestedType>;

    static auto constexpr RowsAtCompileTime = NestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = 1;
    static bool constexpr IsRowMajor        = false;

    PBAT_HOST_DEVICE Diagonal(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(i, i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const { return A(i, i); }

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

template <class TScalar, int M, int N = 1>
class SMatrix
{
  public:
    using Scalar   = TScalar;
    using SelfType = SMatrix<Scalar, M, N>;

    PBAT_HOST_DEVICE SMatrix() : a() {}

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;
    static bool constexpr IsRowMajor        = false;

    template <class /*CMatrix*/ TMatrix>
    PBAT_HOST_DEVICE SMatrix(TMatrix&& B) : a()
    {
        using MatrixType = std::remove_cvref_t<TMatrix>;
        static_assert(CMatrix<MatrixType>, "B must satisfy CMatrix");
        static_assert(
            MatrixType::RowsAtCompileTime == RowsAtCompileTime and
                MatrixType::ColsAtCompileTime == ColsAtCompileTime,
            "Invalid matrix assignment dimensions");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<RowsAtCompileTime>()), ...);
        };
        fCols(std::make_index_sequence<ColsAtCompileTime>());
    }

    template <class TMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<RowsAtCompileTime>()), ...);
        };
        fCols(std::make_index_sequence<ColsAtCompileTime>());
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

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return this->a[j * M + i]; }
    PBAT_HOST_DEVICE auto& operator()(auto i, auto j) { return this->a[j * M + i]; }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const { return a[i]; }
    PBAT_HOST_DEVICE auto& operator()(auto i) { return a[i]; }

    // Smart accessors
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

    PBAT_HOST_DEVICE void SetZero()
    {
        memset(a.data(), 0, RowsAtCompileTime * ColsAtCompileTime * sizeof(Scalar));
    }

  private:
    std::array<Scalar, M * N> a;
};

template <class TScalar, int M, int N>
class SMatrixView
{
  public:
    using Scalar   = TScalar;
    using SelfType = SMatrixView<Scalar, M, N>;

    static auto constexpr RowsAtCompileTime = M;
    static auto constexpr ColsAtCompileTime = N;
    static bool constexpr IsRowMajor        = false;

    PBAT_HOST_DEVICE SMatrixView(Scalar* a) : mA(a) {}

    template <class /*CMatrix*/ TMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<RowsAtCompileTime>()), ...);
        };
        fCols(std::make_index_sequence<ColsAtCompileTime>());
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

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return mA[j * M + i]; }
    PBAT_HOST_DEVICE auto& operator()(auto i, auto j) { return mA[j * M + i]; }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const { return mA[i]; }
    PBAT_HOST_DEVICE auto& operator()(auto i) { return mA[i]; }

    // Smart accessors
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

    PBAT_HOST_DEVICE void SetZero()
    {
        memset(mA, 0, RowsAtCompileTime * ColsAtCompileTime * sizeof(Scalar));
    }

  private:
    Scalar* mA;
};

template <CMatrix TMatrix, int RepeatRows, int RepeatCols>
class TiledView
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = TiledView<NestedType, RepeatRows, RepeatCols>;

    static auto constexpr RowsAtCompileTime = RepeatRows * NestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = RepeatCols * NestedType::ColsAtCompileTime;
    static bool constexpr IsRowMajor        = NestedType::IsRowMajor;

    PBAT_HOST_DEVICE TiledView(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const
    {
        return A(i % NestedType::RowsAtCompileTime, j % NestedType::ColsAtCompileTime);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }

    // Smart accessors
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
    PBAT_HOST_DEVICE ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

  private:
    NestedType const& A;
};

template <auto RepeatRows, auto RepeatCols, class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Repeat(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return TiledView<MatrixType, RepeatRows, RepeatCols>(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_MATRIX_CUH