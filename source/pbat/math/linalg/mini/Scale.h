#ifndef PBAT_MATH_LINALG_MINI_SCALE_H
#define PBAT_MATH_LINALG_MINI_SCALE_H

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
class Scale
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = Scale<NestedType>;

    static auto constexpr RowsAtCompileTime = NestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = NestedType::ColsAtCompileTime;
    static bool constexpr IsRowMajor        = NestedType::IsRowMajor;

    PBAT_HOST_DEVICE Scale(Scalar k, NestedType const& A) : k(k), A(A) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return k * A(i, j); }

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
    Scalar k;
    NestedType const& A;
};

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator*(typename std::remove_cvref_t<TMatrix>::Scalar k, TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return Scale<MatrixType>(k, std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator*(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::Scalar k)
{
    return k * std::forward<TMatrix>(A);
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator*=(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::Scalar k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        ((std::forward<TMatrix>(A)(I, j) *= k), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<MatrixType::RowsAtCompileTime>()), ...);
    };
    fCols(std::make_index_sequence<MatrixType::ColsAtCompileTime>());
    return A;
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator/(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::Scalar k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    using Scalar = typename MatrixType::Scalar;
    return Scale<MatrixType>(Scalar(1. / k), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto operator/=(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::Scalar k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        ((std::forward<TMatrix>(A)(I, j) /= k), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<MatrixType::RowsAtCompileTime>()), ...);
    };
    fCols(std::make_index_sequence<MatrixType::ColsAtCompileTime>());
    return A;
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_SCALE_H