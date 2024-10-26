#ifndef PBAT_MATH_LINALG_MINI_TRANSPOSE_H
#define PBAT_MATH_LINALG_MINI_TRANSPOSE_H

#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TMatrix>
class TransposeView
{
  public:
    using NestedType = TMatrix;
    using Scalar     = typename NestedType::Scalar;
    using SelfType   = TransposeView<NestedType>;

    static auto constexpr RowsAtCompileTime = NestedType::ColsAtCompileTime;
    static auto constexpr ColsAtCompileTime = NestedType::RowsAtCompileTime;

    PBAT_HOST_DEVICE TransposeView(NestedType& A) : A(A) {}

    template <class TOtherMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TOtherMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TOtherMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TOtherMatrix>(B)(I, j)), ...);
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

    PBAT_HOST_DEVICE ConstTransposeView(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return A.Cols(); }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return A.Rows(); }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const { return A(j, i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
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