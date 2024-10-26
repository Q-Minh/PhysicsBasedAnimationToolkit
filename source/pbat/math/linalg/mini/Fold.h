#ifndef PBAT_MATH_LINALG_MINI_FOLD_H
#define PBAT_MATH_LINALG_MINI_FOLD_H

#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class TMatrix, class FUnaryOp>
PBAT_HOST_DEVICE void Fold(TMatrix&& A, FUnaryOp&& unaryOp)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    if constexpr (MatrixType::IsRowMajor)
    {
        auto fCols = [&]<auto... J>(auto i, std::index_sequence<J...>) {
            ((unaryOp(std::forward<TMatrix>(A)(i, J))), ...);
        };
        auto fRows = [&]<auto... I>(std::index_sequence<I...>) {
            (fCols(I, std::make_index_sequence<MatrixType::ColsAtCompileTime>()), ...);
        };
        fRows(std::make_index_sequence<MatrixType::RowsAtCompileTime>());
    }
    else
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            ((unaryOp(std::forward<TMatrix>(A)(I, j))), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<MatrixType::RowsAtCompileTime>()), ...);
        };
        fCols(std::make_index_sequence<MatrixType::ColsAtCompileTime>());
    }
}

template <class TLhsMatrix, class TRhsMatrix, class FBinaryOp>
PBAT_HOST_DEVICE void Fold(TLhsMatrix&& A, TRhsMatrix&& B, FBinaryOp&& binaryOp)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Left input must satisfy concept CMatrix");
    static_assert(CMatrix<RhsMatrixType>, "Right input must satisfy concept CMatrix");
    if constexpr (LhsMatrixType::IsRowMajor and RhsMatrixType::IsRowMajor)
    {
        auto fCols = [&]<auto... J>(auto i, std::index_sequence<J...>) {
            ((binaryOp(std::forward<TLhsMatrix>(A)(i, J), std::forward<TRhsMatrix>(B)(i, J))), ...);
        };
        auto fRows = [&]<auto... I>(std::index_sequence<I...>) {
            (fCols(I, std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>()), ...);
        };
        fRows(std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>());
    }
    else
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            ((binaryOp(std::forward<TLhsMatrix>(A)(I, j), std::forward<TRhsMatrix>(B)(I, j))), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>()), ...);
        };
        fCols(std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>());
    }
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_FOLD_H