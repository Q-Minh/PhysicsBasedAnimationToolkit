#ifndef PBAT_MATH_LINALG_MINI_REDUCTIONS_H
#define PBAT_MATH_LINALG_MINI_REDUCTIONS_H

#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Trace(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    static_assert(
        MatrixType::RowsAtCompileTime == MatrixType::ColsAtCompileTime,
        "Cannot compute trace of non-square matrix");
    auto sum = [&]<auto... I>(std::index_sequence<I...>) {
        return (std::forward<TMatrix>(A)(I, I) + ...);
    };
    return sum(std::make_index_sequence<MatrixType::RowsAtCompileTime>{});
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto Dot(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return Trace(std::forward<TLhsMatrix>(A).Transpose() * std::forward<TRhsMatrix>(B));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_REDUCTIONS_H