#ifndef PBAT_MATH_LINALG_MINI_DETERMINANT_H
#define PBAT_MATH_LINALG_MINI_DETERMINANT_H

#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class TMatrix>
PBAT_HOST_DEVICE auto Determinant(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "A must satisfy CMatrix");
    static_assert(
        MatrixType::RowsAtCompileTime == MatrixType::ColsAtCompileTime,
        "Cannot compute determinant of non-square matrix");
    static_assert(
        MatrixType::RowsAtCompileTime < 4,
        "Determinant of matrix of dimensions >= 4 too costly");
    if constexpr (MatrixType::RowsAtCompileTime == 1)
    {
        return A(0, 0);
    }
    if constexpr (MatrixType::RowsAtCompileTime == 2)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }
    if constexpr (MatrixType::RowsAtCompileTime == 3)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
               A(0, 1) * (A(1, 0) * A(2, 2) - A(2, 0) * A(1, 2)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1));
    }
    using ScalarType = typename MatrixType::Scalar;
    return ScalarType{0.};
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_DETERMINANT_H