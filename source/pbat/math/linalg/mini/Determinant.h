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
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(
        MatrixType::kRows == MatrixType::kCols,
        "Cannot compute determinant of non-square matrix");
    static_assert(MatrixType::kRows < 4, "Determinant of matrix of dimensions >= 4 too costly");
    if constexpr (MatrixType::kRows == 1)
    {
        return A(0, 0);
    }
    else if constexpr (MatrixType::kRows == 2)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }
    else if constexpr (MatrixType::kRows == 3)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
               A(0, 1) * (A(1, 0) * A(2, 2) - A(2, 0) * A(1, 2)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1));
    }
    else
    {
        using ScalarType = typename MatrixType::ScalarType;
        return ScalarType{0.};
    }
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_DETERMINANT_H