#ifndef PBAT_MATH_LINALG_MINI_INVERSE_H
#define PBAT_MATH_LINALG_MINI_INVERSE_H

#include "Concepts.h"
#include "Matrix.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE SMatrix<
    typename std::remove_cvref_t<TMatrix>::ScalarType,
    std::remove_cvref_t<TMatrix>::kRows,
    std::remove_cvref_t<TMatrix>::kCols>
Inverse(TMatrix&& A)
{
    using InputMatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(InputMatrixType);
    static_assert(
        InputMatrixType::kRows < 4 and InputMatrixType::kRows > 1,
        "Cannot compute inverse of large matrix or scalar");
    static_assert(
        InputMatrixType::kRows == InputMatrixType::kCols,
        "Cannot compute inverse of non-square matrix");
    using ScalarType = typename InputMatrixType::ScalarType;
    using MatrixType = SMatrix<ScalarType, InputMatrixType::kRows, InputMatrixType::kCols>;
    MatrixType Ainv{};
    if constexpr (MatrixType::kRows == 2)
    {
        auto const a0 = ScalarType(1.0) / (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
        Ainv(0, 0)    = a0 * A(1, 1);
        Ainv(1, 0)    = -a0 * A(1, 0);
        Ainv(0, 1)    = -a0 * A(0, 1);
        Ainv(1, 1)    = a0 * A(0, 0);
    }
    if constexpr (MatrixType::kRows == 3)
    {
        auto const a0 = A(1, 1) * A(2, 2);
        auto const a1 = A(2, 1) * A(1, 2);
        auto const a2 = A(1, 0) * A(2, 1);
        auto const a3 = A(1, 0) * A(2, 2);
        auto const a4 = A(2, 0) * A(1, 1);
        auto const a5 =
            ScalarType(1.0) / (a0 * A(0, 0) - a1 * A(0, 0) + a2 * A(0, 2) - a3 * A(0, 1) -
                               a4 * A(0, 2) + A(2, 0) * A(0, 1) * A(1, 2));
        Ainv(0, 0) = a5 * (a0 - a1);
        Ainv(1, 0) = a5 * (-a3 + A(2, 0) * A(1, 2));
        Ainv(2, 0) = a5 * (a2 - a4);
        Ainv(0, 1) = a5 * (-A(0, 1) * A(2, 2) + A(2, 1) * A(0, 2));
        Ainv(1, 1) = a5 * (A(0, 0) * A(2, 2) - A(2, 0) * A(0, 2));
        Ainv(2, 1) = a5 * (-A(0, 0) * A(2, 1) + A(2, 0) * A(0, 1));
        Ainv(0, 2) = a5 * (A(0, 1) * A(1, 2) - A(1, 1) * A(0, 2));
        Ainv(1, 2) = a5 * (-A(0, 0) * A(1, 2) + A(1, 0) * A(0, 2));
        Ainv(2, 2) = a5 * (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
    }
    return Ainv;
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_INVERSE_H
