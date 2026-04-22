#ifndef PBAT_MATH_LINALG_MINI_TRIANGULAR_SOLVE_H
#define PBAT_MATH_LINALG_MINI_TRIANGULAR_SOLVE_H

#include "Concepts.h"
#include "Matrix.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

/**
 * @brief Solve the lower-triangular system L * x = b by forward substitution.
 *
 * @tparam TMatrixL Lower-triangular matrix type (N x N) satisfying CMatrix
 * @tparam TMatrixB Right-hand side matrix type (N x K) satisfying CMatrix
 * @param L Lower-triangular matrix
 * @param b Right-hand side vector/matrix
 * @return Solution x such that L * x = b
 *
 * @note Only the lower-triangular part of L is read. The diagonal must be non-zero.
 */
template <class /*CMatrix*/ TMatrixL, class /*CMatrix*/ TMatrixB>
PBAT_HOST_DEVICE auto LowerTriangularSolve(TMatrixL&& L, TMatrixB&& b)
{
    using LType = std::decay_t<TMatrixL>;
    using BType = std::decay_t<TMatrixB>;
    PBAT_MINI_CHECK_CMATRIX(LType);
    PBAT_MINI_CHECK_CMATRIX(BType);

    using ScalarType            = typename LType::ScalarType;
    static auto constexpr kN    = LType::kRows;
    static auto constexpr kK    = BType::kCols;
    static_assert(LType::kRows == LType::kCols, "L must be square");
    static_assert(LType::kCols == BType::kRows, "Dimension mismatch between L and b");

    SMatrix<ScalarType, kN, kK> x = b;

    for (auto k = 0; k < kK; ++k)
    {
        for (auto i = 0; i < kN; ++i)
        {
            for (auto j = 0; j < i; ++j)
            {
                x(i, k) -= L(i, j) * x(j, k);
            }
            x(i, k) /= L(i, i);
        }
    }
    return x;
}

/**
 * @brief Solve the upper-triangular system R * x = b by back substitution.
 *
 * @tparam TMatrixR Upper-triangular matrix type (N x N) satisfying CMatrix
 * @tparam TMatrixB Right-hand side matrix type (N x K) satisfying CMatrix
 * @param R Upper-triangular matrix
 * @param b Right-hand side vector/matrix
 * @return Solution x such that R * x = b
 *
 * @note Only the upper-triangular part of R is read. The diagonal must be non-zero.
 */
template <class /*CMatrix*/ TMatrixR, class /*CMatrix*/ TMatrixB>
PBAT_HOST_DEVICE auto UpperTriangularSolve(TMatrixR&& R, TMatrixB&& b)
{
    using RType = std::decay_t<TMatrixR>;
    using BType = std::decay_t<TMatrixB>;
    PBAT_MINI_CHECK_CMATRIX(RType);
    PBAT_MINI_CHECK_CMATRIX(BType);

    using ScalarType            = typename RType::ScalarType;
    static auto constexpr kN    = RType::kRows;
    static auto constexpr kK    = BType::kCols;
    static_assert(RType::kRows == RType::kCols, "R must be square");
    static_assert(RType::kCols == BType::kRows, "Dimension mismatch between R and b");

    SMatrix<ScalarType, kN, kK> x = b;

    for (auto k = 0; k < kK; ++k)
    {
        for (auto i = kN - 1; i >= 0; --i)
        {
            for (auto j = i + 1; j < kN; ++j)
            {
                x(i, k) -= R(i, j) * x(j, k);
            }
            x(i, k) /= R(i, i);
        }
    }
    return x;
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_TRIANGULAR_SOLVE_H
