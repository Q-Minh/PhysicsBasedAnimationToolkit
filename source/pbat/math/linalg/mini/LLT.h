#ifndef PBAT_MATH_LINALG_MINI_LLT_H
#define PBAT_MATH_LINALG_MINI_LLT_H

#include "Concepts.h"
#include "Matrix.h"
#include "TriangularSolve.h"
#include "pbat/HostDevice.h"

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

/**
 * @brief Result of LLT (Cholesky) decomposition
 * @tparam TScalar Scalar type
 * @tparam N Matrix dimension
 */
template <class TScalar, int N>
struct LLTResult
{
    SMatrix<TScalar, N, N> L; ///< Lower-triangular Cholesky factor
    bool success{true};       ///< Whether the decomposition succeeded (input was SPD)
};

/**
 * @brief Compute LLT (Cholesky) decomposition of a symmetric positive-definite matrix.
 *
 * Given a symmetric positive-definite matrix A, computes a lower-triangular matrix L
 * such that A = L * L^T.
 *
 * Uses the outer-product (left-looking) Cholesky algorithm with O(N^3/3) operations.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input symmetric positive-definite matrix of size N x N
 * @param eps Base epsilon for numerical zero checks on the diagonal, scaled by the
 *        trace of A. Defaults to std::numeric_limits<ScalarType>::epsilon().
 * @return LLTResult containing lower-triangular L and a success flag. If the input
 *         is not positive-definite (a diagonal element becomes non-positive), success
 *         is set to false and L contains the partial factorization.
 *
 * @note Only the lower-triangular part of A is read.
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto LLT(
    TMatrix&& A,
    typename std::decay_t<TMatrix>::ScalarType eps =
        std::numeric_limits<typename std::decay_t<TMatrix>::ScalarType>::epsilon())
{
    using MatrixType = std::decay_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);

    using ScalarType         = typename MatrixType::ScalarType;
    static auto constexpr kN = MatrixType::kRows;
    static_assert(MatrixType::kRows == MatrixType::kCols, "LLT requires a square matrix");

    LLTResult<ScalarType, kN> result{};
    auto& L = result.L;
    L.SetZero();

    // Scale epsilon by the trace (sum of diagonal) for a scale-invariant threshold
    ScalarType tr{0};
    for (auto i = 0; i < kN; ++i)
        tr += A(i, i);
    eps *= tr;

    // Outer-product (left-looking) Cholesky
    for (auto j = 0; j < kN; ++j)
    {
        // Compute diagonal element L(j,j)
        ScalarType diag = A(j, j);
        for (auto k = 0; k < j; ++k)
            diag -= L(j, k) * L(j, k);

        if (diag <= eps)
        {
            result.success = false;
            return result;
        }

        using namespace std;
        L(j, j) = sqrt(diag);

        // Compute off-diagonal elements L(i,j) for i > j
        for (auto i = j + 1; i < kN; ++i)
        {
            ScalarType sum = A(i, j);
            for (auto k = 0; k < j; ++k)
                sum -= L(i, k) * L(j, k);
            L(i, j) = sum / L(j, j);
        }
    }
    return result;
}

/**
 * @brief Solve the linear system A * x = b using Cholesky (LLT) factorization.
 *
 * Given A = L * L^T, solves:
 *   1. L * y = b   (forward substitution)
 *   2. L^T * x = y (back substitution)
 *
 * @tparam TMatrixL Lower-triangular Cholesky factor type (N x N) satisfying CMatrix
 * @tparam TMatrixB Right-hand side matrix type (N x K) satisfying CMatrix
 * @param L Lower-triangular Cholesky factor from LLT()
 * @param b Right-hand side vector/matrix
 * @return Solution x such that (L * L^T) * x = b
 */
template <class /*CMatrix*/ TMatrixL, class /*CMatrix*/ TMatrixB>
PBAT_HOST_DEVICE auto LLTSolve(TMatrixL&& L, TMatrixB&& b)
{
    // Solve L * y = b
    auto y = LowerTriangularSolve(L, std::forward<TMatrixB>(b));
    // Solve L^T * x = y
    return UpperTriangularSolve(L.Transpose(), y);
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_LLT_H
