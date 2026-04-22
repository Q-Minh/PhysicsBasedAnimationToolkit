#ifndef PBAT_MATH_LINALG_MINI_QR_H
#define PBAT_MATH_LINALG_MINI_QR_H

#include "BinaryOperations.h"
#include "Concepts.h"
#include "Matrix.h"
#include "Norm.h"
#include "TriangularSolve.h"
#include "UnaryOperations.h"
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
 * @brief Result of QR decomposition
 * @tparam TScalar Scalar type
 * @tparam M Number of rows
 * @tparam N Number of columns
 */
template <class TScalar, int M, int N>
struct QRResult
{
    SMatrix<TScalar, M, N> Q; ///< Orthogonal matrix (thin Q for M > N)
    SMatrix<TScalar, N, N> R; ///< Upper triangular matrix
};

/**
 * @brief Compute QR decomposition using Modified Gram-Schmidt orthogonalization.
 *
 * Modified Gram-Schmidt is numerically more stable than classical Gram-Schmidt
 * as it orthogonalizes against already-computed orthogonal vectors rather than
 * the original vectors, reducing accumulation of rounding errors.
 *
 * This implementation has deterministic O(MN^2) runtime with no branching except
 * for the near-zero column check, making it suitable for GPU execution.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input matrix of size M x N (M >= N for full rank)
 * @param eps Base epsilon for numerical zero checks, scaled internally by matrix norm.
 *        Defaults to std::numeric_limits<ScalarType>::epsilon().
 * @return QRResult containing orthogonal Q and upper triangular R
 *
 * @note For numerical robustness, columns with near-zero norm are handled gracefully
 *       by setting the corresponding Q column to zero and R diagonal to zero.
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto QR(
    TMatrix&& A,
    typename std::decay_t<TMatrix>::ScalarType eps =
        std::numeric_limits<typename std::decay_t<TMatrix>::ScalarType>::epsilon())
{
    using MatrixType = std::decay_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);

    using ScalarType            = typename MatrixType::ScalarType;
    static auto constexpr kRows = MatrixType::kRows;
    static auto constexpr kCols = MatrixType::kCols;
    static_assert(kRows >= kCols, "QR decomposition requires M >= N");
    QRResult<ScalarType, kRows, kCols> result{};
    auto& Q = result.Q;
    auto& R = result.R;
    // Initialize Q with A
    Q = A;
    // Initialize R to zero
    R.SetZero();
    // Scale epsilon by matrix norm
    ScalarType const normA = Norm(A);
    eps *= normA;
    // Modified Gram-Schmidt
    for (auto j = 0; j < kCols; ++j)
    {
        // Compute norm of column j
        ScalarType norm = Norm(Q.Col(j));
        R(j, j)         = norm;
        // Normalize column j (or set to zero if degenerate)
        // Use branchless selection: multiply by 1/norm if norm > eps, else by 0
        ScalarType const invNorm = (norm > eps) ? (ScalarType{1} / norm) : ScalarType{0};
        Q.Col(j) *= invNorm;
        // Orthogonalize remaining columns against column j
        for (auto k = j + 1; k < kCols; ++k)
        {
            // Compute dot product of columns j and k
            R(j, k) = Dot(Q.Col(j), Q.Col(k));
            // Subtract projection
            Q.Col(k) -= R(j, k) * Q.Col(j);
        }
    }
    return result;
}

/**
 * @brief Compute Givens rotation coefficients for zeroing out element b.
 *
 * Given scalars a and b, computes c and s such that:
 * [c  s] [a]   [r]
 * [-s c] [b] = [0]
 *
 * This implementation uses the numerically stable formulation from
 * @cite golub2013matrix.
 *
 * @tparam TScalar Scalar type
 * @param a First element
 * @param b Second element (to be zeroed)
 * @return Pair (c, s) representing cosine and sine of rotation angle
 */
template <class TScalar>
PBAT_HOST_DEVICE auto GivensRotation(TScalar a, TScalar b)
{
    using namespace std;
    TScalar c, s;

    TScalar const absa = abs(a);
    TScalar const absb = abs(b);

    // Scale epsilon by magnitude of inputs (infinity norm, avoids sqrt)
    TScalar const maxAbs = max(absa, absb);
    TScalar const eps    = maxAbs * std::numeric_limits<TScalar>::epsilon();

    // Must be <=, since eps may be zero (for a zero matrix)
    if (absb <= eps)
    {
        c = TScalar{1};
        s = TScalar{0};
    }
    else if (absa <= eps)
    {
        c = TScalar{0};
        s = (b > TScalar{0}) ? TScalar{1} : TScalar{-1};
    }
    else if (absb > absa)
    {
        TScalar const t = a / b;
        TScalar const u = copysign(sqrt(TScalar{1} + t * t), b);
        s               = TScalar{1} / u;
        c               = s * t;
    }
    else
    {
        TScalar const t = b / a;
        TScalar const u = copysign(sqrt(TScalar{1} + t * t), a);
        c               = TScalar{1} / u;
        s               = c * t;
    }
    return SVector<TScalar, 2>{c, s};
}

/**
 * @brief Solve the upper-triangular system R * x = y by back substitution,
 *        skipping rows whose diagonal R(i,i) is numerically zero.
 *
 * When R(i,i) <= eps, the i-th component of x is set to zero instead of
 * dividing by a near-zero value. This yields the minimum-norm least-squares
 * solution for rank-deficient systems obtained from QR decomposition.
 *
 * @tparam TMatrixR Upper-triangular matrix type (N x N) satisfying CMatrix
 * @tparam TMatrixB Right-hand side matrix type (N x K) satisfying CMatrix
 * @param R Upper-triangular matrix from QR()
 * @param b Right-hand side vector/matrix (typically Q^T * b)
 * @param eps Threshold below which a diagonal element is treated as zero.
 *        Defaults to std::numeric_limits<ScalarType>::epsilon().
 * @return Solution x; components corresponding to zero diagonals are set to zero.
 */
template <class /*CMatrix*/ TMatrixR, class /*CMatrix*/ TMatrixB>
PBAT_HOST_DEVICE auto
RankDeficientUpperTriangularSolve(
    TMatrixR&& R,
    TMatrixB&& b,
    typename std::decay_t<TMatrixR>::ScalarType eps =
        std::numeric_limits<typename std::decay_t<TMatrixR>::ScalarType>::epsilon())
{
    using RType = std::decay_t<TMatrixR>;
    using BType = std::decay_t<TMatrixB>;
    PBAT_MINI_CHECK_CMATRIX(RType);
    PBAT_MINI_CHECK_CMATRIX(BType);

    using ScalarType         = typename RType::ScalarType;
    static auto constexpr kN = RType::kRows;
    static auto constexpr kK = BType::kCols;
    static_assert(RType::kRows == RType::kCols, "R must be square");
    static_assert(RType::kCols == BType::kRows, "Dimension mismatch between R and b");

    // Scale epsilon by the largest diagonal magnitude for a scale-invariant threshold
    using namespace std;
    ScalarType maxDiag{0};
    for (auto i = 0; i < kN; ++i)
        maxDiag = max(maxDiag, abs(R(i, i)));
    eps *= maxDiag;

    SMatrix<ScalarType, kN, kK> x = b;

    for (auto k = 0; k < kK; ++k)
    {
        for (auto i = kN - 1; i >= 0; --i)
        {
            for (auto j = i + 1; j < kN; ++j)
            {
                x(i, k) -= R(i, j) * x(j, k);
            }
            // Skip rank-deficient rows: set x_i = 0 instead of dividing by ~0
            x(i, k) = (abs(R(i, i)) > eps) ? (x(i, k) / R(i, i)) : ScalarType{0};
        }
    }
    return x;
}

/**
 * @brief Solve the linear system A * x = b using QR factorization.
 *
 * Given A = Q * R, solves:
 *   1. Compute y = Q^T * b
 *   2. Solve R * x = y (back substitution, tolerating zero diagonals in R)
 *
 * When A is rank-deficient, some diagonal elements of R will be zero (corresponding
 * to linearly dependent columns detected during QR()). The solve sets the
 * corresponding components of x to zero, yielding the minimum-norm least-squares
 * solution. This mirrors the approach used by the EVD solver path, which skips
 * degenerate eigenvalue directions.
 *
 * @tparam TMatrixQ Orthogonal matrix type (M x N) satisfying CMatrix
 * @tparam TMatrixR Upper-triangular matrix type (N x N) satisfying CMatrix
 * @tparam TMatrixB Right-hand side matrix type (M x K) satisfying CMatrix
 * @param Q Orthogonal factor from QR()
 * @param R Upper-triangular factor from QR()
 * @param b Right-hand side vector/matrix
 * @param eps Base epsilon for detecting zero diagonals in R, scaled internally
 *        by the largest diagonal magnitude. Defaults to
 *        std::numeric_limits<ScalarType>::epsilon().
 * @return Solution x such that A * x = b in the least-squares sense
 */
template <class /*CMatrix*/ TMatrixQ, class /*CMatrix*/ TMatrixR, class /*CMatrix*/ TMatrixB>
PBAT_HOST_DEVICE auto QRSolve(
    TMatrixQ&& Q,
    TMatrixR&& R,
    TMatrixB&& b,
    typename std::decay_t<TMatrixR>::ScalarType eps =
        std::numeric_limits<typename std::decay_t<TMatrixR>::ScalarType>::epsilon())
{
    // Compute y = Q^T * b
    auto y = Q.Transpose() * b;
    // Solve R * x = y, skipping rank-deficient rows
    return RankDeficientUpperTriangularSolve(std::forward<TMatrixR>(R), y, eps);
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_QR_H
