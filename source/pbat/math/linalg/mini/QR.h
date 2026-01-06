#ifndef PBAT_MATH_LINALG_MINI_QR_H
#define PBAT_MATH_LINALG_MINI_QR_H

#include "Concepts.h"
#include "Matrix.h"
#include "Norm.h"
#include "pbat/HostDevice.h"

#include <cmath>
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
 * @return QRResult containing orthogonal Q and upper triangular R
 *
 * @note For numerical robustness, columns with near-zero norm are handled gracefully
 *       by setting the corresponding Q column to zero and R diagonal to zero.
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto QR(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
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
    // Threshold for near-zero detection (scaled by matrix dimension)
    ScalarType const eps = ScalarType(kRows) * std::numeric_limits<ScalarType>::epsilon();
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
    TScalar const eps = std::numeric_limits<TScalar>::epsilon();

    TScalar const absa = abs(a);
    TScalar const absb = abs(b);
    if (absb < eps)
    {
        c = TScalar{1};
        s = TScalar{0};
    }
    else if (absa < eps)
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

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_QR_H
