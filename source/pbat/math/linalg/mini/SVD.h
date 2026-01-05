#ifndef PBAT_MATH_LINALG_MINI_SVD_H
#define PBAT_MATH_LINALG_MINI_SVD_H

#include "BinaryOperations.h"
#include "Concepts.h"
#include "Eigenvalues.h"
#include "Geometry.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Transpose.h"
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
 * @brief Result of Singular Value Decomposition
 * @tparam TScalar Scalar type
 * @tparam M Number of rows
 * @tparam N Number of columns
 */
template <class TScalar, int M, int N>
struct SVDResult
{
    SMatrix<TScalar, M, M> U;            ///< Left singular vectors (orthonormal)
    SVector<TScalar, (M < N ? M : N)> S; ///< Singular values (non-negative, descending)
    SMatrix<TScalar, N, N> V;            ///< Right singular vectors (orthonormal)
};

/**
 * @brief Compute SVD of a 2x2 matrix using analytic formulas.
 *
 * Uses the eigenvalue decomposition of A^T*A to compute V and singular values,
 * then computes U = A*V*S^{-1} with special handling for zero singular values.
 *
 * This is an O(1) algorithm with deterministic runtime, suitable for GPU execution.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input 2x2 matrix
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return SVDResult containing U, S (singular values), V such that A = U * diag(S) * V^T
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SVD2x2(TMatrix&& A, bool bSortSingularValues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 2 and MatrixType::kCols == 2, "Matrix must be 2x2");
    using ScalarType = typename MatrixType::ScalarType;
    SVDResult<ScalarType, 2, 2> result{};
    // Compute A^T * A
    SMatrix<ScalarType, 2, 2> AtA;
    AtA(0, 0) = A(0, 0) * A(0, 0) + A(1, 0) * A(1, 0);
    AtA(0, 1) = A(0, 0) * A(0, 1) + A(1, 0) * A(1, 1);
    AtA(1, 0) = AtA(0, 1);
    AtA(1, 1) = A(0, 1) * A(0, 1) + A(1, 1) * A(1, 1);
    // Eigendecomposition of A^T * A gives V and sigma^2
    auto [lambda, V] = SymmetricEigen2x2(AtA);
    // Singular values are sqrt of eigenvalues (clamp negatives from numerical error)
    ScalarType const eps = ScalarType{4} * std::numeric_limits<ScalarType>::epsilon(); // 2x2 matrix
    // Eigenvalues come in ascending order; optionally reverse for descending singular values
    int const idx0      = bSortSingularValues ? 1 : 0;
    int const idx1      = bSortSingularValues ? 0 : 1;
    ScalarType sigma0Sq = lambda(idx0);
    ScalarType sigma1Sq = lambda(idx1);
    // Clamp small negative values
    using namespace std;
    sigma0Sq            = max(sigma0Sq, ScalarType{0});
    sigma1Sq            = max(sigma1Sq, ScalarType{0});
    ScalarType sigma0   = sqrt(sigma0Sq);
    ScalarType sigma1   = sqrt(sigma1Sq);
    result.S(0) = sigma0;
    result.S(1) = sigma1;
    // V columns in order corresponding to singular values
    result.V(0, 0) = V(0, idx0);
    result.V(1, 0) = V(1, idx0);
    result.V(0, 1) = V(0, idx1);
    result.V(1, 1) = V(1, idx1);
    // Compute U = A * V * S^{-1}
    // For each column: u_i = A * v_i / sigma_i
    for (int j = 0; j < 2; ++j)
    {
        ScalarType sigma = result.S(j);
        ScalarType vx    = result.V(0, j);
        ScalarType vy    = result.V(1, j);
        // A * v
        ScalarType avx = A(0, 0) * vx + A(0, 1) * vy;
        ScalarType avy = A(1, 0) * vx + A(1, 1) * vy;
        if (sigma > eps)
        {
            ScalarType invSigma = ScalarType{1} / sigma;
            result.U(0, j)      = avx * invSigma;
            result.U(1, j)      = avy * invSigma;
        }
        else
        {
            // Zero singular value: need to find orthogonal vector
            if (j == 0)
            {
                // First column with zero sigma: use any unit vector
                result.U(0, 0) = ScalarType{1};
                result.U(1, 0) = ScalarType{0};
            }
            else
            {
                // Second column: orthogonal to first
                result.U(0, 1) = -result.U(1, 0);
                result.U(1, 1) = result.U(0, 0);
            }
        }
    }
    // Ensure U is orthonormal (correct for any numerical drift)
    // Normalize first column
    ScalarType u0norm = result.U(0, 0) * result.U(0, 0) + result.U(1, 0) * result.U(1, 0);
    if (u0norm > eps * eps)
    {
        using namespace std;
        ScalarType invNorm = ScalarType{1} / sqrt(u0norm);
        result.U(0, 0) *= invNorm;
        result.U(1, 0) *= invNorm;
    }
    // Make second column orthogonal to first
    ScalarType dot = Dot(result.U.Col(0), result.U.Col(1));
    result.U.Col(1) -= dot * result.U.Col(0);
    // Normalize second column
    ScalarType u1norm = Dot(result.U.Col(1), result.U.Col(1));
    if (u1norm > eps * eps)
    {
        using namespace std;
        ScalarType invNorm = ScalarType{1} / sqrt(u1norm);
        result.U.Col(1) *= invNorm;
    }
    else
    {
        // Degenerate case: construct orthogonal vector
        result.U(0, 1) = -result.U(1, 0);
        result.U(1, 1) = result.U(0, 0);
    }
    return result;
}

/**
 * @brief Compute SVD of a 3x3 matrix using eigenvalue decomposition.
 *
 * Uses the eigenvalue decomposition of A^T*A to compute V and singular values,
 * then computes U = A*V*S^{-1} with orthogonalization for robustness.
 *
 * This is an O(1) algorithm with deterministic runtime, suitable for GPU execution.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input 3x3 matrix
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return SVDResult containing U, S (singular values), V such that A = U * diag(S) * V^T
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SVD3x3(TMatrix&& A, bool bSortSingularValues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 3 and MatrixType::kCols == 3, "Matrix must be 3x3");

    using ScalarType = typename MatrixType::ScalarType;

    SVDResult<ScalarType, 3, 3> result{};

    ScalarType const eps = ScalarType{9} * std::numeric_limits<ScalarType>::epsilon(); // 3x3 matrix

    // Compute A^T * A
    SMatrix<ScalarType, 3, 3> AtA = A.Transpose() * A;

    // Eigendecomposition of A^T * A gives V and sigma^2
    auto [lambda, V] = SymmetricEigen3x3(AtA);

    // Singular values are sqrt of eigenvalues
    // Eigenvalues come in ascending order; optionally reverse for descending singular values
    for (int j = 0; j < 3; ++j)
    {
        int const srcIdx   = bSortSingularValues ? (2 - j) : j;
        ScalarType sigmaSq = lambda(srcIdx);
        using namespace std;
        sigmaSq          = max(sigmaSq, ScalarType{0});
        ScalarType sigma = sqrt(sigmaSq);
        result.S(j) = sigma;
        // V column
        result.V.Col(j) = V.Col(srcIdx);
    }
    // Compute U = A * V * S^{-1} for non-zero singular values
    result.U.SetZero();
    for (int j = 0; j < 3; ++j)
    {
        ScalarType sigma = result.S(j);
        // A * v_j
        SVector<ScalarType, 3> Avj = A * result.V.Col(j);
        if (sigma > eps)
        {
            ScalarType invSigma = ScalarType{1} / sigma;
            result.U.Col(j)     = Avj * invSigma;
        }
    }

    // Orthogonalize U using modified Gram-Schmidt for robustness
    // Column 0
    ScalarType n0 = Norm(result.U.Col(0));
    if (n0 > eps)
    {
        ScalarType invN0 = ScalarType{1} / n0;
        result.U.Col(0) *= invN0;
    }
    else
    {
        // First column is zero, use a unit vector
        result.U(0, 0) = ScalarType{1};
        result.U(1, 0) = ScalarType{0};
        result.U(2, 0) = ScalarType{0};
    }

    // Column 1: orthogonalize against column 0
    ScalarType d01 = Dot(result.U.Col(0), result.U.Col(1));
    result.U.Col(1) -= d01 * result.U.Col(0);
    ScalarType n1 = Norm(result.U.Col(1));
    if (n1 > eps)
    {
        ScalarType invN1 = ScalarType{1} / n1;
        result.U(0, 1) *= invN1;
        result.U(1, 1) *= invN1;
        result.U(2, 1) *= invN1;
    }
    else
    {
        // Find vector orthogonal to U(:,0)
        using namespace std;
        ScalarType abs0 = fabs(result.U(0, 0));
        ScalarType abs1 = fabs(result.U(1, 0));
        if (abs0 < abs1)
        {
            // Cross with x-axis
            result.U(0, 1) = ScalarType{0};
            result.U(1, 1) = -result.U(2, 0);
            result.U(2, 1) = result.U(1, 0);
        }
        else
        {
            // Cross with y-axis
            result.U(0, 1) = result.U(2, 0);
            result.U(1, 1) = ScalarType{0};
            result.U(2, 1) = -result.U(0, 0);
        }
        n1               = Norm(result.U.Col(1));
        ScalarType invN1 = ScalarType{1} / n1;
        result.U(0, 1) *= invN1;
        result.U(1, 1) *= invN1;
        result.U(2, 1) *= invN1;
    }

    // Column 2: orthogonalize against columns 0 and 1, then normalize
    ScalarType d02 = Dot(result.U.Col(0), result.U.Col(2));
    ScalarType d12 = Dot(result.U.Col(1), result.U.Col(2));
    result.U.Col(2) -= d02 * result.U.Col(0);
    result.U.Col(2) -= d12 * result.U.Col(1);
    ScalarType n2 = Norm(result.U.Col(2));
    if (n2 > eps)
    {
        // We have a valid third column from A * V(:,2) / sigma_2
        // Normalize it
        ScalarType invN2 = ScalarType{1} / n2;
        result.U(0, 2) *= invN2;
        result.U(1, 2) *= invN2;
        result.U(2, 2) *= invN2;

        // Check if it's consistent with a right-handed system
        // If not, we need to flip both U(:,2) and V(:,2) to maintain A = U * S * V^T
        // with non-negative singular values
        SVector<ScalarType, 3> crossU01 = Cross(result.U.Col(0), result.U.Col(1));
        ScalarType dotCheck             = Dot(crossU01, result.U.Col(2));
        if (dotCheck < ScalarType{0})
        {
            // The third column is in the wrong direction
            // Flip both U(:,2) and V(:,2) to maintain A = U * S * V^T
            // (flipping both keeps the product U * S * V^T unchanged)
            result.U(0, 2) = -result.U(0, 2);
            result.U(1, 2) = -result.U(1, 2);
            result.U(2, 2) = -result.U(2, 2);
            result.V(0, 2) = -result.V(0, 2);
            result.V(1, 2) = -result.V(1, 2);
            result.V(2, 2) = -result.V(2, 2);
        }
    }
    else
    {
        // Third column is degenerate (zero singular value), use cross product
        result.U.Col(2) = Cross(result.U.Col(0), result.U.Col(1));
    }

    return result;
}

/**
 * @brief Compute SVD of a matrix.
 *
 * Dispatcher that calls the appropriate 2x2 or 3x3 analytic solver.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input square matrix (2x2 or 3x3)
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return SVDResult containing U, S, V such that A = U * diag(S) * V^T
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SVD(TMatrix&& A, bool bSortSingularValues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == MatrixType::kCols, "Matrix must be square");
    static_assert(
        MatrixType::kRows == 2 or MatrixType::kRows == 3,
        "Only 2x2 and 3x3 matrices supported");

    if constexpr (MatrixType::kRows == 2)
        return SVD2x2(std::forward<TMatrix>(A), bSortSingularValues);
    else
        return SVD3x3(std::forward<TMatrix>(A), bSortSingularValues);
}

/**
 * @brief Compute only the singular values of a 2x2 matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input 2x2 matrix
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return Vector of 2 singular values (descending order if bSortSingularValues is true)
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SingularValues2x2(TMatrix&& A, bool bSortSingularValues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 2 and MatrixType::kCols == 2, "Matrix must be 2x2");

    using ScalarType = typename MatrixType::ScalarType;

    // Compute A^T * A
    SMatrix<ScalarType, 2, 2> AtA;
    AtA(0, 0)   = A(0, 0) * A(0, 0) + A(1, 0) * A(1, 0);
    AtA(0, 1)   = A(0, 0) * A(0, 1) + A(1, 0) * A(1, 1);
    AtA(1, 0)   = AtA(0, 1);
    AtA(1, 1)   = A(0, 1) * A(0, 1) + A(1, 1) * A(1, 1);
    auto lambda = SymmetricEigenvalues2x2(AtA);
    SVector<ScalarType, 2> singularValues;
    // Eigenvalues are ascending; optionally reverse for descending singular values
    for (int i = 0; i < 2; ++i)
    {
        int const srcIdx   = bSortSingularValues ? (1 - i) : i;
        ScalarType sigmaSq = lambda(srcIdx);
        using namespace std;
        sigmaSq              = max(sigmaSq, ScalarType{0});
        singularValues(i)    = sqrt(sigmaSq);
    }
    return singularValues;
}

/**
 * @brief Compute only the singular values of a 3x3 matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input 3x3 matrix
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return Vector of 3 singular values (descending order if bSortSingularValues is true)
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SingularValues3x3(TMatrix&& A, bool bSortSingularValues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 3 and MatrixType::kCols == 3, "Matrix must be 3x3");
    using ScalarType = typename MatrixType::ScalarType;
    // Compute A^T * A
    SMatrix<ScalarType, 3, 3> AtA = A.Transpose() * A;
    auto lambda                   = SymmetricEigenvalues3x3(AtA);
    SVector<ScalarType, 3> singularValues;
    // Eigenvalues are ascending; optionally reverse for descending singular values
    for (int i = 0; i < 3; ++i)
    {
        int const srcIdx   = bSortSingularValues ? (2 - i) : i;
        ScalarType sigmaSq = lambda(srcIdx);
        using namespace std;
        sigmaSq              = max(sigmaSq, ScalarType{0});
        singularValues(i)    = sqrt(sigmaSq);
    }
    return singularValues;
}

/**
 * @brief Compute only singular values of a matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input square matrix (2x2 or 3x3)
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return Vector of singular values (descending order if bSortSingularValues is true)
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SingularValues(TMatrix&& A, bool bSortSingularValues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == MatrixType::kCols, "Matrix must be square");
    static_assert(
        MatrixType::kRows == 2 or MatrixType::kRows == 3,
        "Only 2x2 and 3x3 matrices supported");

    if constexpr (MatrixType::kRows == 2)
        return SingularValues2x2(std::forward<TMatrix>(A), bSortSingularValues);
    else
        return SingularValues3x3(std::forward<TMatrix>(A), bSortSingularValues);
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_SVD_H
