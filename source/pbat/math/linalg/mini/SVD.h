#ifndef PBAT_MATH_LINALG_MINI_SVD_H
#define PBAT_MATH_LINALG_MINI_SVD_H

#include "Concepts.h"
#include "Eigenvalues.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Transpose.h"
#include "pbat/HostDevice.h"

#include <limits>
#include <math.h>
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
 * @return SVDResult containing U, S (singular values), V such that A = U * diag(S) * V^T
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SVD2x2(TMatrix&& A)
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
    ScalarType const eps = ScalarType{128} * std::numeric_limits<ScalarType>::epsilon();

    ScalarType sigma0Sq = lambda(1); // Largest first for descending order
    ScalarType sigma1Sq = lambda(0);

    // Clamp small negative values
    if (sigma0Sq < ScalarType{0})
        sigma0Sq = ScalarType{0};
    if (sigma1Sq < ScalarType{0})
        sigma1Sq = ScalarType{0};

    ScalarType sigma0, sigma1;
    if constexpr (std::is_same_v<ScalarType, float>)
    {
        sigma0 = sqrtf(sigma0Sq);
        sigma1 = sqrtf(sigma1Sq);
    }
    else
    {
        sigma0 = sqrt(sigma0Sq);
        sigma1 = sqrt(sigma1Sq);
    }

    result.S(0) = sigma0;
    result.S(1) = sigma1;

    // V columns in order corresponding to descending singular values
    result.V(0, 0) = V(0, 1);
    result.V(1, 0) = V(1, 1);
    result.V(0, 1) = V(0, 0);
    result.V(1, 1) = V(1, 0);

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
        ScalarType invNorm;
        if constexpr (std::is_same_v<ScalarType, float>)
            invNorm = ScalarType{1} / sqrtf(u0norm);
        else
            invNorm = ScalarType{1} / sqrt(u0norm);
        result.U(0, 0) *= invNorm;
        result.U(1, 0) *= invNorm;
    }

    // Make second column orthogonal to first
    ScalarType dot = result.U(0, 0) * result.U(0, 1) + result.U(1, 0) * result.U(1, 1);
    result.U(0, 1) -= dot * result.U(0, 0);
    result.U(1, 1) -= dot * result.U(1, 0);

    // Normalize second column
    ScalarType u1norm = result.U(0, 1) * result.U(0, 1) + result.U(1, 1) * result.U(1, 1);
    if (u1norm > eps * eps)
    {
        ScalarType invNorm;
        if constexpr (std::is_same_v<ScalarType, float>)
            invNorm = ScalarType{1} / sqrtf(u1norm);
        else
            invNorm = ScalarType{1} / sqrt(u1norm);
        result.U(0, 1) *= invNorm;
        result.U(1, 1) *= invNorm;
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
 * @return SVDResult containing U, S (singular values), V such that A = U * diag(S) * V^T
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SVD3x3(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 3 and MatrixType::kCols == 3, "Matrix must be 3x3");

    using ScalarType = typename MatrixType::ScalarType;

    SVDResult<ScalarType, 3, 3> result{};

    ScalarType const eps = ScalarType{128} * std::numeric_limits<ScalarType>::epsilon();

    // Compute A^T * A
    SMatrix<ScalarType, 3, 3> AtA;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = i; j < 3; ++j)
        {
            ScalarType sum{0};
            for (int k = 0; k < 3; ++k)
                sum += A(k, i) * A(k, j);
            AtA(i, j) = sum;
            AtA(j, i) = sum;
        }
    }

    // Eigendecomposition of A^T * A gives V and sigma^2
    auto [lambda, V] = SymmetricEigen3x3(AtA);

    // Singular values are sqrt of eigenvalues, in descending order
    // Eigenvalues come in ascending order, so reverse
    for (int j = 0; j < 3; ++j)
    {
        ScalarType sigmaSq = lambda(2 - j);
        if (sigmaSq < ScalarType{0})
            sigmaSq = ScalarType{0};

        ScalarType sigma;
        if constexpr (std::is_same_v<ScalarType, float>)
            sigma = sqrtf(sigmaSq);
        else
            sigma = sqrt(sigmaSq);

        result.S(j) = sigma;

        // V column (reverse order for descending singular values)
        result.V(0, j) = V(0, 2 - j);
        result.V(1, j) = V(1, 2 - j);
        result.V(2, j) = V(2, 2 - j);
    }

    // Compute U = A * V * S^{-1} for non-zero singular values
    for (int j = 0; j < 3; ++j)
    {
        ScalarType sigma = result.S(j);

        // A * v_j
        ScalarType av0 =
            A(0, 0) * result.V(0, j) + A(0, 1) * result.V(1, j) + A(0, 2) * result.V(2, j);
        ScalarType av1 =
            A(1, 0) * result.V(0, j) + A(1, 1) * result.V(1, j) + A(1, 2) * result.V(2, j);
        ScalarType av2 =
            A(2, 0) * result.V(0, j) + A(2, 1) * result.V(1, j) + A(2, 2) * result.V(2, j);

        if (sigma > eps)
        {
            ScalarType invSigma = ScalarType{1} / sigma;
            result.U(0, j)      = av0 * invSigma;
            result.U(1, j)      = av1 * invSigma;
            result.U(2, j)      = av2 * invSigma;
        }
        else
        {
            // Zero singular value: will be fixed by orthogonalization below
            result.U(0, j) = (j == 0) ? ScalarType{1} : ScalarType{0};
            result.U(1, j) = (j == 1) ? ScalarType{1} : ScalarType{0};
            result.U(2, j) = (j == 2) ? ScalarType{1} : ScalarType{0};
        }
    }

    // Orthogonalize U using modified Gram-Schmidt for robustness
    // Column 0
    ScalarType n0 = result.U(0, 0) * result.U(0, 0) + result.U(1, 0) * result.U(1, 0) +
                    result.U(2, 0) * result.U(2, 0);
    if (n0 > eps * eps)
    {
        ScalarType invN0;
        if constexpr (std::is_same_v<ScalarType, float>)
            invN0 = ScalarType{1} / sqrtf(n0);
        else
            invN0 = ScalarType{1} / sqrt(n0);
        result.U(0, 0) *= invN0;
        result.U(1, 0) *= invN0;
        result.U(2, 0) *= invN0;
    }

    // Column 1: orthogonalize against column 0
    ScalarType d01 = result.U(0, 0) * result.U(0, 1) + result.U(1, 0) * result.U(1, 1) +
                     result.U(2, 0) * result.U(2, 1);
    result.U(0, 1) -= d01 * result.U(0, 0);
    result.U(1, 1) -= d01 * result.U(1, 0);
    result.U(2, 1) -= d01 * result.U(2, 0);

    ScalarType n1 = result.U(0, 1) * result.U(0, 1) + result.U(1, 1) * result.U(1, 1) +
                    result.U(2, 1) * result.U(2, 1);
    if (n1 > eps * eps)
    {
        ScalarType invN1;
        if constexpr (std::is_same_v<ScalarType, float>)
            invN1 = ScalarType{1} / sqrtf(n1);
        else
            invN1 = ScalarType{1} / sqrt(n1);
        result.U(0, 1) *= invN1;
        result.U(1, 1) *= invN1;
        result.U(2, 1) *= invN1;
    }
    else
    {
        // Find vector orthogonal to U(:,0)
        ScalarType abs0, abs1;
        if constexpr (std::is_same_v<ScalarType, float>)
        {
            abs0 = fabsf(result.U(0, 0));
            abs1 = fabsf(result.U(1, 0));
        }
        else
        {
            abs0 = fabs(result.U(0, 0));
            abs1 = fabs(result.U(1, 0));
        }

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
        n1 = result.U(0, 1) * result.U(0, 1) + result.U(1, 1) * result.U(1, 1) +
             result.U(2, 1) * result.U(2, 1);
        ScalarType invN1;
        if constexpr (std::is_same_v<ScalarType, float>)
            invN1 = ScalarType{1} / sqrtf(n1);
        else
            invN1 = ScalarType{1} / sqrt(n1);
        result.U(0, 1) *= invN1;
        result.U(1, 1) *= invN1;
        result.U(2, 1) *= invN1;
    }

    // Column 2: cross product of columns 0 and 1
    result.U(0, 2) = result.U(1, 0) * result.U(2, 1) - result.U(2, 0) * result.U(1, 1);
    result.U(1, 2) = result.U(2, 0) * result.U(0, 1) - result.U(0, 0) * result.U(2, 1);
    result.U(2, 2) = result.U(0, 0) * result.U(1, 1) - result.U(1, 0) * result.U(0, 1);

    return result;
}

/**
 * @brief Compute SVD of a matrix.
 *
 * Dispatcher that calls the appropriate 2x2 or 3x3 analytic solver.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input square matrix (2x2 or 3x3)
 * @return SVDResult containing U, S, V such that A = U * diag(S) * V^T
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SVD(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == MatrixType::kCols, "Matrix must be square");
    static_assert(
        MatrixType::kRows == 2 or MatrixType::kRows == 3,
        "Only 2x2 and 3x3 matrices supported");

    if constexpr (MatrixType::kRows == 2)
        return SVD2x2(std::forward<TMatrix>(A));
    else
        return SVD3x3(std::forward<TMatrix>(A));
}

/**
 * @brief Compute only the singular values of a 2x2 matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input 2x2 matrix
 * @return Vector of 2 singular values in descending order
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SingularValues2x2(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 2 and MatrixType::kCols == 2, "Matrix must be 2x2");

    using ScalarType = typename MatrixType::ScalarType;

    // Compute A^T * A
    SMatrix<ScalarType, 2, 2> AtA;
    AtA(0, 0) = A(0, 0) * A(0, 0) + A(1, 0) * A(1, 0);
    AtA(0, 1) = A(0, 0) * A(0, 1) + A(1, 0) * A(1, 1);
    AtA(1, 0) = AtA(0, 1);
    AtA(1, 1) = A(0, 1) * A(0, 1) + A(1, 1) * A(1, 1);

    auto lambda = SymmetricEigenvalues2x2(AtA);

    SVector<ScalarType, 2> singularValues;

    // Singular values in descending order (eigenvalues are ascending)
    for (int i = 0; i < 2; ++i)
    {
        ScalarType sigmaSq = lambda(1 - i);
        if (sigmaSq < ScalarType{0})
            sigmaSq = ScalarType{0};

        if constexpr (std::is_same_v<ScalarType, float>)
            singularValues(i) = sqrtf(sigmaSq);
        else
            singularValues(i) = sqrt(sigmaSq);
    }

    return singularValues;
}

/**
 * @brief Compute only the singular values of a 3x3 matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input 3x3 matrix
 * @return Vector of 3 singular values in descending order
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SingularValues3x3(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 3 and MatrixType::kCols == 3, "Matrix must be 3x3");

    using ScalarType = typename MatrixType::ScalarType;

    // Compute A^T * A
    SMatrix<ScalarType, 3, 3> AtA;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = i; j < 3; ++j)
        {
            ScalarType sum{0};
            for (int k = 0; k < 3; ++k)
                sum += A(k, i) * A(k, j);
            AtA(i, j) = sum;
            AtA(j, i) = sum;
        }
    }

    auto lambda = SymmetricEigenvalues3x3(AtA);

    SVector<ScalarType, 3> singularValues;

    // Singular values in descending order (eigenvalues are ascending)
    for (int i = 0; i < 3; ++i)
    {
        ScalarType sigmaSq = lambda(2 - i);
        if (sigmaSq < ScalarType{0})
            sigmaSq = ScalarType{0};

        if constexpr (std::is_same_v<ScalarType, float>)
            singularValues(i) = sqrtf(sigmaSq);
        else
            singularValues(i) = sqrt(sigmaSq);
    }

    return singularValues;
}

/**
 * @brief Compute only singular values of a matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input square matrix (2x2 or 3x3)
 * @return Vector of singular values in descending order
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SingularValues(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == MatrixType::kCols, "Matrix must be square");
    static_assert(
        MatrixType::kRows == 2 or MatrixType::kRows == 3,
        "Only 2x2 and 3x3 matrices supported");

    if constexpr (MatrixType::kRows == 2)
        return SingularValues2x2(std::forward<TMatrix>(A));
    else
        return SingularValues3x3(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_SVD_H
