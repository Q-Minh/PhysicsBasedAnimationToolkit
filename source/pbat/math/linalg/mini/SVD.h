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
    sigma0Sq          = max(sigma0Sq, ScalarType{0});
    sigma1Sq          = max(sigma1Sq, ScalarType{0});
    ScalarType sigma0 = sqrt(sigma0Sq);
    ScalarType sigma1 = sqrt(sigma1Sq);
    result.S(0)       = sigma0;
    result.S(1)       = sigma1;
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
        result.S(j)      = sigma;
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
        sigmaSq           = max(sigmaSq, ScalarType{0});
        singularValues(i) = sqrt(sigmaSq);
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
        sigmaSq           = max(sigmaSq, ScalarType{0});
        singularValues(i) = sqrt(sigmaSq);
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

/**
 * @brief Compute Jacobi rotation parameters for a 2x2 symmetric matrix.
 *
 * Given a 2x2 symmetric matrix:
 *   [a  b]
 *   [b  c]
 *
 * Computes the rotation angle θ such that:
 *   [cos(θ)  -sin(θ)] [a  b] [cos(θ)   sin(θ)]   [d1  0]
 *   [sin(θ)   cos(θ)] [b  c] [-sin(θ)  cos(θ)] = [0  d2]
 *
 * Returns (cos(θ), sin(θ)) as a 2-vector.
 *
 * @tparam TScalar Scalar type
 * @param a Top-left element
 * @param b Off-diagonal element
 * @param c Bottom-right element
 * @return SVector<TScalar, 2> containing {cos(θ), sin(θ)}
 */
template <class TScalar>
PBAT_HOST_DEVICE auto JacobiRotation(TScalar a, TScalar b, TScalar c) -> SVector<TScalar, 2>
{
    using namespace std;

    TScalar const eps = std::numeric_limits<TScalar>::epsilon();

    // If b is essentially zero, no rotation needed
    if (fabs(b) < eps * (fabs(a) + fabs(c) + TScalar{1}))
    {
        return SVector<TScalar, 2>{TScalar{1}, TScalar{0}};
    }

    // Compute rotation angle using the formula:
    // tan(2θ) = 2b / (a - c)
    // We use a numerically stable formulation

    TScalar const tau = (c - a) / (TScalar{2} * b);
    TScalar t;

    // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
    // This avoids catastrophic cancellation
    if (tau >= TScalar{0})
        t = TScalar{1} / (tau + sqrt(TScalar{1} + tau * tau));
    else
        t = TScalar{-1} / (-tau + sqrt(TScalar{1} + tau * tau));

    // cos(θ) = 1 / sqrt(1 + t^2)
    TScalar const cosTheta = TScalar{1} / sqrt(TScalar{1} + t * t);
    TScalar const sinTheta = t * cosTheta;

    return SVector<TScalar, 2>{cosTheta, sinTheta};
}

/**
 * @brief Compute SVD of a general MxN matrix using the one-sided Jacobi method.
 *
 * The Jacobi SVD is a highly accurate iterative method that computes the SVD
 * by applying a sequence of Jacobi rotations. This implementation uses the
 * one-sided Jacobi method which operates on A directly rather than A^T*A,
 * avoiding the squaring of the condition number.
 *
 * Algorithm:
 * 1. Initialize V = I
 * 2. Repeat until convergence:
 *    - For each pair of columns (i, j) with i < j:
 *      - Compute the 2x2 Gram matrix G = [A_i·A_i  A_i·A_j; A_j·A_i  A_j·A_j]
 *      - Compute Jacobi rotation to diagonalize G
 *      - Apply rotation to columns i,j of A and V
 * 3. Singular values are column norms of final A
 * 4. U columns are normalized A columns
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input MxN matrix (M >= N for this implementation)
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @param maxSweeps Maximum number of sweeps through all column pairs (default: 20)
 * @return SVDResult containing U, S (singular values), V such that A = U * diag(S) * V^T
 *
 * @note This implementation assumes M >= N. For M < N, transpose the matrix,
 *       compute SVD, then swap U and V.
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto JacobiSVD(TMatrix&& A, bool bSortSingularValues = true, int maxSweeps = 20)
{
    using namespace std;

    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);

    using ScalarType             = typename MatrixType::ScalarType;
    static auto constexpr kRows  = MatrixType::kRows;
    static auto constexpr kCols  = MatrixType::kCols;
    static auto constexpr kRank  = (kRows < kCols) ? kRows : kCols;
    static auto constexpr kTrans = (kRows < kCols);

    // For M < N, we work with A^T instead
    // SVD(A) = U * S * V^T  =>  SVD(A^T) = V * S * U^T
    // So we compute SVD of the transpose and swap U, V at the end

    // Working dimensions (ensure we work with a "tall" or square matrix)
    static auto constexpr kWorkRows = kTrans ? kCols : kRows;
    static auto constexpr kWorkCols = kTrans ? kRows : kCols;

    SVDResult<ScalarType, kRows, kCols> result{};

    // Working copy of the matrix (possibly transposed)
    SMatrix<ScalarType, kWorkRows, kWorkCols> B;
    if constexpr (kTrans)
    {
        B = A.Transpose();
    }
    else
    {
        B = A;
    }

    // Initialize V to identity
    SMatrix<ScalarType, kWorkCols, kWorkCols> V = Identity<ScalarType, kWorkCols, kWorkCols>();

    // Convergence tolerance
    ScalarType const eps = ScalarType(kWorkRows) * std::numeric_limits<ScalarType>::epsilon();

    // Jacobi sweeps
    for (int sweep = 0; sweep < maxSweeps; ++sweep)
    {
        // Track if any rotation was significant
        ScalarType maxOffDiag = ScalarType{0};

        // Sweep through all column pairs
        for (int i = 0; i < kWorkCols - 1; ++i)
        {
            for (int j = i + 1; j < kWorkCols; ++j)
            {
                // Compute elements of the 2x2 Gram matrix for columns i and j
                // G = [B_i · B_i,  B_i · B_j]
                //     [B_j · B_i,  B_j · B_j]
                ScalarType aii = ScalarType{0};
                ScalarType aij = ScalarType{0};
                ScalarType ajj = ScalarType{0};

                for (int k = 0; k < kWorkRows; ++k)
                {
                    aii += B(k, i) * B(k, i);
                    aij += B(k, i) * B(k, j);
                    ajj += B(k, j) * B(k, j);
                }

                // Track maximum off-diagonal for convergence check
                ScalarType absAij = fabs(aij);
                if (absAij > maxOffDiag)
                    maxOffDiag = absAij;

                // Skip if already orthogonal
                if (absAij < eps * (sqrt(aii * ajj) + ScalarType{1}))
                    continue;

                // Compute Jacobi rotation to zero out aij
                SVector<ScalarType, 2> cosSinTheta = JacobiRotation(aii, aij, ajj);
                ScalarType cosTheta                = cosSinTheta(0);
                ScalarType sinTheta                = cosSinTheta(1);

                // Apply rotation to columns i and j of B: B := B * R
                for (int k = 0; k < kWorkRows; ++k)
                {
                    ScalarType bi = B(k, i);
                    ScalarType bj = B(k, j);
                    B(k, i)       = cosTheta * bi - sinTheta * bj;
                    B(k, j)       = sinTheta * bi + cosTheta * bj;
                }

                // Apply rotation to columns i and j of V: V := V * R
                for (int k = 0; k < kWorkCols; ++k)
                {
                    ScalarType vi = V(k, i);
                    ScalarType vj = V(k, j);
                    V(k, i)       = cosTheta * vi - sinTheta * vj;
                    V(k, j)       = sinTheta * vi + cosTheta * vj;
                }
            }
        }

        // Check for convergence
        if (maxOffDiag < eps)
            break;
    }

    // Extract singular values (column norms of B) and compute U (normalized columns)
    SMatrix<ScalarType, kWorkRows, kWorkRows> U = Identity<ScalarType, kWorkRows, kWorkRows>();
    SVector<ScalarType, kRank> S;
    for (int j = 0; j < kRank; ++j)
    {
        // Compute column norm - this is the singular value
        ScalarType sigma = Norm(B.Col(j));
        S(j)             = sigma;
    }

    // Compute U columns from the original relationship: A*V = U*Σ
    // So U_j = A*V_j / σ_j for non-zero σ_j
    // This avoids accumulating numerical errors from the Jacobi iterations in B.
    if constexpr (kTrans)
    {
        U.template Slice<kWorkRows, kWorkCols>(0, 0) = A.Transpose() * V;
    }
    else
    {
        U.template Slice<kWorkRows, kWorkCols>(0, 0) = A * V;
    }
    for (int j = 0; j < kRank; ++j)
    {
        if (S(j) > eps)
        {
            // U_j = (A * V_j) / σ_j, but B = A * V already (for M >= N case)
            // For M < N, we transposed, so B = A^T * V, meaning B_j = A^T * V_j
            // Actually B was computed as the working matrix, let's just use A*V directly
            ScalarType invSigma = ScalarType{1} / S(j);
            U.Col(j) *= invSigma;
        }
    }

    // Complete U to an orthonormal basis for columns with zero singular values
    // and for extra columns (j >= kRank).
    // We process in two passes:
    // 1. First orthogonalize zero-singular-value columns against all valid columns
    // 2. Then orthogonalize them against each other in order

    // Cache which columns have near-zero singular values
    SVector<bool, kRank> bNearZeroSV = S <= eps;

    // First pass: orthogonalize each zero-sv column against ALL valid columns
    for (int j = 0; j < kWorkRows; ++j)
    {
        bool bNeedsOrthogonalization = (j >= kRank) or bNearZeroSV(j);
        if (bNeedsOrthogonalization)
        {
            U.Col(j) = Unit<ScalarType, kWorkRows>(j);
            // Orthogonalize against all valid columns (non-zero singular values)
            for (int k = 0; k < j; ++k)
            {
                if (not bNearZeroSV(k))
                {
                    ScalarType dot = Dot(U.Col(k), U.Col(j));
                    U.Col(j) -= dot * U.Col(k);
                }
            }
            for (int k = j + 1; k < kRank; ++k)
            {
                if (not bNearZeroSV(k))
                {
                    ScalarType dot = Dot(U.Col(k), U.Col(j));
                    U.Col(j) -= dot * U.Col(k);
                }
            }
        }
    }

    // Second pass: orthogonalize zero-sv columns against previously processed zero-sv columns
    // and normalize
    for (int j = 0; j < kWorkRows; ++j)
    {
        bool bNeedsOrthogonalization = (j >= kRank) or bNearZeroSV(j);
        if (bNeedsOrthogonalization)
        {
            // Orthogonalize against previously processed zero-sv columns
            for (int k = 0; k < j; ++k)
            {
                bool kWasProcessed = (k >= kRank) or bNearZeroSV(k);
                if (kWasProcessed)
                {
                    ScalarType dot = Dot(U.Col(k), U.Col(j));
                    U.Col(j) -= dot * U.Col(k);
                }
            }
            // Normalize
            ScalarType norm = Norm(U.Col(j));
            if (norm > eps)
            {
                ScalarType invNorm = ScalarType{1} / norm;
                U.Col(j) *= invNorm;
            }
        }
    }

    // Sort singular values in descending order if requested
    if (bSortSingularValues)
    {
        // Selection sort (efficient for small matrices)
        for (int i = 0; i < kRank - 1; ++i)
        {
            int maxIdx = i;
            for (int j = i + 1; j < kRank; ++j)
            {
                if (S(j) > S(maxIdx))
                    maxIdx = j;
            }
            if (maxIdx != i)
            {
                // Swap singular values
                ScalarType tmp = S(i);
                S(i)           = S(maxIdx);
                S(maxIdx)      = tmp;

                // Swap U columns
                for (int k = 0; k < kWorkRows; ++k)
                {
                    tmp          = U(k, i);
                    U(k, i)      = U(k, maxIdx);
                    U(k, maxIdx) = tmp;
                }

                // Swap V columns
                for (int k = 0; k < kWorkCols; ++k)
                {
                    tmp          = V(k, i);
                    V(k, i)      = V(k, maxIdx);
                    V(k, maxIdx) = tmp;
                }
            }
        }
    }

    // Copy results, handling the transpose case
    if constexpr (kTrans)
    {
        // For A^T: U and V are swapped
        // A = U_orig * S * V_orig^T
        // A^T = V_orig * S * U_orig^T
        // We computed SVD(A^T) = U * S * V^T, so U_orig = V, V_orig = U
        // result.U is kRows x kRows, V is kWorkCols x kWorkCols = kRows x kRows
        // result.V is kCols x kCols, U is kWorkRows x kWorkRows = kCols x kCols
        for (int i = 0; i < kRows; ++i)
            for (int j = 0; j < kRows; ++j)
                result.U(i, j) = V(i, j);
        for (int i = 0; i < kCols; ++i)
            for (int j = 0; j < kCols; ++j)
                result.V(i, j) = U(i, j);
    }
    else
    {
        for (int i = 0; i < kRows; ++i)
            for (int j = 0; j < kRows; ++j)
                result.U(i, j) = U(i, j);
        for (int i = 0; i < kCols; ++i)
            for (int j = 0; j < kCols; ++j)
                result.V(i, j) = V(i, j);
    }
    result.S = S;

    return result;
}

/**
 * @brief Compute only singular values of a general MxN matrix using the Jacobi method.
 *
 * This is a simplified version of JacobiSVD that only computes singular values
 * without the singular vectors, which is faster when only the singular values are needed.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Input MxN matrix
 * @param bSortSingularValues If true, singular values are sorted in descending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @param maxSweeps Maximum number of Jacobi sweeps (default: 20)
 * @return Vector of singular values
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto
JacobiSingularValues(TMatrix&& A, bool bSortSingularValues = true, int maxSweeps = 20)
{
    using namespace std;

    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);

    using ScalarType             = typename MatrixType::ScalarType;
    static auto constexpr kRows  = MatrixType::kRows;
    static auto constexpr kCols  = MatrixType::kCols;
    static auto constexpr kRank  = (kRows < kCols) ? kRows : kCols;
    static auto constexpr kTrans = (kRows < kCols);

    // Working dimensions
    static auto constexpr kWorkRows = kTrans ? kCols : kRows;
    static auto constexpr kWorkCols = kTrans ? kRows : kCols;

    // Working copy of the matrix (possibly transposed)
    SMatrix<ScalarType, kWorkRows, kWorkCols> B;
    if constexpr (kTrans)
    {
        for (int i = 0; i < kRows; ++i)
            for (int j = 0; j < kCols; ++j)
                B(j, i) = A(i, j);
    }
    else
    {
        B = A;
    }

    // Convergence tolerance
    ScalarType const eps =
        ScalarType(kWorkRows * kWorkCols) * std::numeric_limits<ScalarType>::epsilon();

    // Jacobi sweeps (without accumulating V)
    for (int sweep = 0; sweep < maxSweeps; ++sweep)
    {
        ScalarType maxOffDiag = ScalarType{0};

        for (int i = 0; i < kWorkCols - 1; ++i)
        {
            for (int j = i + 1; j < kWorkCols; ++j)
            {
                ScalarType aii = ScalarType{0};
                ScalarType aij = ScalarType{0};
                ScalarType ajj = ScalarType{0};

                for (int k = 0; k < kWorkRows; ++k)
                {
                    aii += B(k, i) * B(k, i);
                    aij += B(k, i) * B(k, j);
                    ajj += B(k, j) * B(k, j);
                }

                ScalarType absAij = fabs(aij);
                if (absAij > maxOffDiag)
                    maxOffDiag = absAij;

                if (absAij < eps * (sqrt(aii * ajj) + ScalarType{1}))
                    continue;

                SVector<ScalarType, 2> cosSinTheta = JacobiRotation(aii, aij, ajj);
                ScalarType cosTheta                = cosSinTheta(0);
                ScalarType sinTheta                = cosSinTheta(1);

                for (int k = 0; k < kWorkRows; ++k)
                {
                    ScalarType bi = B(k, i);
                    ScalarType bj = B(k, j);
                    B(k, i)       = cosTheta * bi - sinTheta * bj;
                    B(k, j)       = sinTheta * bi + cosTheta * bj;
                }
            }
        }

        if (maxOffDiag < eps)
            break;
    }

    // Extract singular values (column norms)
    SVector<ScalarType, kRank> S;
    for (int j = 0; j < kRank; ++j)
    {
        ScalarType normSq = ScalarType{0};
        for (int i = 0; i < kWorkRows; ++i)
            normSq += B(i, j) * B(i, j);
        S(j) = sqrt(normSq);
    }

    // Sort in descending order if requested
    if (bSortSingularValues)
    {
        for (int i = 0; i < kRank - 1; ++i)
        {
            int maxIdx = i;
            for (int j = i + 1; j < kRank; ++j)
            {
                if (S(j) > S(maxIdx))
                    maxIdx = j;
            }
            if (maxIdx != i)
            {
                ScalarType tmp = S(i);
                S(i)           = S(maxIdx);
                S(maxIdx)      = tmp;
            }
        }
    }

    return S;
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_SVD_H
