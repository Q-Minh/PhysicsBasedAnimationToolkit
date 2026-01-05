#ifndef PBAT_MATH_LINALG_MINI_EIGENVALUES_H
#define PBAT_MATH_LINALG_MINI_EIGENVALUES_H

#include "Concepts.h"
#include "Matrix.h"
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
 * @brief Result of eigenvalue decomposition for symmetric matrices
 * @tparam TScalar Scalar type
 * @tparam N Matrix dimension
 */
template <class TScalar, int N>
struct SymmetricEigenResult
{
    SVector<TScalar, N> lambda; ///< Eigenvalues (ascending order if sorted)
    SMatrix<TScalar, N, N> V;   ///< Eigenvectors as columns
};

/**
 * @brief Compute eigenvalues and eigenvectors of a 2x2 symmetric matrix analytically.
 *
 * For a 2x2 symmetric matrix:
 * [a  b]
 * [b  c]
 *
 * The eigenvalues are computed using the quadratic formula with numerical
 * robustness improvements to avoid catastrophic cancellation.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A `2 x 2` symmetric matrix
 * @param bSortEigenvalues If true, eigenvalues are sorted in ascending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return SymmetricEigenResult with eigenvalues and orthonormal eigenvectors
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SymmetricEigen2x2(TMatrix&& A, bool bSortEigenvalues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 2 and MatrixType::kCols == 2, "Matrix must be 2x2");

    using ScalarType = typename MatrixType::ScalarType;

    SymmetricEigenResult<ScalarType, 2> result{};

    ScalarType const a = A(0, 0);
    ScalarType const b = A(0, 1); // = A(1,0) for symmetric
    ScalarType const c = A(1, 1);

    // Trace and determinant
    ScalarType const trace = a + c;
    ScalarType const det   = a * c - b * b;

    // Discriminant: trace^2 - 4*det = (a-c)^2 + 4*b^2 >= 0
    ScalarType const diff  = a - c;
    ScalarType const discr = diff * diff + ScalarType{4} * b * b;

    ScalarType sqrtDiscr;
    if constexpr (std::is_same_v<ScalarType, float>)
        sqrtDiscr = sqrtf(discr);
    else
        sqrtDiscr = sqrt(discr);

    // Eigenvalues (ascending order)
    result.lambda(0) = ScalarType{0.5} * (trace - sqrtDiscr);
    result.lambda(1) = ScalarType{0.5} * (trace + sqrtDiscr);

    // Eigenvectors
    // For numerical stability, we compute the eigenvector for the eigenvalue
    // that's furthest from a (or c), then use orthogonality for the other.
    ScalarType const eps = ScalarType{4} * std::numeric_limits<ScalarType>::epsilon(); // 2x2 matrix

    ScalarType absB;
    if constexpr (std::is_same_v<ScalarType, float>)
        absB = fabsf(b);
    else
        absB = fabs(b);

    if (absB < eps * (ScalarType{1} + (a > c ? a : c)))
    {
        // Matrix is essentially diagonal
        if (a <= c)
        {
            result.V(0, 0) = ScalarType{1};
            result.V(1, 0) = ScalarType{0};
            result.V(0, 1) = ScalarType{0};
            result.V(1, 1) = ScalarType{1};
        }
        else
        {
            result.V(0, 0) = ScalarType{0};
            result.V(1, 0) = ScalarType{1};
            result.V(0, 1) = ScalarType{1};
            result.V(1, 1) = ScalarType{0};
        }
    }
    else
    {
        // Standard case: compute eigenvector from (A - λI)v = 0
        // For λ1 (smaller), eigenvector is proportional to (b, λ1 - a) or (λ1 - c, b)
        // Use the formulation that avoids subtracting similar numbers

        ScalarType v0x, v0y, v1x, v1y;

        // For first eigenvalue
        ScalarType const lambda0_minus_a = result.lambda(0) - a;
        ScalarType const lambda0_minus_c = result.lambda(0) - c;

        ScalarType abs_lma, abs_lmc;
        if constexpr (std::is_same_v<ScalarType, float>)
        {
            abs_lma = fabsf(lambda0_minus_a);
            abs_lmc = fabsf(lambda0_minus_c);
        }
        else
        {
            abs_lma = fabs(lambda0_minus_a);
            abs_lmc = fabs(lambda0_minus_c);
        }

        if (abs_lma > abs_lmc)
        {
            // Use (λ - c, b)
            v0x = lambda0_minus_c;
            v0y = b;
        }
        else
        {
            // Use (b, λ - a)
            v0x = b;
            v0y = lambda0_minus_a;
        }

        // Normalize first eigenvector
        ScalarType norm0sq = v0x * v0x + v0y * v0y;
        ScalarType invNorm0;
        if constexpr (std::is_same_v<ScalarType, float>)
            invNorm0 = ScalarType{1} / sqrtf(norm0sq);
        else
            invNorm0 = ScalarType{1} / sqrt(norm0sq);

        v0x *= invNorm0;
        v0y *= invNorm0;

        // Second eigenvector is orthogonal to first
        v1x = -v0y;
        v1y = v0x;

        result.V(0, 0) = v0x;
        result.V(1, 0) = v0y;
        result.V(0, 1) = v1x;
        result.V(1, 1) = v1y;
    }

    return result;
}

/**
 * @brief Compute eigenvalues and eigenvectors of a 3x3 symmetric matrix analytically.
 *
 * Uses Cardano's formula for the cubic characteristic polynomial with
 * numerical robustness improvements. This is an O(1) algorithm with no
 * iteration, making it deterministic and suitable for GPU execution.
 *
 * See @cite kopp2008symmetric3x3eigen
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A `3 x 3` symmetric matrix
 * @param bSortEigenvalues If true, eigenvalues are sorted in ascending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return SymmetricEigenResult with eigenvalues and orthonormal eigenvectors
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SymmetricEigen3x3(TMatrix&& A, bool bSortEigenvalues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 3 and MatrixType::kCols == 3, "Matrix must be 3x3");

    using ScalarType = typename MatrixType::ScalarType;

    SymmetricEigenResult<ScalarType, 3> result{};

    // Extract elements (symmetric, so only upper triangle needed)
    ScalarType const a11 = A(0, 0);
    ScalarType const a12 = A(0, 1);
    ScalarType const a13 = A(0, 2);
    ScalarType const a22 = A(1, 1);
    ScalarType const a23 = A(1, 2);
    ScalarType const a33 = A(2, 2);

    ScalarType const eps = ScalarType{9} * std::numeric_limits<ScalarType>::epsilon(); // 3x3 matrix

    // Compute characteristic polynomial coefficients
    // det(A - λI) = -λ³ + c2*λ² + c1*λ + c0 = 0
    // where c2 = trace(A), c1 = sum of 2x2 principal minors, c0 = det(A)

    ScalarType const trace = a11 + a22 + a33;
    ScalarType const mean  = trace / ScalarType{3};

    // Shift matrix by mean to improve numerical stability (Kopp's method)
    ScalarType const b11 = a11 - mean;
    ScalarType const b22 = a22 - mean;
    ScalarType const b33 = a33 - mean;

    // q = det(B) / 2, p = ||B||_F^2 / 6 where B = A - mean*I
    ScalarType const p =
        (b11 * b11 + b22 * b22 + b33 * b33 + ScalarType{2} * (a12 * a12 + a13 * a13 + a23 * a23)) /
        ScalarType{6};

    // Determinant of shifted matrix B
    ScalarType const detB = b11 * (b22 * b33 - a23 * a23) - a12 * (a12 * b33 - a23 * a13) +
                            a13 * (a12 * a23 - b22 * a13);

    ScalarType const q = detB / ScalarType{2};

    // For a symmetric matrix, p >= 0 and the discriminant p^3 - q^2 >= 0
    ScalarType p_cubed = p * p * p;
    ScalarType q_sq    = q * q;

    // Clamp the ratio for numerical stability
    ScalarType ratio;
    if (p_cubed < eps)
    {
        // Matrix is essentially a multiple of identity
        result.lambda(0) = mean;
        result.lambda(1) = mean;
        result.lambda(2) = mean;

        result.V(0, 0) = ScalarType{1};
        result.V(1, 0) = ScalarType{0};
        result.V(2, 0) = ScalarType{0};
        result.V(0, 1) = ScalarType{0};
        result.V(1, 1) = ScalarType{1};
        result.V(2, 1) = ScalarType{0};
        result.V(0, 2) = ScalarType{0};
        result.V(1, 2) = ScalarType{0};
        result.V(2, 2) = ScalarType{1};

        return result;
    }

    // Compute sqrt(p) first - we need it for the ratio calculation
    ScalarType sqrtP;
    if constexpr (std::is_same_v<ScalarType, float>)
        sqrtP = sqrtf(p);
    else
        sqrtP = sqrt(p);

    // ratio = q / p^(3/2) = q / (p * sqrt(p))
    ratio = q / (p * sqrtP);

    // Clamp to [-1, 1] for acos (numerical robustness)
    if constexpr (std::is_same_v<ScalarType, float>)
        ratio = fminf(fmaxf(ratio, ScalarType{-1}), ScalarType{1});
    else
        ratio = fmin(fmax(ratio, ScalarType{-1}), ScalarType{1});

    // Eigenvalues from Cardano's formula
    ScalarType phi;
    if constexpr (std::is_same_v<ScalarType, float>)
        phi = acosf(ratio) / ScalarType{3};
    else
        phi = acos(ratio) / ScalarType{3};

    ScalarType const twosqrtP = ScalarType{2} * sqrtP;

    // Eigenvalues in descending order from Cardano
    ScalarType cos_phi, cos_phi_2pi3, cos_phi_4pi3;
    ScalarType const pi     = ScalarType{3.14159265358979323846};
    ScalarType const twopi3 = ScalarType{2} * pi / ScalarType{3};

    if constexpr (std::is_same_v<ScalarType, float>)
    {
        cos_phi      = cosf(phi);
        cos_phi_2pi3 = cosf(phi + twopi3);
        cos_phi_4pi3 = cosf(phi + ScalarType{2} * twopi3);
    }
    else
    {
        cos_phi      = cos(phi);
        cos_phi_2pi3 = cos(phi + twopi3);
        cos_phi_4pi3 = cos(phi + ScalarType{2} * twopi3);
    }

    // Eigenvalues (will sort to ascending order if requested)
    ScalarType eig0 = mean + twosqrtP * cos_phi;
    ScalarType eig1 = mean + twosqrtP * cos_phi_2pi3;
    ScalarType eig2 = mean + twosqrtP * cos_phi_4pi3;

    // Sort eigenvalues in ascending order using sorting network
    // (branchless when compiler optimizes min/max)
    if (bSortEigenvalues)
    {
        ScalarType t;
        if (eig0 > eig1)
        {
            t    = eig0;
            eig0 = eig1;
            eig1 = t;
        }
        if (eig1 > eig2)
        {
            t    = eig1;
            eig1 = eig2;
            eig2 = t;
        }
        if (eig0 > eig1)
        {
            t    = eig0;
            eig0 = eig1;
            eig1 = t;
        }
    }

    result.lambda(0) = eig0;
    result.lambda(1) = eig1;
    result.lambda(2) = eig2;

    // Compute eigenvectors using cross products for robustness
    // For each eigenvalue λ, find eigenvector from null space of (A - λI)
    for (int k = 0; k < 3; ++k)
    {
        ScalarType const lambda = result.lambda(k);

        // Rows of (A - λI)
        ScalarType r0x = a11 - lambda, r0y = a12, r0z = a13;
        ScalarType r1x = a12, r1y = a22 - lambda, r1z = a23;
        ScalarType r2x = a13, r2y = a23, r2z = a33 - lambda;

        // Cross products of rows to find null space direction
        ScalarType c01x = r0y * r1z - r0z * r1y;
        ScalarType c01y = r0z * r1x - r0x * r1z;
        ScalarType c01z = r0x * r1y - r0y * r1x;

        ScalarType c02x = r0y * r2z - r0z * r2y;
        ScalarType c02y = r0z * r2x - r0x * r2z;
        ScalarType c02z = r0x * r2y - r0y * r2x;

        ScalarType c12x = r1y * r2z - r1z * r2y;
        ScalarType c12y = r1z * r2x - r1x * r2z;
        ScalarType c12z = r1x * r2y - r1y * r2x;

        // Pick the cross product with largest magnitude
        ScalarType n01 = c01x * c01x + c01y * c01y + c01z * c01z;
        ScalarType n02 = c02x * c02x + c02y * c02y + c02z * c02z;
        ScalarType n12 = c12x * c12x + c12y * c12y + c12z * c12z;

        ScalarType vx, vy, vz, normSq;
        if (n01 >= n02 and n01 >= n12)
        {
            vx     = c01x;
            vy     = c01y;
            vz     = c01z;
            normSq = n01;
        }
        else if (n02 >= n12)
        {
            vx     = c02x;
            vy     = c02y;
            vz     = c02z;
            normSq = n02;
        }
        else
        {
            vx     = c12x;
            vy     = c12y;
            vz     = c12z;
            normSq = n12;
        }

        // Normalize
        ScalarType invNorm;
        if (normSq > eps * eps)
        {
            if constexpr (std::is_same_v<ScalarType, float>)
                invNorm = ScalarType{1} / sqrtf(normSq);
            else
                invNorm = ScalarType{1} / sqrt(normSq);
        }
        else
        {
            // Fallback for degenerate case (repeated eigenvalue)
            // Use a unit vector orthogonal to previous eigenvectors
            if (k == 0)
            {
                vx      = ScalarType{1};
                vy      = ScalarType{0};
                vz      = ScalarType{0};
                invNorm = ScalarType{1};
            }
            else if (k == 1)
            {
                // Orthogonal to first eigenvector
                ScalarType const v0x = result.V(0, 0);
                ScalarType const v0y = result.V(1, 0);
                ScalarType const v0z = result.V(2, 0);

                // Pick a non-parallel axis
                ScalarType absv0x, absv0y;
                if constexpr (std::is_same_v<ScalarType, float>)
                {
                    absv0x = fabsf(v0x);
                    absv0y = fabsf(v0y);
                }
                else
                {
                    absv0x = fabs(v0x);
                    absv0y = fabs(v0y);
                }

                if (absv0x < absv0y)
                {
                    // Cross with x-axis
                    vx = ScalarType{0};
                    vy = -v0z;
                    vz = v0y;
                }
                else
                {
                    // Cross with y-axis
                    vx = v0z;
                    vy = ScalarType{0};
                    vz = -v0x;
                }
                normSq = vx * vx + vy * vy + vz * vz;
                if constexpr (std::is_same_v<ScalarType, float>)
                    invNorm = ScalarType{1} / sqrtf(normSq);
                else
                    invNorm = ScalarType{1} / sqrt(normSq);
            }
            else
            {
                // Cross product of first two eigenvectors
                ScalarType const v0x = result.V(0, 0);
                ScalarType const v0y = result.V(1, 0);
                ScalarType const v0z = result.V(2, 0);
                ScalarType const v1x = result.V(0, 1);
                ScalarType const v1y = result.V(1, 1);
                ScalarType const v1z = result.V(2, 1);

                vx      = v0y * v1z - v0z * v1y;
                vy      = v0z * v1x - v0x * v1z;
                vz      = v0x * v1y - v0y * v1x;
                invNorm = ScalarType{1}; // Already normalized if v0, v1 are orthonormal
            }
        }

        result.V(0, k) = vx * invNorm;
        result.V(1, k) = vy * invNorm;
        result.V(2, k) = vz * invNorm;
    }

    return result;
}

/**
 * @brief Compute eigenvalue decomposition of a symmetric matrix.
 *
 * Dispatcher that calls the appropriate 2x2 or 3x3 analytic solver.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Symmetric square matrix (2x2 or 3x3)
 * @param bSortEigenvalues If true, eigenvalues are sorted in ascending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return SymmetricEigenResult with eigenvalues and orthonormal eigenvectors
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SymmetricEigen(TMatrix&& A, bool bSortEigenvalues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == MatrixType::kCols, "Matrix must be square");
    static_assert(
        MatrixType::kRows == 2 or MatrixType::kRows == 3,
        "Only 2x2 and 3x3 matrices supported");

    if constexpr (MatrixType::kRows == 2)
        return SymmetricEigen2x2(std::forward<TMatrix>(A), bSortEigenvalues);
    else
        return SymmetricEigen3x3(std::forward<TMatrix>(A), bSortEigenvalues);
}

/**
 * @brief Compute only the eigenvalues of a 2x2 symmetric matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Symmetric 2x2 matrix
 * @param bSortEigenvalues If true, eigenvalues are sorted in ascending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return Vector of 2 eigenvalues (ascending order if bSortEigenvalues is true)
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SymmetricEigenvalues2x2(TMatrix&& A, bool bSortEigenvalues = true)
    -> SVector<typename std::remove_cvref_t<TMatrix>::ScalarType, 2>
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == 2 and MatrixType::kCols == 2, "Matrix must be 2x2");

    using ScalarType = typename MatrixType::ScalarType;

    ScalarType const a = A(0, 0);
    ScalarType const b = A(0, 1);
    ScalarType const c = A(1, 1);

    ScalarType const trace = a + c;
    ScalarType const diff  = a - c;
    ScalarType const discr = diff * diff + ScalarType{4} * b * b;

    ScalarType sqrtDiscr;
    if constexpr (std::is_same_v<ScalarType, float>)
        sqrtDiscr = sqrtf(discr);
    else
        sqrtDiscr = sqrt(discr);

    SVector<ScalarType, 2> eigenvalues;
    // Compute in ascending order by default (trace - sqrt <= trace + sqrt)
    eigenvalues(0) = ScalarType{0.5} * (trace - sqrtDiscr);
    eigenvalues(1) = ScalarType{0.5} * (trace + sqrtDiscr);

    // Note: The formula inherently produces ascending order, so bSortEigenvalues
    // doesn't change behavior here but is kept for API consistency
    (void)bSortEigenvalues;

    return eigenvalues;
}

/**
 * @brief Compute only the eigenvalues of a 3x3 symmetric matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Symmetric 3x3 matrix
 * @param bSortEigenvalues If true, eigenvalues are sorted in ascending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return Vector of 3 eigenvalues (ascending order if bSortEigenvalues is true)
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SymmetricEigenvalues3x3(TMatrix&& A, bool bSortEigenvalues = true)
{
    // Use full decomposition - eigenvalue-only version could be optimized
    // but the overhead of computing eigenvectors is small
    return SymmetricEigen3x3(std::forward<TMatrix>(A), bSortEigenvalues).lambda;
}

/**
 * @brief Compute only eigenvalues of a symmetric matrix.
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param A Symmetric square matrix (2x2 or 3x3)
 * @param bSortEigenvalues If true, eigenvalues are sorted in ascending order.
 *        Set to false to avoid unnecessary work when order doesn't matter.
 * @return Vector of eigenvalues (ascending order if bSortEigenvalues is true)
 */
template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SymmetricEigenvalues(TMatrix&& A, bool bSortEigenvalues = true)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    static_assert(MatrixType::kRows == MatrixType::kCols, "Matrix must be square");
    static_assert(
        MatrixType::kRows == 2 or MatrixType::kRows == 3,
        "Only 2x2 and 3x3 matrices supported");

    if constexpr (MatrixType::kRows == 2)
        return SymmetricEigenvalues2x2(std::forward<TMatrix>(A), bSortEigenvalues);
    else
        return SymmetricEigenvalues3x3(std::forward<TMatrix>(A), bSortEigenvalues);
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_EIGENVALUES_H
