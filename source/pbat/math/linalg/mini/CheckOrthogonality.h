#ifndef PBAT_MATH_LINALG_MINI_TESTORTHOGONALITY_H
#define PBAT_MATH_LINALG_MINI_TESTORTHOGONALITY_H

#include "Concepts.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Transpose.h"

#include <doctest/doctest.h>

namespace pbat::math::linalg::mini::test {

/**
 * @brief Check that a square matrix is orthonormal (M^T * M = I).
 *
 * @tparam TMatrix Matrix type satisfying CMatrix concept
 * @param M The matrix to check for orthonormality
 * @param tol Tolerance for the squared Frobenius norm of (M^T * M - I)
 */
template <class /*CMatrix*/ TMatrix>
void CheckOrthonormality(TMatrix const& M, typename std::remove_cvref_t<TMatrix>::ScalarType tol)
{
    using MatrixType                = std::remove_cvref_t<TMatrix>;
    using ScalarType                = typename MatrixType::ScalarType;
    static auto constexpr kRows     = MatrixType::kRows;
    static auto constexpr kCols     = MatrixType::kCols;
    SMatrix<ScalarType, kCols, kCols> MtM = M.Transpose() * M;
    Identity<ScalarType, kCols, kCols> I{};
    ScalarType orthogonalityError = SquaredNorm(MtM - I);
    CHECK_LE(orthogonalityError, tol);
}

} // namespace pbat::math::linalg::mini::test

#endif // PBAT_MATH_LINALG_MINI_TESTORTHOGONALITY_H
