#include "LLT.h"

#include "BinaryOperations.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Transpose.h"
#include "pbat/Aliases.h"

#include <Eigen/Cholesky>
#include <doctest/doctest.h>

namespace pbat::math::linalg::mini::test {

/**
 * @brief Check LLT reconstruction error: ||L * L^T - A||.
 */
template <class /*CMatrix*/ TMatrixL, class /*CMatrix*/ TMatrixA>
void CheckLLTReconstruction(
    TMatrixL const& L,
    TMatrixA const& A,
    typename std::remove_cvref_t<TMatrixA>::ScalarType tol)
{
    using ScalarType               = typename std::remove_cvref_t<TMatrixA>::ScalarType;
    ScalarType nA                  = Norm(A);
    auto LLt                       = L * L.Transpose();
    ScalarType reconstructionError = Norm(LLt - A);
    CHECK_LE(reconstructionError, nA * tol);
}

/**
 * @brief Check that L is lower triangular.
 */
template <class /*CMatrix*/ TMatrixL>
void CheckLowerTriangular(TMatrixL const& L, typename std::remove_cvref_t<TMatrixL>::ScalarType tol)
{
    static auto constexpr kN = std::remove_cvref_t<TMatrixL>::kRows;
    for (auto j = 0; j < kN; ++j)
        for (auto i = 0; i < j; ++i)
            CHECK_LE(std::abs(L(i, j)), tol);
}

} // namespace pbat::math::linalg::mini::test

TEST_CASE("[math][linalg][mini] LLT")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("2x2 SPD matrix")
    {
        // A = [4 2; 2 5] is SPD
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 4.0;
        A(0, 1) = 2.0;
        A(1, 0) = 2.0;
        A(1, 1) = 5.0;

        auto [L, success] = LLT(A);

        CHECK(success);
        test::CheckLowerTriangular(L, ScalarType{1e-14});
        test::CheckLLTReconstruction(L, A, ScalarType{1e-7});
    }

    SUBCASE("3x3 SPD matrix")
    {
        // Build A = M^T * M + I for guaranteed SPD
        SMatrix<ScalarType, 3, 3> M;
        M(0, 0) = 1.0;
        M(0, 1) = 2.0;
        M(0, 2) = 0.0;
        M(1, 0) = 0.0;
        M(1, 1) = 1.0;
        M(1, 2) = 3.0;
        M(2, 0) = 2.0;
        M(2, 1) = 0.0;
        M(2, 2) = 1.0;
        Identity<ScalarType, 3, 3> I{};
        SMatrix<ScalarType, 3, 3> A = M.Transpose() * M + I;

        auto [L, success] = LLT(A);

        CHECK(success);
        test::CheckLowerTriangular(L, ScalarType{1e-14});
        test::CheckLLTReconstruction(L, A, ScalarType{1e-7});
    }

    SUBCASE("1x1 matrix")
    {
        SMatrix<ScalarType, 1, 1> A;
        A(0, 0) = 9.0;

        auto [L, success] = LLT(A);

        CHECK(success);
        CHECK_EQ(L(0, 0), doctest::Approx(3.0).epsilon(1e-14));
    }

    SUBCASE("Non-SPD matrix fails")
    {
        // A = [1 0; 0 -1] is not positive-definite
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 1.0;
        A(0, 1) = 0.0;
        A(1, 0) = 0.0;
        A(1, 1) = -1.0;

        auto [L, success] = LLT(A);

        CHECK_FALSE(success);
    }

    SUBCASE("LLTSolve")
    {
        // A = [4 2; 2 5], b = [8, 9]
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 4.0;
        A(0, 1) = 2.0;
        A(1, 0) = 2.0;
        A(1, 1) = 5.0;

        SVector<ScalarType, 2> b{8.0, 9.0};

        auto [L, success] = LLT(A);
        CHECK(success);

        auto x = LLTSolve(L, b);

        // Check A * x = b
        SVector<ScalarType, 2> Ax = A * x;
        ScalarType error          = Norm(Ax - b);
        CHECK_LE(error, ScalarType{1e-7} * Norm(A));
    }

    SUBCASE("LLTSolve with multiple RHS")
    {
        SMatrix<ScalarType, 3, 3> M;
        M(0, 0) = 2.0;
        M(0, 1) = 1.0;
        M(0, 2) = 0.0;
        M(1, 0) = 0.0;
        M(1, 1) = 3.0;
        M(1, 2) = 1.0;
        M(2, 0) = 1.0;
        M(2, 1) = 0.0;
        M(2, 2) = 2.0;
        Identity<ScalarType, 3, 3> I{};
        SMatrix<ScalarType, 3, 3> A = M.Transpose() * M + I;

        SMatrix<ScalarType, 3, 2> B;
        B(0, 0) = 1.0;
        B(0, 1) = 4.0;
        B(1, 0) = 2.0;
        B(1, 1) = 5.0;
        B(2, 0) = 3.0;
        B(2, 1) = 6.0;

        auto [L, success] = LLT(A);
        CHECK(success);

        auto X = LLTSolve(L, B);

        // Check A * X = B
        SMatrix<ScalarType, 3, 2> AX = A * X;
        ScalarType error             = Norm(AX - B);
        CHECK_LE(error, ScalarType{1e-7} * Norm(A));
    }

    SUBCASE("Compare with Eigen")
    {
        SMatrix<ScalarType, 3, 3> Mmini;
        Mmini(0, 0) = 1.0;
        Mmini(0, 1) = 2.0;
        Mmini(0, 2) = 3.0;
        Mmini(1, 0) = 0.0;
        Mmini(1, 1) = 4.0;
        Mmini(1, 2) = 5.0;
        Mmini(2, 0) = 0.0;
        Mmini(2, 1) = 0.0;
        Mmini(2, 2) = 6.0;
        Identity<ScalarType, 3, 3> I{};
        SMatrix<ScalarType, 3, 3> Amini = Mmini.Transpose() * Mmini + I;

        pbat::Matrix<3, 3> Aeigen;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                Aeigen(i, j) = Amini(i, j);

        auto [Lmini, success] = LLT(Amini);
        CHECK(success);

        Eigen::LLT<pbat::Matrix<3, 3>> llt(Aeigen);
        pbat::Matrix<3, 3> Leigen = llt.matrixL();

        // Both L factors should reconstruct A
        test::CheckLLTReconstruction(Lmini, Amini, ScalarType{1e-7});

        // Compare L factors directly
        for (int j = 0; j < 3; ++j)
            for (int i = j; i < 3; ++i)
                CHECK_EQ(Lmini(i, j), doctest::Approx(Leigen(i, j)).epsilon(1e-7));
    }
}
