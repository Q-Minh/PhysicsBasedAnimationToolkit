#include "SVD.h"

#include "BinaryOperations.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Scale.h"
#include "Transpose.h"
#include "pbat/Aliases.h"

#include <Eigen/SVD>
#include <cmath>
#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] SVD2x2")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Simple matrix")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 3.0;
        A(0, 1) = 0.0;
        A(1, 0) = 0.0;
        A(1, 1) = 2.0;

        auto [U, S, V] = SVD2x2(A);

        // Singular values should be 3 and 2 in descending order
        CHECK_EQ(S(0), doctest::Approx(3.0).epsilon(1e-12));
        CHECK_EQ(S(1), doctest::Approx(2.0).epsilon(1e-12));

        // Check U and V are orthogonal
        SMatrix<ScalarType, 2, 2> UtU = U.Transpose() * U;
        SMatrix<ScalarType, 2, 2> VtV = V.Transpose() * V;
        ScalarType orthogonalityU     = SquaredNorm(UtU - Identity<ScalarType, 2, 2>());
        ScalarType orthogonalityV     = SquaredNorm(VtV - Identity<ScalarType, 2, 2>());
        CHECK_LE(orthogonalityU, 1e-12);
        CHECK_LE(orthogonalityV, 1e-12);

        // Check reconstruction: A = U * diag(S) * V^T
        SMatrix<ScalarType, 2, 2> Sigma = Zeros<ScalarType, 2, 2>();
        Sigma(0, 0)                     = S(0);
        Sigma(1, 1)                     = S(1);

        SMatrix<ScalarType, 2, 2> reconstructed = U * Sigma * V.Transpose();
        ScalarType reconstructionError          = SquaredNorm(reconstructed - A);
        CHECK_LE(reconstructionError, 1e-12);
    }

    SUBCASE("General matrix")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(1, 0) = 3.0;
        A(1, 1) = 4.0;

        auto [U, S, V] = SVD2x2(A);

        // Singular values should be non-negative and descending
        CHECK(S(0) >= S(1));
        CHECK(S(1) >= 0.0);

        // Check reconstruction
        SMatrix<ScalarType, 2, 2> Sigma = Zeros<ScalarType, 2, 2>();
        Sigma(0, 0)                     = S(0);
        Sigma(1, 1)                     = S(1);

        SMatrix<ScalarType, 2, 2> reconstructed = U * Sigma * V.Transpose();
        ScalarType reconstructionError          = SquaredNorm(reconstructed - A);
        CHECK_LE(reconstructionError, 1e-10);
    }

    SUBCASE("Compare with Eigen")
    {
        SMatrix<ScalarType, 2, 2> Amini;
        Amini(0, 0) = 5.0;
        Amini(0, 1) = 7.0;
        Amini(1, 0) = 2.0;
        Amini(1, 1) = 9.0;

        pbat::Matrix<2, 2> Aeigen;
        Aeigen << 5.0, 7.0, 2.0, 9.0;

        auto [U, S, V] = SVD2x2(Amini);

        Eigen::JacobiSVD<pbat::Matrix<2, 2>> svd(Aeigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto Seigen = svd.singularValues();

        // Singular values should match
        CHECK_EQ(S(0), doctest::Approx(Seigen(0)).epsilon(1e-5));
        CHECK_EQ(S(1), doctest::Approx(Seigen(1)).epsilon(1e-5));
    }

    SUBCASE("Rank-deficient matrix")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(1, 0) = 2.0;
        A(1, 1) = 4.0; // Rank 1 matrix

        auto [U, S, V] = SVD2x2(A);

        // Second singular value should be ~0
        CHECK_EQ(S(1), doctest::Approx(0.0).epsilon(1e-10));

        // U and V should still be orthogonal
        SMatrix<ScalarType, 2, 2> UtU = U.Transpose() * U;
        SMatrix<ScalarType, 2, 2> VtV = V.Transpose() * V;
        ScalarType orthogonalityU     = SquaredNorm(UtU - Identity<ScalarType, 2, 2>());
        ScalarType orthogonalityV     = SquaredNorm(VtV - Identity<ScalarType, 2, 2>());
        CHECK_LE(orthogonalityU, 1e-10);
        CHECK_LE(orthogonalityV, 1e-10);
    }
}

TEST_CASE("[math][linalg][mini] SVD3x3")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Diagonal matrix")
    {
        SMatrix<ScalarType, 3, 3> A = Zeros<ScalarType, 3, 3>();
        A(0, 0)                     = 5.0;
        A(1, 1)                     = 3.0;
        A(2, 2)                     = 1.0;

        auto [U, S, V] = SVD3x3(A);

        // Singular values in descending order
        CHECK_EQ(S(0), doctest::Approx(5.0).epsilon(1e-10));
        CHECK_EQ(S(1), doctest::Approx(3.0).epsilon(1e-10));
        CHECK_EQ(S(2), doctest::Approx(1.0).epsilon(1e-10));

        // Check orthogonality
        SMatrix<ScalarType, 3, 3> UtU = U.Transpose() * U;
        SMatrix<ScalarType, 3, 3> VtV = V.Transpose() * V;
        ScalarType orthogonalityU     = SquaredNorm(UtU - Identity<ScalarType, 3, 3>());
        ScalarType orthogonalityV     = SquaredNorm(VtV - Identity<ScalarType, 3, 3>());
        CHECK_LE(orthogonalityU, 1e-10);
        CHECK_LE(orthogonalityV, 1e-10);
    }

    SUBCASE("General matrix")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(1, 0) = 4.0;
        A(1, 1) = 5.0;
        A(1, 2) = 6.0;
        A(2, 0) = 7.0;
        A(2, 1) = 8.0;
        A(2, 2) = 10.0;

        auto [U, S, V] = SVD3x3(A);

        // Singular values should be non-negative and descending
        CHECK(S(0) >= S(1));
        CHECK(S(1) >= S(2));
        CHECK(S(2) >= 0.0);

        // Check reconstruction
        SMatrix<ScalarType, 3, 3> Sigma = Zeros<ScalarType, 3, 3>();
        Sigma(0, 0)                     = S(0);
        Sigma(1, 1)                     = S(1);
        Sigma(2, 2)                     = S(2);

        SMatrix<ScalarType, 3, 3> reconstructed = U * Sigma * V.Transpose();
        ScalarType reconstructionError          = SquaredNorm(reconstructed - A);
        CHECK_LE(reconstructionError, 1e-10);
    }

    SUBCASE("Compare with Eigen")
    {
        SMatrix<ScalarType, 3, 3> Amini;
        Amini(0, 0) = 1.0;
        Amini(0, 1) = 2.0;
        Amini(0, 2) = 3.0;
        Amini(1, 0) = 4.0;
        Amini(1, 1) = 5.0;
        Amini(1, 2) = 6.0;
        Amini(2, 0) = 7.0;
        Amini(2, 1) = 8.0;
        Amini(2, 2) = 9.0;

        pbat::Matrix<3, 3> Aeigen;
        Aeigen << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;

        auto [U, S, V] = SVD3x3(Amini);

        Eigen::JacobiSVD<pbat::Matrix<3, 3>> svd(Aeigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto Seigen = svd.singularValues();

        // Singular values should match
        CHECK_EQ(S(0), doctest::Approx(Seigen(0)).epsilon(1e-10));
        CHECK_EQ(S(1), doctest::Approx(Seigen(1)).epsilon(1e-10));
        CHECK_EQ(S(2), doctest::Approx(Seigen(2)).epsilon(1e-10));
    }

    SUBCASE("Rotation matrix")
    {
        // Rotation matrix has all singular values = 1
        ScalarType const theta      = 0.5;
        SMatrix<ScalarType, 3, 3> R = Zeros<ScalarType, 3, 3>();
        R(0, 0)                     = std::cos(theta);
        R(0, 1)                     = -std::sin(theta);
        R(1, 0)                     = std::sin(theta);
        R(1, 1)                     = std::cos(theta);
        R(2, 2)                     = 1.0;

        auto [U, S, V] = SVD3x3(R);

        CHECK_EQ(S(0), doctest::Approx(1.0).epsilon(1e-10));
        CHECK_EQ(S(1), doctest::Approx(1.0).epsilon(1e-10));
        CHECK_EQ(S(2), doctest::Approx(1.0).epsilon(1e-10));
    }
}

TEST_CASE("[math][linalg][mini] SingularValues")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("2x2 singular values only")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 4.0;
        A(0, 1) = 3.0;
        A(1, 0) = 3.0;
        A(1, 1) = 4.0;

        SVector<ScalarType, 2> S = SingularValues(A);

        pbat::Matrix<2, 2> Aeigen;
        Aeigen << 4.0, 3.0, 3.0, 4.0;
        Eigen::JacobiSVD<pbat::Matrix<2, 2>> svd(Aeigen);
        auto Seigen = svd.singularValues();
        CHECK_EQ(S(0), doctest::Approx(Seigen(0)).epsilon(1e-5));
        CHECK_EQ(S(1), doctest::Approx(Seigen(1)).epsilon(1e-5));
    }

    SUBCASE("3x3 singular values only")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 1.0;
        A(0, 1) = 0.0;
        A(0, 2) = 0.0;
        A(1, 0) = 0.0;
        A(1, 1) = 2.0;
        A(1, 2) = 0.0;
        A(2, 0) = 0.0;
        A(2, 1) = 0.0;
        A(2, 2) = 3.0;

        SVector<ScalarType, 3> S = SingularValues(A);

        // Descending order
        CHECK_EQ(S(0), doctest::Approx(3.0).epsilon(1e-12));
        CHECK_EQ(S(1), doctest::Approx(2.0).epsilon(1e-12));
        CHECK_EQ(S(2), doctest::Approx(1.0).epsilon(1e-12));
    }
}
