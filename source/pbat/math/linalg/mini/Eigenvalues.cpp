#include "Eigenvalues.h"

#include "BinaryOperations.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Transpose.h"
#include "pbat/Aliases.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] SymmetricEigen2x2")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Diagonal matrix")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 3.0;
        A(0, 1) = 0.0;
        A(1, 0) = 0.0;
        A(1, 1) = 1.0;

        auto [eigenvalues, eigenvectors] = SymmetricEigen2x2(A);

        // Eigenvalues in ascending order
        CHECK_EQ(eigenvalues(0), doctest::Approx(1.0).epsilon(1e-12));
        CHECK_EQ(eigenvalues(1), doctest::Approx(3.0).epsilon(1e-12));

        // Eigenvectors are orthonormal
        SMatrix<ScalarType, 2, 2> VtV = eigenvectors.Transpose() * eigenvectors;
        ScalarType orthogonality      = SquaredNorm(VtV - Identity<ScalarType, 2, 2>());
        CHECK_LE(orthogonality, 1e-12);
    }

    SUBCASE("Symmetric matrix with off-diagonal elements")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 2.0;
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        A(1, 1) = 2.0;

        auto [eigenvalues, eigenvectors] = SymmetricEigen2x2(A);

        // Eigenvalues should be 1 and 3
        CHECK_EQ(eigenvalues(0), doctest::Approx(1.0).epsilon(1e-12));
        CHECK_EQ(eigenvalues(1), doctest::Approx(3.0).epsilon(1e-12));

        // Check eigenvalue equation: A * v = λ * v
        for (int k = 0; k < 2; ++k)
        {
            SVector<ScalarType, 2> v;
            v(0) = eigenvectors(0, k);
            v(1) = eigenvectors(1, k);

            SVector<ScalarType, 2> Av      = A * v;
            SVector<ScalarType, 2> lambdaV = eigenvalues(k) * v;
            ScalarType eigenError          = SquaredNorm(Av - lambdaV);
            CHECK_LE(eigenError, 1e-12);
        }
    }

    SUBCASE("Compare with Eigen")
    {
        SMatrix<ScalarType, 2, 2> Amini;
        Amini(0, 0) = 4.0;
        Amini(0, 1) = 2.0;
        Amini(1, 0) = 2.0;
        Amini(1, 1) = 5.0;

        pbat::Matrix<2, 2> Aeigen;
        Aeigen << 4.0, 2.0, 2.0, 5.0;

        auto [eigMini, vecMini] = SymmetricEigen2x2(Amini);

        Eigen::SelfAdjointEigenSolver<pbat::Matrix<2, 2>> solver(Aeigen);
        auto eigEigen = solver.eigenvalues();

        CHECK_EQ(eigMini(0), doctest::Approx(eigEigen(0)).epsilon(1e-12));
        CHECK_EQ(eigMini(1), doctest::Approx(eigEigen(1)).epsilon(1e-12));
    }
}

TEST_CASE("[math][linalg][mini] SymmetricEigen3x3")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Diagonal matrix")
    {
        SMatrix<ScalarType, 3, 3> A = Zeros<ScalarType, 3, 3>();
        A(0, 0)                     = 5.0;
        A(1, 1)                     = 2.0;
        A(2, 2)                     = 8.0;

        auto [lambda, V] = SymmetricEigen3x3(A);

        // Eigenvalues in ascending order
        CHECK_EQ(lambda(0), doctest::Approx(2.0).epsilon(1e-10));
        CHECK_EQ(lambda(1), doctest::Approx(5.0).epsilon(1e-10));
        CHECK_EQ(lambda(2), doctest::Approx(8.0).epsilon(1e-10));

        // Eigenvectors are orthonormal
        SMatrix<ScalarType, 3, 3> VtV = V.Transpose() * V;
        pbat::math::linalg::mini::Identity<ScalarType, 3, 3> I{};
        ScalarType orthogonality = SquaredNorm(VtV - I);
        CHECK_LE(orthogonality, 1e-10);
    }

    SUBCASE("Symmetric matrix")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 0.0;
        A(1, 0) = 2.0;
        A(1, 1) = 5.0;
        A(1, 2) = 3.0;
        A(2, 0) = 0.0;
        A(2, 1) = 3.0;
        A(2, 2) = 4.0;

        auto [lambda, V] = SymmetricEigen3x3(A);

        // Check eigenvalue equation: A * v = λ * v
        for (int k = 0; k < 3; ++k)
        {
            SVector<ScalarType, 3> v       = V.Col(k);
            SVector<ScalarType, 3> Av      = A * v;
            SVector<ScalarType, 3> lambdaV = lambda(k) * v;
            ScalarType error               = SquaredNorm(Av - lambdaV);
            CHECK_LE(error, 1e-10);
        }

        // Eigenvectors are orthonormal
        SMatrix<ScalarType, 3, 3> VtV = V.Transpose() * V;
        pbat::math::linalg::mini::Identity<ScalarType, 3, 3> I{};
        ScalarType orthogonality = SquaredNorm(VtV - I);
        CHECK_LE(orthogonality, 1e-10);
    }

    SUBCASE("Compare with Eigen")
    {
        SMatrix<ScalarType, 3, 3> Amini;
        Amini(0, 0) = 6.0;
        Amini(0, 1) = 2.0;
        Amini(0, 2) = 1.0;
        Amini(1, 0) = 2.0;
        Amini(1, 1) = 3.0;
        Amini(1, 2) = 1.0;
        Amini(2, 0) = 1.0;
        Amini(2, 1) = 1.0;
        Amini(2, 2) = 1.0;

        pbat::Matrix<3, 3> Aeigen;
        Aeigen << 6.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0;

        auto [eigMini, vecMini] = SymmetricEigen3x3(Amini);

        Eigen::SelfAdjointEigenSolver<pbat::Matrix<3, 3>> solver(Aeigen);
        auto eigEigen = solver.eigenvalues();

        // Eigenvalues should match (Eigen also returns in ascending order)
        CHECK_EQ(eigMini(0), doctest::Approx(eigEigen(0)).epsilon(1e-10));
        CHECK_EQ(eigMini(1), doctest::Approx(eigEigen(1)).epsilon(1e-10));
        CHECK_EQ(eigMini(2), doctest::Approx(eigEigen(2)).epsilon(1e-10));
    }

    SUBCASE("Near-identity matrix")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 1.0 + 1e-8;
        A(0, 1) = 1e-10;
        A(0, 2) = 1e-10;
        A(1, 0) = 1e-10;
        A(1, 1) = 1.0;
        A(1, 2) = 1e-10;
        A(2, 0) = 1e-10;
        A(2, 1) = 1e-10;
        A(2, 2) = 1.0 - 1e-8;

        auto [eigenvalues, eigenvectors] = SymmetricEigen3x3(A);

        // All eigenvalues should be approximately 1
        CHECK_EQ(eigenvalues(0), doctest::Approx(1.0).epsilon(1e-6));
        CHECK_EQ(eigenvalues(1), doctest::Approx(1.0).epsilon(1e-6));
        CHECK_EQ(eigenvalues(2), doctest::Approx(1.0).epsilon(1e-6));
    }

    SUBCASE("Repeated eigenvalues")
    {
        // 2*I has triple eigenvalue 2
        SMatrix<ScalarType, 3, 3> A = Zeros<ScalarType, 3, 3>();
        A(0, 0)                     = 2.0;
        A(1, 1)                     = 2.0;
        A(2, 2)                     = 2.0;

        auto [eigenvalues, eigenvectors] = SymmetricEigen3x3(A);

        CHECK_EQ(eigenvalues(0), doctest::Approx(2.0).epsilon(1e-10));
        CHECK_EQ(eigenvalues(1), doctest::Approx(2.0).epsilon(1e-10));
        CHECK_EQ(eigenvalues(2), doctest::Approx(2.0).epsilon(1e-10));

        // Eigenvectors should still be orthonormal
        SMatrix<ScalarType, 3, 3> VtV = eigenvectors.Transpose() * eigenvectors;
        pbat::math::linalg::mini::Identity<ScalarType, 3, 3> I{};
        ScalarType orthogonality = SquaredNorm(VtV - I);
        CHECK_LE(orthogonality, 1e-10);
    }
}

TEST_CASE("[math][linalg][mini] SymmetricEigenvalues")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("2x2 eigenvalues only")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 5.0;
        A(0, 1) = 2.0;
        A(1, 0) = 2.0;
        A(1, 1) = 3.0;

        SVector<ScalarType, 2> eigenvalues = SymmetricEigenvalues(A);

        pbat::Matrix<2, 2> Aeigen;
        Aeigen << 5.0, 2.0, 2.0, 3.0;
        Eigen::SelfAdjointEigenSolver<pbat::Matrix<2, 2>> solver(Aeigen);
        auto eigEigen = solver.eigenvalues();

        CHECK_EQ(eigenvalues(0), doctest::Approx(eigEigen(0)).epsilon(1e-12));
        CHECK_EQ(eigenvalues(1), doctest::Approx(eigEigen(1)).epsilon(1e-12));
    }

    SUBCASE("3x3 eigenvalues only")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 1.0;
        A(0, 1) = 0.5;
        A(0, 2) = 0.2;
        A(1, 0) = 0.5;
        A(1, 1) = 2.0;
        A(1, 2) = 0.3;
        A(2, 0) = 0.2;
        A(2, 1) = 0.3;
        A(2, 2) = 3.0;

        SVector<ScalarType, 3> eigenvalues = SymmetricEigenvalues(A);

        pbat::Matrix<3, 3> Aeigen;
        Aeigen << 1.0, 0.5, 0.2, 0.5, 2.0, 0.3, 0.2, 0.3, 3.0;
        Eigen::SelfAdjointEigenSolver<pbat::Matrix<3, 3>> solver(Aeigen);
        auto eigEigen = solver.eigenvalues();

        CHECK_EQ(eigenvalues(0), doctest::Approx(eigEigen(0)).epsilon(1e-10));
        CHECK_EQ(eigenvalues(1), doctest::Approx(eigEigen(1)).epsilon(1e-10));
        CHECK_EQ(eigenvalues(2), doctest::Approx(eigEigen(2)).epsilon(1e-10));
    }
}
