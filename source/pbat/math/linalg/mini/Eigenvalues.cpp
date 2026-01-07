#include "Eigenvalues.h"

#include "BinaryOperations.h"
#include "CheckOrthogonality.h"
#include "Eigen.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Transpose.h"
#include "pbat/Aliases.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <doctest/doctest.h>

namespace pbat::math::linalg::mini::test {

/**
 * @brief Check symmetric eigendecomposition reconstruction error: ||V * diag(lambda) * V^T - A||^2.
 *
 * @tparam TMatrixV Eigenvector matrix type
 * @tparam TVectorL Eigenvalue vector type
 * @tparam TMatrixA Original matrix type
 * @param V Eigenvector matrix (columns are eigenvectors)
 * @param lambda Eigenvalues
 * @param A Original symmetric matrix
 * @param tol Tolerance for the squared Frobenius norm of the reconstruction error
 */
template <class /*CMatrix*/ TMatrixV, class /*CMatrix*/ TVectorL, class /*CMatrix*/ TMatrixA>
void CheckEigenReconstruction(
    TMatrixV const& V,
    TVectorL const& lambda,
    TMatrixA const& A,
    typename std::decay_t<TMatrixA>::ScalarType tol)
{
    using ScalarType            = typename std::decay_t<TMatrixA>::ScalarType;
    static auto constexpr kDims = std::decay_t<TVectorL>::kRows;
    ScalarType nA               = Norm(A);

    // Build diagonal matrix from eigenvalues
    SMatrix<ScalarType, kDims, kDims> D = Zeros<ScalarType, kDims, kDims>();
    for (int i = 0; i < kDims; ++i)
        D(i, i) = lambda(i);

    SMatrix<ScalarType, kDims, kDims> reconstructed = V * D * V.Transpose();
    ScalarType reconstructionError                  = Norm(reconstructed - A);
    CHECK_LE(reconstructionError, nA * tol);
}

/**
 * @brief Check that eigenvalue equation holds: A * v = lambda * v for each eigenpair.
 *
 * @tparam TMatrixA Original matrix type
 * @tparam TMatrixV Eigenvector matrix type
 * @tparam TVectorL Eigenvalue vector type
 * @param A Original matrix
 * @param V Eigenvector matrix (columns are eigenvectors)
 * @param lambda Eigenvalues
 * @param tol Tolerance for the squared norm of (A*v - lambda*v) for each eigenpair
 */
template <class /*CMatrix*/ TMatrixA, class /*CMatrix*/ TMatrixV, class /*CMatrix*/ TVectorL>
void CheckEigenEquation(
    TMatrixA const& A,
    TMatrixV const& V,
    TVectorL const& lambda,
    typename std::decay_t<TMatrixA>::ScalarType tol)
{
    using ScalarType            = typename std::decay_t<TMatrixA>::ScalarType;
    static auto constexpr kDims = std::decay_t<TVectorL>::kRows;
    ScalarType nA               = Norm(A);
    for (int k = 0; k < kDims; ++k)
    {
        auto v                             = V.Col(k);
        SVector<ScalarType, kDims> Av      = A * v;
        SVector<ScalarType, kDims> lambdaV = lambda(k) * v;
        ScalarType error                   = Norm(Av - lambdaV);
        CHECK_LE(error, nA * tol);
    }
}

} // namespace pbat::math::linalg::mini::test

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
        test::CheckOrthonormality(eigenvectors, ScalarType{1e-12});
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
        test::CheckEigenEquation(A, eigenvectors, eigenvalues, ScalarType{1e-12});
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

    SUBCASE("Zero matrix")
    {
        SMatrix<ScalarType, 2, 2> A = Zeros<ScalarType, 2, 2>();

        auto [eigenvalues, eigenvectors] = SymmetricEigen2x2(A);

        // All eigenvalues should be zero
        CHECK_EQ(eigenvalues(0), doctest::Approx(0.0).epsilon(1e-12));
        CHECK_EQ(eigenvalues(1), doctest::Approx(0.0).epsilon(1e-12));

        // Eigenvectors should be orthonormal (identity matrix is a valid choice)
        test::CheckOrthonormality(eigenvectors, ScalarType{1e-12});
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
        test::CheckOrthonormality(V, ScalarType{1e-10});
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
        test::CheckEigenEquation(A, V, lambda, ScalarType{1e-5});

        // Eigenvectors are orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-6});
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
        CHECK_EQ(eigMini(0), doctest::Approx(eigEigen(0)).epsilon(1e-6));
        CHECK_EQ(eigMini(1), doctest::Approx(eigEigen(1)).epsilon(1e-6));
        CHECK_EQ(eigMini(2), doctest::Approx(eigEigen(2)).epsilon(1e-6));
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

        CHECK_EQ(eigenvalues(0), doctest::Approx(2.0).epsilon(1e-6));
        CHECK_EQ(eigenvalues(1), doctest::Approx(2.0).epsilon(1e-6));
        CHECK_EQ(eigenvalues(2), doctest::Approx(2.0).epsilon(1e-6));

        // Eigenvectors should still be orthonormal
        test::CheckOrthonormality(eigenvectors, ScalarType{1e-10});
    }

    SUBCASE("Zero matrix")
    {
        SMatrix<ScalarType, 3, 3> A = Zeros<ScalarType, 3, 3>();

        auto [eigenvalues, eigenvectors] = SymmetricEigen3x3(A);

        // All eigenvalues should be zero
        CHECK_EQ(eigenvalues(0), doctest::Approx(0.0).epsilon(1e-10));
        CHECK_EQ(eigenvalues(1), doctest::Approx(0.0).epsilon(1e-10));
        CHECK_EQ(eigenvalues(2), doctest::Approx(0.0).epsilon(1e-10));

        // Eigenvectors should be orthonormal
        test::CheckOrthonormality(eigenvectors, ScalarType{1e-10});

        // Reconstruction should be zero
        test::CheckEigenReconstruction(eigenvectors, eigenvalues, A, ScalarType{1e-10});
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

        CHECK_EQ(eigenvalues(0), doctest::Approx(eigEigen(0)).epsilon(1e-6));
        CHECK_EQ(eigenvalues(1), doctest::Approx(eigEigen(1)).epsilon(1e-6));
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

        CHECK_EQ(eigenvalues(0), doctest::Approx(eigEigen(0)).epsilon(1e-6));
        CHECK_EQ(eigenvalues(1), doctest::Approx(eigEigen(1)).epsilon(1e-6));
        CHECK_EQ(eigenvalues(2), doctest::Approx(eigEigen(2)).epsilon(1e-6));
    }
}

TEST_CASE("[math][linalg][mini] SymmetricEigenNxN")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("4x4 diagonal matrix")
    {
        SMatrix<ScalarType, 4, 4> A = Zeros<ScalarType, 4, 4>();
        A(0, 0)                     = 7.0;
        A(1, 1)                     = 2.0;
        A(2, 2)                     = 9.0;
        A(3, 3)                     = 4.0;

        auto [lambda, V] = SymmetricEigenNxN(A);

        // Eigenvalues in ascending order: 2, 4, 7, 9
        CHECK_EQ(lambda(0), doctest::Approx(2.0).epsilon(1e-4));
        CHECK_EQ(lambda(1), doctest::Approx(4.0).epsilon(1e-4));
        CHECK_EQ(lambda(2), doctest::Approx(7.0).epsilon(1e-4));
        CHECK_EQ(lambda(3), doctest::Approx(9.0).epsilon(1e-4));

        // Eigenvectors are orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-7});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(V, lambda, A, ScalarType{1e-4});
    }

    SUBCASE("4x4 symmetric matrix")
    {
        SMatrix<ScalarType, 4, 4> Amini;
        // Create a symmetric positive definite matrix
        Amini(0, 0) = 5.0;
        Amini(0, 1) = 1.0;
        Amini(0, 2) = 0.5;
        Amini(0, 3) = 0.2;
        Amini(1, 0) = 1.0;
        Amini(1, 1) = 4.0;
        Amini(1, 2) = 1.0;
        Amini(1, 3) = 0.3;
        Amini(2, 0) = 0.5;
        Amini(2, 1) = 1.0;
        Amini(2, 2) = 6.0;
        Amini(2, 3) = 0.8;
        Amini(3, 0) = 0.2;
        Amini(3, 1) = 0.3;
        Amini(3, 2) = 0.8;
        Amini(3, 3) = 3.0;

        auto [lambda, V] = SymmetricEigenNxN(Amini);

        // Check eigenvalue equation: A * v = λ * v
        test::CheckEigenEquation(Amini, V, lambda, ScalarType{1e-4});

        // Eigenvectors are orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-6});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(V, lambda, Amini, ScalarType{1e-4});
    }

    SUBCASE("5x5 zero matrix")
    {
        SMatrix<ScalarType, 5, 5> A = Zeros<ScalarType, 5, 5>();

        auto [lambda, V] = SymmetricEigenNxN(A);

        // All eigenvalues should be zero
        for (int i = 0; i < 5; ++i)
            CHECK_EQ(lambda(i), doctest::Approx(0.0).epsilon(1e-6));

        // Eigenvectors should be orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-7});

        // Reconstruction should be zero
        test::CheckEigenReconstruction(V, lambda, A, ScalarType{1e-4});
    }

    SUBCASE("4x4 compare with Eigen")
    {
        SMatrix<ScalarType, 4, 4> Amini;
        Amini(0, 0) = 4.0;
        Amini(0, 1) = 1.0;
        Amini(0, 2) = 0.0;
        Amini(0, 3) = 0.5;
        Amini(1, 0) = 1.0;
        Amini(1, 1) = 3.0;
        Amini(1, 2) = 1.0;
        Amini(1, 3) = 0.0;
        Amini(2, 0) = 0.0;
        Amini(2, 1) = 1.0;
        Amini(2, 2) = 2.0;
        Amini(2, 3) = 1.0;
        Amini(3, 0) = 0.5;
        Amini(3, 1) = 0.0;
        Amini(3, 2) = 1.0;
        Amini(3, 3) = 5.0;

        pbat::Matrix<4, 4> Aeigen;
        Aeigen << 4.0, 1.0, 0.0, 0.5, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.5, 0.0, 1.0, 5.0;

        auto [eigMini, vecMini] = SymmetricEigenNxN(Amini);

        Eigen::SelfAdjointEigenSolver<pbat::Matrix<4, 4>> solver(Aeigen);
        auto eigEigen = solver.eigenvalues();

        // Eigenvalues should match (Eigen also returns in ascending order)
        for (int i = 0; i < 4; ++i)
        {
            CHECK_EQ(eigMini(i), doctest::Approx(eigEigen(i)).epsilon(1e-5));
        }

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(vecMini, eigMini, Amini, ScalarType{1e-4});
    }

    SUBCASE("5x5 symmetric matrix")
    {
        SMatrix<ScalarType, 5, 5> Amini;
        // Initialize as symmetric positive definite
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                Amini(i, j) = (i == j) ? ScalarType(5 + i) : ScalarType(1) / ScalarType(1 + i + j);
            }
        }
        // Ensure symmetry
        for (int i = 0; i < 5; ++i)
        {
            for (int j = i + 1; j < 5; ++j)
            {
                Amini(j, i) = Amini(i, j);
            }
        }

        auto [lambda, V] = SymmetricEigenNxN(Amini);

        // Check eigenvalue equation: A * v = λ * v
        test::CheckEigenEquation(Amini, V, lambda, ScalarType{1e-4});

        // Eigenvectors are orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-6});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(V, lambda, Amini, ScalarType{1e-3});
    }

    SUBCASE("6x6 compare with Eigen")
    {
        SMatrix<ScalarType, 6, 6> Amini;
        // Build a symmetric matrix with various eigenvalue magnitudes
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                ScalarType val =
                    (i == j) ? ScalarType(i + 1) * ScalarType(2) : ScalarType(0.1) * (i + j);
                Amini(i, j) = val;
                Amini(j, i) = val;
            }
        }

        pbat::Matrix<6, 6> Aeigen;
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                Aeigen(i, j) = Amini(i, j);
            }
        }

        auto [eigMini, vecMini] = SymmetricEigenNxN(Amini);

        Eigen::SelfAdjointEigenSolver<pbat::Matrix<6, 6>> solver(Aeigen);
        auto eigEigen = solver.eigenvalues();

        for (int i = 0; i < 6; ++i)
        {
            CHECK_EQ(eigMini(i), doctest::Approx(eigEigen(i)).epsilon(1e-5));
        }

        // Check eigenvalue equation
        test::CheckEigenEquation(Amini, vecMini, eigMini, ScalarType{1e-4});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(vecMini, eigMini, Amini, ScalarType{1e-3});
    }

    SUBCASE("Repeated eigenvalues 4x4")
    {
        // Matrix with eigenvalue 3 repeated twice
        SMatrix<ScalarType, 4, 4> A = Zeros<ScalarType, 4, 4>();
        A(0, 0)                     = 3.0;
        A(1, 1)                     = 3.0;
        A(2, 2)                     = 5.0;
        A(3, 3)                     = 7.0;

        auto [lambda, V] = SymmetricEigenNxN(A);

        CHECK_EQ(lambda(0), doctest::Approx(3.0).epsilon(1e-10));
        CHECK_EQ(lambda(1), doctest::Approx(3.0).epsilon(1e-10));
        CHECK_EQ(lambda(2), doctest::Approx(5.0).epsilon(1e-10));
        CHECK_EQ(lambda(3), doctest::Approx(7.0).epsilon(1e-10));

        // Eigenvectors should be orthonormal even with repeated eigenvalues
        test::CheckOrthonormality(V, ScalarType{1e-6});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(V, lambda, A, ScalarType{1e-4});
    }

    SUBCASE("Near-identity 4x4")
    {
        SMatrix<ScalarType, 4, 4> A;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                A(i, j) =
                    (i == j) ? ScalarType(1.0) + ScalarType(1e-8) * (i - 1.5) : ScalarType(1e-10);
            }
        }
        // Enforce symmetry
        for (int i = 0; i < 4; ++i)
        {
            for (int j = i + 1; j < 4; ++j)
            {
                A(j, i) = A(i, j);
            }
        }

        auto [lambda, V] = SymmetricEigenNxN(A);

        // All eigenvalues approximately 1
        for (int i = 0; i < 4; ++i)
        {
            CHECK_EQ(lambda(i), doctest::Approx(1.0).epsilon(1e-6));
        }

        // Eigenvectors should be orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-9});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(V, lambda, A, ScalarType{1e-9});
    }

    SUBCASE("Unsorted eigenvalues")
    {
        SMatrix<ScalarType, 4, 4> A = Zeros<ScalarType, 4, 4>();
        A(0, 0)                     = 7.0;
        A(1, 1)                     = 2.0;
        A(2, 2)                     = 9.0;
        A(3, 3)                     = 4.0;

        // Request unsorted eigenvalues
        auto [lambda, V] = SymmetricEigenNxN(A, false);

        // Check eigenvalue equation (order doesn't matter)
        test::CheckEigenEquation(A, V, lambda, ScalarType{1e-10});

        // Eigenvectors are orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-10});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(V, lambda, A, ScalarType{1e-10});
    }

    SUBCASE("9x9 symmetric matrix")
    {
        Eigen::Matrix<ScalarType, 9, 9> B = Eigen::Matrix<ScalarType, 9, 9>::Random();
        SMatrix<ScalarType, 9, 9> A       = Zeros<ScalarType, 9, 9>();
        A = ScalarType(1e6) * (FromEigen(B) + FromEigen(B).Transpose()) * ScalarType(0.5);
        auto [lambda, V] = SymmetricEigenNxN(A);

        // Check eigenvalue equation
        test::CheckEigenEquation(A, V, lambda, ScalarType{1e-4});

        // Eigenvectors are orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-6});

        // Reconstruction: A = V * diag(lambda) * V^T
        test::CheckEigenReconstruction(V, lambda, A, ScalarType{1e-4});
    }
}

TEST_CASE("[math][linalg][mini] WilkinsonShift")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Diagonal 2x2 block")
    {
        // For a diagonal block [3, 0; 0, 5], shift should be 5 (closer to c=5)
        ScalarType shift = WilkinsonShift(ScalarType{3}, ScalarType{0}, ScalarType{5});
        CHECK_EQ(shift, doctest::Approx(5.0).epsilon(1e-12));
    }

    SUBCASE("Symmetric 2x2 block")
    {
        // Block [2, 1; 1, 2] has eigenvalues 1 and 3
        // c = 2, so shift should be the eigenvalue closer to 2, which is either 1 or 3
        // Actually both are equidistant, but the formula should return one of them
        ScalarType shift = WilkinsonShift(ScalarType{2}, ScalarType{1}, ScalarType{2});
        // Eigenvalues are 1 and 3, both at distance 1 from c=2
        bool isEigenvalue = (std::abs(shift - 1.0) < 1e-10) || (std::abs(shift - 3.0) < 1e-10);
        CHECK(isEigenvalue);
    }

    SUBCASE("Asymmetric diagonal values")
    {
        // Block [1, 2; 2, 5] has eigenvalues approximately 0.17 and 5.83
        // c = 5, so shift should be closer to 5.83
        ScalarType shift = WilkinsonShift(ScalarType{1}, ScalarType{2}, ScalarType{5});
        // The eigenvalue closer to c=5 is 5.83...
        CHECK_EQ(shift, doctest::Approx(5.828427).epsilon(1e-5));
    }
}
