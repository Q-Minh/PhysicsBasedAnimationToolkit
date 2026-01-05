#include "QR.h"

#include "BinaryOperations.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Transpose.h"
#include "pbat/Aliases.h"

#include <Eigen/QR>
#include <cmath>
#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] QR")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("2x2 QR decomposition")
    {
        SMatrix<ScalarType, 2, 2> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(1, 0) = 3.0;
        A(1, 1) = 4.0;

        auto [Q, R] = QR(A);

        // Check that Q is orthogonal: Q^T * Q = I
        SMatrix<ScalarType, 2, 2> QtQ = Q.Transpose() * Q;
        ScalarType orthogonality      = (SquaredNorm(QtQ - Identity<ScalarType, 2, 2>()));
        CHECK_LE(orthogonality, 1e-10);

        // Check that R is upper triangular
        CHECK_EQ(R(1, 0), doctest::Approx(0.0).epsilon(1e-10));

        // Check that Q * R = A
        SMatrix<ScalarType, 2, 2> QR_reconstructed = Q * R;
        ScalarType reconstructionError             = SquaredNorm(QR_reconstructed - A);
        CHECK_LE(reconstructionError, 1e-10);
    }

    SUBCASE("3x3 QR decomposition")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 12.0;
        A(0, 1) = -51.0;
        A(0, 2) = 4.0;
        A(1, 0) = 6.0;
        A(1, 1) = 167.0;
        A(1, 2) = -68.0;
        A(2, 0) = -4.0;
        A(2, 1) = 24.0;
        A(2, 2) = -41.0;

        auto [Q, R] = QR(A);

        // Check that Q is orthogonal: Q^T * Q = I
        SMatrix<ScalarType, 3, 3> QtQ = Q.Transpose() * Q;
        ScalarType orthogonality      = SquaredNorm(QtQ - Identity<ScalarType, 3, 3>());
        CHECK_LE(orthogonality, 1e-10);

        // Check that R is upper triangular
        CHECK_EQ(R(1, 0), doctest::Approx(0.0).epsilon(1e-10));
        CHECK_EQ(R(2, 0), doctest::Approx(0.0).epsilon(1e-10));
        CHECK_EQ(R(2, 1), doctest::Approx(0.0).epsilon(1e-10));

        // Check that Q * R = A
        SMatrix<ScalarType, 3, 3> QR_reconstructed = Q * R;
        ScalarType reconstructionError             = SquaredNorm(QR_reconstructed - A);
        CHECK_LE(reconstructionError, 1e-10);
    }

    SUBCASE("3x2 thin QR decomposition")
    {
        SMatrix<ScalarType, 3, 2> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(1, 0) = 3.0;
        A(1, 1) = 4.0;
        A(2, 0) = 5.0;
        A(2, 1) = 6.0;

        auto [Q, R] = QR(A);

        // Check that Q has orthonormal columns: Q^T * Q = I (2x2)
        SMatrix<ScalarType, 2, 2> QtQ = Q.Transpose() * Q;
        ScalarType orthogonality      = SquaredNorm(QtQ - Identity<ScalarType, 2, 2>());
        CHECK_LE(orthogonality, 1e-10);

        // Check that R is upper triangular
        CHECK_EQ(R(1, 0), doctest::Approx(0.0).epsilon(1e-10));

        // Check that Q * R = A
        SMatrix<ScalarType, 3, 2> QR_reconstructed = Q * R;
        ScalarType reconstructionError             = SquaredNorm(QR_reconstructed - A);
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
        Amini(2, 2) = 10.0;

        pbat::Matrix<3, 3> Aeigen;
        Aeigen << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0;

        auto [Qmini, Rmini] = QR(Amini);
        Eigen::HouseholderQR<pbat::Matrix<3, 3>> qr(Aeigen);
        pbat::Matrix<3, 3> Qeigen = qr.householderQ();
        pbat::Matrix<3, 3> Reigen = qr.matrixQR().triangularView<Eigen::Upper>();

        // Reconstruct A from both
        SMatrix<ScalarType, 3, 3> Amini_reconstructed = Qmini * Rmini;
        pbat::Matrix<3, 3> Aeigen_reconstructed       = Qeigen * Reigen;

        // Both reconstructions should match the original
        ScalarType reconstructionError = SquaredNorm(Amini_reconstructed - Amini);
        CHECK_LE(reconstructionError, 1e-10);
    }
}

TEST_CASE("[math][linalg][mini] Givens rotation")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Standard case")
    {
        ScalarType a              = 3.0;
        ScalarType b              = 4.0;
        SVector<ScalarType, 2> cs = GivensRotation(a, b);
        ScalarType c              = cs(0);
        ScalarType s              = cs(1);

        // Check that rotation zeroes b
        ScalarType r    = c * a + s * b;
        ScalarType zero = -s * a + c * b;
        CHECK_EQ(zero, doctest::Approx(0.0).epsilon(1e-14));
        CHECK_EQ(r, doctest::Approx(5.0).epsilon(1e-14)); // sqrt(9 + 16) = 5

        // Check that rotation is orthogonal
        CHECK_EQ(c * c + s * s, doctest::Approx(1.0).epsilon(1e-14));
    }

    SUBCASE("b is zero")
    {
        ScalarType a              = 5.0;
        ScalarType b              = 0.0;
        SVector<ScalarType, 2> cs = GivensRotation(a, b);
        ScalarType c              = cs(0);
        ScalarType s              = cs(1);

        CHECK_EQ(c, doctest::Approx(1.0).epsilon(1e-14));
        CHECK_EQ(s, doctest::Approx(0.0).epsilon(1e-14));
    }

    SUBCASE("a is zero")
    {
        ScalarType a              = 0.0;
        ScalarType b              = 5.0;
        SVector<ScalarType, 2> cs = GivensRotation(a, b);
        ScalarType c              = cs(0);
        ScalarType s              = cs(1);

        // Rotation should produce r = |b|
        ScalarType r    = c * a + s * b;
        ScalarType zero = -s * a + c * b;
        CHECK_EQ(zero, doctest::Approx(0.0).epsilon(1e-14));
        CHECK_EQ(std::abs(r), doctest::Approx(5.0).epsilon(1e-14));
    }
}
