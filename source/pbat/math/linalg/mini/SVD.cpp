#include "SVD.h"

#include "BinaryOperations.h"
#include "CheckOrthogonality.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "Scale.h"
#include "Transpose.h"
#include "pbat/Aliases.h"

#include <Eigen/SVD>
#include <cmath>
#include <doctest/doctest.h>

namespace pbat::math::linalg::mini::test {

/**
 * @brief Check SVD reconstruction error: ||U * diag(S) * V^T - A||^2.
 *
 * @tparam TMatrixU U matrix type
 * @tparam TVectorS Singular values vector type
 * @tparam TMatrixV V matrix type
 * @tparam TMatrixA Original matrix type
 * @param U Left singular vectors
 * @param S Singular values
 * @param V Right singular vectors
 * @param A Original matrix
 * @param tol Tolerance for the squared Frobenius norm of the reconstruction error
 */
template <
    class /*CMatrix*/ TMatrixU,
    class /*CMatrix*/ TVectorS,
    class /*CMatrix*/ TMatrixV,
    class /*CMatrix*/ TMatrixA>
void CheckSVDReconstruction(
    TMatrixU const& U,
    TVectorS const& S,
    TMatrixV const& V,
    TMatrixA const& A,
    typename std::remove_cvref_t<TMatrixA>::ScalarType tol)
{
    using ScalarType            = typename std::remove_cvref_t<TMatrixA>::ScalarType;
    static auto constexpr kDims = std::remove_cvref_t<TVectorS>::kRows;
    ScalarType nA               = Norm(A);
    // Build diagonal matrix from singular values
    SMatrix<ScalarType, kDims, kDims> Sigma = Zeros<ScalarType, kDims, kDims>();
    for (int i = 0; i < kDims; ++i)
        Sigma(i, i) = S(i);
    SMatrix<ScalarType, kDims, kDims> reconstructed = U * Sigma * V.Transpose();
    ScalarType reconstructionError                  = Norm(reconstructed - A);
    CHECK_LE(reconstructionError, nA * tol);
}

} // namespace pbat::math::linalg::mini::test

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
        test::CheckOrthonormality(U, ScalarType{1e-12});
        test::CheckOrthonormality(V, ScalarType{1e-12});

        // Check reconstruction: A = U * diag(S) * V^T
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});
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
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});
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

        // Check reconstruction
        test::CheckSVDReconstruction(U, S, V, Amini, ScalarType{1e-5});

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

        // Check reconstruction
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});

        // U and V should still be orthogonal
        test::CheckOrthonormality(U, ScalarType{1e-10});
        test::CheckOrthonormality(V, ScalarType{1e-10});
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
        CHECK_EQ(S(0), doctest::Approx(5.0).epsilon(1e-5));
        CHECK_EQ(S(1), doctest::Approx(3.0).epsilon(1e-5));
        CHECK_EQ(S(2), doctest::Approx(1.0).epsilon(1e-5));

        // Check orthogonality
        test::CheckOrthonormality(U, ScalarType{1e-10});
        test::CheckOrthonormality(V, ScalarType{1e-10});
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
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-3});

        // Check orthogonality
        test::CheckOrthonormality(U, ScalarType{1e-7});
        test::CheckOrthonormality(V, ScalarType{1e-7});
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
        CHECK_EQ(S(0), doctest::Approx(Seigen(0)).epsilon(1e-5));
        CHECK_EQ(S(1), doctest::Approx(Seigen(1)).epsilon(1e-5));
        CHECK_EQ(S(2), doctest::Approx(Seigen(2)).epsilon(1e-2));

        // Check orthogonality
        test::CheckOrthonormality(U, ScalarType{1e-9});
        test::CheckOrthonormality(V, ScalarType{1e-9});
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

        CHECK_EQ(S(0), doctest::Approx(1.0).epsilon(1e-5));
        CHECK_EQ(S(1), doctest::Approx(1.0).epsilon(1e-5));
        CHECK_EQ(S(2), doctest::Approx(1.0).epsilon(1e-5));

        // Check orthogonality
        test::CheckOrthonormality(U, ScalarType{1e-10});
        test::CheckOrthonormality(V, ScalarType{1e-10});
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
        CHECK_EQ(S(0), doctest::Approx(3.0).epsilon(1e-5));
        CHECK_EQ(S(1), doctest::Approx(2.0).epsilon(1e-5));
        CHECK_EQ(S(2), doctest::Approx(1.0).epsilon(1e-5));
    }
}

TEST_CASE("[math][linalg][mini] JacobiSVD")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("3x3 diagonal matrix")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 3.0;
        A(0, 1) = 0.0;
        A(0, 2) = 0.0;
        A(1, 0) = 0.0;
        A(1, 1) = 1.0;
        A(1, 2) = 0.0;
        A(2, 0) = 0.0;
        A(2, 1) = 0.0;
        A(2, 2) = 2.0;

        auto [U, S, V] = JacobiSVD(A);

        // Singular values in descending order: 3, 2, 1
        CHECK_EQ(S(0), doctest::Approx(3.0).epsilon(1e-10));
        CHECK_EQ(S(1), doctest::Approx(2.0).epsilon(1e-10));
        CHECK_EQ(S(2), doctest::Approx(1.0).epsilon(1e-10));

        // U and V orthonormal
        test::CheckOrthonormality(U, ScalarType{1e-10});
        test::CheckOrthonormality(V, ScalarType{1e-10});

        // Reconstruction
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});
    }

    SUBCASE("3x3 general matrix")
    {
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 0.0;
        A(1, 0) = 0.0;
        A(1, 1) = 3.0;
        A(1, 2) = 1.0;
        A(2, 0) = 2.0;
        A(2, 1) = 0.0;
        A(2, 2) = 4.0;

        auto [U, S, V] = JacobiSVD(A);

        // U and V orthonormal
        test::CheckOrthonormality(U, ScalarType{1e-10});
        test::CheckOrthonormality(V, ScalarType{1e-10});

        // Reconstruction
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});

        // Compare with Eigen
        pbat::Matrix<3, 3> Aeigen;
        Aeigen << 1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0;
        Eigen::JacobiSVD<pbat::Matrix<3, 3>> svd(Aeigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto Seigen = svd.singularValues();

        for (int i = 0; i < 3; ++i)
            CHECK_EQ(S(i), doctest::Approx(Seigen(i)).epsilon(1e-5));
    }

    SUBCASE("4x4 general matrix")
    {
        SMatrix<ScalarType, 4, 4> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(0, 3) = 0.5;
        A(1, 0) = 4.0;
        A(1, 1) = 5.0;
        A(1, 2) = 6.0;
        A(1, 3) = 1.5;
        A(2, 0) = 7.0;
        A(2, 1) = 8.0;
        A(2, 2) = 9.0;
        A(2, 3) = 2.5;
        A(3, 0) = 0.1;
        A(3, 1) = 0.2;
        A(3, 2) = 0.3;
        A(3, 3) = 10.0;

        auto [U, S, V] = JacobiSVD(A);

        // U and V orthonormal
        test::CheckOrthonormality(U, ScalarType{1e-9});
        test::CheckOrthonormality(V, ScalarType{1e-9});

        // Reconstruction
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});

        // Compare with Eigen
        pbat::Matrix<4, 4> Aeigen;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                Aeigen(i, j) = A(i, j);

        Eigen::JacobiSVD<pbat::Matrix<4, 4>> svd(Aeigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto Seigen = svd.singularValues();

        for (int i = 0; i < 4; ++i)
            CHECK_EQ(S(i), doctest::Approx(Seigen(i)).epsilon(1e-5));
    }

    SUBCASE("4x3 tall matrix (M > N)")
    {
        SMatrix<ScalarType, 4, 3> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(1, 0) = 4.0;
        A(1, 1) = 5.0;
        A(1, 2) = 6.0;
        A(2, 0) = 7.0;
        A(2, 1) = 8.0;
        A(2, 2) = 9.0;
        A(3, 0) = 10.0;
        A(3, 1) = 11.0;
        A(3, 2) = 12.0;

        auto [U, S, V] = JacobiSVD(A);

        // U is 4x4 orthonormal
        test::CheckOrthonormality(U, ScalarType{1e-9});
        // V is 3x3 orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-9});

        // Reconstruction: A = U[:, :3] * diag(S) * V^T
        SMatrix<ScalarType, 4, 3> Arecon;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                ScalarType sum = ScalarType{0};
                for (int k = 0; k < 3; ++k)
                    sum += U(i, k) * S(k) * V(j, k);
                Arecon(i, j) = sum;
            }
        }
        ScalarType reconError = Norm(A - Arecon);
        CHECK_LE(reconError, 1e-4);

        // Compare with Eigen
        pbat::Matrix<4, 3> Aeigen;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j)
                Aeigen(i, j) = A(i, j);

        Eigen::JacobiSVD<pbat::Matrix<4, 3>> svd(Aeigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto Seigen = svd.singularValues();

        for (int i = 0; i < 3; ++i)
            CHECK_EQ(S(i), doctest::Approx(Seigen(i)).epsilon(1e-5));
    }

    SUBCASE("3x4 wide matrix (M < N)")
    {
        SMatrix<ScalarType, 3, 4> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(0, 3) = 4.0;
        A(1, 0) = 5.0;
        A(1, 1) = 6.0;
        A(1, 2) = 7.0;
        A(1, 3) = 8.0;
        A(2, 0) = 9.0;
        A(2, 1) = 10.0;
        A(2, 2) = 11.0;
        A(2, 3) = 12.0;

        auto [U, S, V] = JacobiSVD(A);

        // U is 3x3 orthonormal
        test::CheckOrthonormality(U, ScalarType{1e-9});
        // V is 4x4 orthonormal
        test::CheckOrthonormality(V, ScalarType{1e-9});

        // Reconstruction: A = U * diag(S) * V[:, :3]^T
        SMatrix<ScalarType, 3, 4> Arecon;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                ScalarType sum = ScalarType{0};
                for (int k = 0; k < 3; ++k)
                    sum += U(i, k) * S(k) * V(j, k);
                Arecon(i, j) = sum;
            }
        }
        ScalarType reconError = Norm(A - Arecon);
        CHECK_LE(reconError, 1e-4);

        // Compare with Eigen
        pbat::Matrix<3, 4> Aeigen;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                Aeigen(i, j) = A(i, j);

        Eigen::JacobiSVD<pbat::Matrix<3, 4>> svd(Aeigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto Seigen = svd.singularValues();

        for (int i = 0; i < 3; ++i)
            CHECK_EQ(S(i), doctest::Approx(Seigen(i)).epsilon(1e-5));
    }

    SUBCASE("5x5 symmetric positive definite")
    {
        SMatrix<ScalarType, 5, 5> A;
        // Create SPD matrix: A = B * B^T + I
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                A(i, j) = ScalarType(1) / ScalarType(1 + i + j);
            }
        }
        // Make it SPD by adding diagonal
        for (int i = 0; i < 5; ++i)
            A(i, i) += ScalarType{5};

        auto [U, S, V] = JacobiSVD(A);

        test::CheckOrthonormality(U, ScalarType{1e-9});
        test::CheckOrthonormality(V, ScalarType{1e-9});
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});

        // For SPD matrix, all singular values should be positive
        for (int i = 0; i < 5; ++i)
            CHECK_GT(S(i), ScalarType{0});
    }

    SUBCASE("Rank-deficient matrix")
    {
        // 3x3 matrix with rank 2
        SMatrix<ScalarType, 3, 3> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(1, 0) = 2.0;
        A(1, 1) = 4.0;
        A(1, 2) = 6.0; // Row 1 = 2 * Row 0
        A(2, 0) = 1.0;
        A(2, 1) = 1.0;
        A(2, 2) = 1.0;

        auto [U, S, V] = JacobiSVD(A);

        test::CheckOrthonormality(U, ScalarType{1e-10});
        test::CheckOrthonormality(V, ScalarType{1e-10});
        test::CheckSVDReconstruction(U, S, V, A, ScalarType{1e-5});

        // Third singular value should be essentially zero
        CHECK_LT(S(2), 1e-6);
    }
}

TEST_CASE("[math][linalg][mini] JacobiSingularValues")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("4x4 compare with full SVD")
    {
        SMatrix<ScalarType, 4, 4> A;
        A(0, 0) = 1.0;
        A(0, 1) = 2.0;
        A(0, 2) = 3.0;
        A(0, 3) = 0.5;
        A(1, 0) = 4.0;
        A(1, 1) = 5.0;
        A(1, 2) = 6.0;
        A(1, 3) = 1.5;
        A(2, 0) = 7.0;
        A(2, 1) = 8.0;
        A(2, 2) = 9.0;
        A(2, 3) = 2.5;
        A(3, 0) = 0.1;
        A(3, 1) = 0.2;
        A(3, 2) = 0.3;
        A(3, 3) = 10.0;

        auto S_full = JacobiSVD(A).S;
        auto S_only = JacobiSingularValues(A);

        for (int i = 0; i < 4; ++i)
            CHECK_EQ(S_full(i), doctest::Approx(S_only(i)).epsilon(1e-5));
    }

    SUBCASE("5x3 tall matrix")
    {
        SMatrix<ScalarType, 5, 3> A;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 3; ++j)
                A(i, j) = ScalarType(i + 1) * ScalarType(j + 1) + ScalarType(0.1) * (i - j);

        auto S = JacobiSingularValues(A);

        // Compare with Eigen
        pbat::Matrix<5, 3> Aeigen;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 3; ++j)
                Aeigen(i, j) = A(i, j);

        Eigen::JacobiSVD<pbat::Matrix<5, 3>> svd(Aeigen);
        auto Seigen = svd.singularValues();

        for (int i = 0; i < 3; ++i)
            CHECK_EQ(S(i), doctest::Approx(Seigen(i)).epsilon(1e-5));
    }
}

TEST_CASE("[math][linalg][mini] JacobiRotation")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Diagonalizes 2x2 symmetric matrix")
    {
        ScalarType a = 5.0, b = 2.0, c = 3.0;

        SVector<ScalarType, 2> jr = JacobiRotation(a, b, c);
        ScalarType cosTheta       = jr(0);
        ScalarType sinTheta       = jr(1);

        // Apply rotation: R^T * [a b; b c] * R should be diagonal
        // R = [c -s; s c]
        // New off-diagonal: (c^2 - s^2) * b + cs * (a - c)
        ScalarType c2  = cosTheta * cosTheta;
        ScalarType s2  = sinTheta * sinTheta;
        ScalarType cs  = cosTheta * sinTheta;
        ScalarType off = (c2 - s2) * b + cs * (a - c);
        CHECK_LT(std::fabs(off), 1e-14);
    }

    SUBCASE("Already diagonal matrix")
    {
        ScalarType a = 5.0, b = 0.0, c = 3.0;

        SVector<ScalarType, 2> jr = JacobiRotation(a, b, c);
        ScalarType cosTheta       = jr(0);
        ScalarType sinTheta       = jr(1);

        // Should return identity rotation
        CHECK_EQ(cosTheta, doctest::Approx(1.0).epsilon(1e-14));
        CHECK_LT(std::fabs(sinTheta), 1e-14);
    }
}
