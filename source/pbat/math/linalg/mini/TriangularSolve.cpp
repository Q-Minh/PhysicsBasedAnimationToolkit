#include "TriangularSolve.h"

#include "BinaryOperations.h"
#include "Matrix.h"
#include "Norm.h"
#include "Product.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] TriangularSolve")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;

    SUBCASE("Lower triangular solve 3x3")
    {
        SMatrix<ScalarType, 3, 3> L;
        L.SetZero();
        L(0, 0) = 2.0;
        L(1, 0) = 1.0;
        L(1, 1) = 3.0;
        L(2, 0) = -1.0;
        L(2, 1) = 2.0;
        L(2, 2) = 4.0;

        SVector<ScalarType, 3> b{4.0, 7.0, 9.0};

        auto x = LowerTriangularSolve(L, b);

        // Check L * x = b
        SVector<ScalarType, 3> Lx = L * x;
        ScalarType error          = Norm(Lx - b);
        CHECK_LE(error, ScalarType{1e-6});
    }

    SUBCASE("Upper triangular solve 3x3")
    {
        SMatrix<ScalarType, 3, 3> R;
        R.SetZero();
        R(0, 0) = 2.0;
        R(0, 1) = 1.0;
        R(0, 2) = -1.0;
        R(1, 1) = 3.0;
        R(1, 2) = 2.0;
        R(2, 2) = 4.0;

        SVector<ScalarType, 3> b{1.0, 11.0, 12.0};

        auto x = UpperTriangularSolve(R, b);

        // Check R * x = b
        SVector<ScalarType, 3> Rx = R * x;
        ScalarType error          = Norm(Rx - b);
        CHECK_LE(error, ScalarType{1e-6});
    }

    SUBCASE("Lower triangular solve with multiple RHS")
    {
        SMatrix<ScalarType, 2, 2> L;
        L.SetZero();
        L(0, 0) = 3.0;
        L(1, 0) = 2.0;
        L(1, 1) = 5.0;

        SMatrix<ScalarType, 2, 2> B;
        B(0, 0) = 3.0;
        B(0, 1) = 6.0;
        B(1, 0) = 7.0;
        B(1, 1) = 12.0;

        auto X = LowerTriangularSolve(L, B);

        // Check L * X = B
        SMatrix<ScalarType, 2, 2> LX = L * X;
        ScalarType error             = Norm(LX - B);
        CHECK_LE(error, ScalarType{1e-6});
    }

    SUBCASE("Upper triangular solve with multiple RHS")
    {
        SMatrix<ScalarType, 2, 2> R;
        R.SetZero();
        R(0, 0) = 3.0;
        R(0, 1) = 2.0;
        R(1, 1) = 5.0;

        SMatrix<ScalarType, 2, 2> B;
        B(0, 0) = 7.0;
        B(0, 1) = 16.0;
        B(1, 0) = 5.0;
        B(1, 1) = 10.0;

        auto X = UpperTriangularSolve(R, B);

        // Check R * X = B
        SMatrix<ScalarType, 2, 2> RX = R * X;
        ScalarType error             = Norm(RX - B);
        CHECK_LE(error, ScalarType{1e-6});
    }
}
