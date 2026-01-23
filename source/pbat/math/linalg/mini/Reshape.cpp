#include "Reshape.h"

#include "Matrix.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Reshape")
{
    using namespace pbat::math::linalg::mini;

    SUBCASE("Reshape 2x3 to 3x2 column-major")
    {
        // Column-major 2x3 matrix:
        // | 1  3  5 |
        // | 2  4  6 |
        // Linear storage: [1, 2, 3, 4, 5, 6]
        SMatrix<int, 2, 3> A{};
        A(0, 0) = 1; A(0, 1) = 3; A(0, 2) = 5;
        A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;

        auto B = Reshape<3, 2>(A);

        // Reshaped 3x2 column-major:
        // | 1  4 |
        // | 2  5 |
        // | 3  6 |
        CHECK(B.Rows() == 3);
        CHECK(B.Cols() == 2);
        CHECK(B(0, 0) == 1);
        CHECK(B(1, 0) == 2);
        CHECK(B(2, 0) == 3);
        CHECK(B(0, 1) == 4);
        CHECK(B(1, 1) == 5);
        CHECK(B(2, 1) == 6);
    }

    SUBCASE("Reshape 2x3 to 6x1 (flatten)")
    {
        SMatrix<int, 2, 3> A{};
        A(0, 0) = 1; A(0, 1) = 3; A(0, 2) = 5;
        A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;

        auto v = Reshape<6, 1>(A);

        CHECK(v.Rows() == 6);
        CHECK(v.Cols() == 1);
        CHECK(v(0) == 1);
        CHECK(v(1) == 2);
        CHECK(v(2) == 3);
        CHECK(v(3) == 4);
        CHECK(v(4) == 5);
        CHECK(v(5) == 6);
    }

    SUBCASE("Reshape 6x1 to 2x3")
    {
        SMatrix<int, 6, 1> v{};
        v(0) = 1; v(1) = 2; v(2) = 3;
        v(3) = 4; v(4) = 5; v(5) = 6;

        auto A = Reshape<2, 3>(v);

        CHECK(A.Rows() == 2);
        CHECK(A.Cols() == 3);
        CHECK(A(0, 0) == 1);
        CHECK(A(1, 0) == 2);
        CHECK(A(0, 1) == 3);
        CHECK(A(1, 1) == 4);
        CHECK(A(0, 2) == 5);
        CHECK(A(1, 2) == 6);
    }

    SUBCASE("Reshape 2x3 to 3x2 row-major")
    {
        SMatrix<int, 2, 3> A{};
        A(0, 0) = 1; A(0, 1) = 3; A(0, 2) = 5;
        A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;

        auto B = Reshape<3, 2, true>(A);

        // Row-major 3x2 interpretation:
        // Linear [1,2,3,4,5,6] -> row-major means:
        // | 1  2 |
        // | 3  4 |
        // | 5  6 |
        CHECK(B.Rows() == 3);
        CHECK(B.Cols() == 2);
        CHECK(B(0, 0) == 1);
        CHECK(B(0, 1) == 2);
        CHECK(B(1, 0) == 3);
        CHECK(B(1, 1) == 4);
        CHECK(B(2, 0) == 5);
        CHECK(B(2, 1) == 6);
    }

    SUBCASE("Reshape preserves data through assignment")
    {
        SMatrix<float, 3, 4> A{};
        for (int i = 0; i < 12; ++i)
            A(i) = static_cast<float>(i + 1);

        auto B = Reshape<4, 3>(A);
        SMatrix<float, 4, 3> C = B;

        for (int i = 0; i < 12; ++i)
            CHECK(C(i) == static_cast<float>(i + 1));
    }

    SUBCASE("Reshape 1x6 to 2x3")
    {
        SMatrix<int, 1, 6> v{};
        v(0, 0) = 1; v(0, 1) = 2; v(0, 2) = 3;
        v(0, 3) = 4; v(0, 4) = 5; v(0, 5) = 6;

        auto A = Reshape<2, 3>(v);

        CHECK(A.Rows() == 2);
        CHECK(A.Cols() == 3);
        // Linear access should match
        for (int i = 0; i < 6; ++i)
            CHECK(A(i) == i + 1);
    }
}
