#include "Stack.h"

#include "Matrix.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Stack")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 2;
    auto constexpr kCols = 3;
    using ScalarType     = double;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType A{};
    A.SetConstant(ScalarType(1));
    MatrixType B{};
    B.SetConstant(ScalarType(2));
    auto hstack = HStack(A, B);
    CHECK_EQ(hstack.Rows(), A.Rows());
    CHECK_EQ(hstack.Cols(), A.Cols() + B.Cols());
    for (auto i = 0; i < kRows; ++i)
    {
        for (auto j = 0; j < kCols; ++j)
        {
            CHECK_EQ(hstack(i, j), ScalarType(1));
        }
        for (auto j = kCols; j < 2 * kCols; ++j)
        {
            CHECK_EQ(hstack(i, j), ScalarType(2));
        }
    }
    auto vstack = VStack(A, B);
    CHECK_EQ(vstack.Cols(), A.Cols());
    CHECK_EQ(vstack.Rows(), A.Rows() + B.Rows());
    for (auto j = 0; j < kCols; ++j)
    {
        for (auto i = 0; i < kRows; ++i)
        {
            CHECK_EQ(vstack(i, j), ScalarType(1));
        }
        for (auto i = kRows; i < 2 * kRows; ++i)
        {
            CHECK_EQ(vstack(i, j), ScalarType(2));
        }
    }
}
