#include "SubMatrix.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] SubMatrix")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType     = pbat::Scalar;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType A         = Ones<ScalarType, kRows, kCols>();
    auto slice           = A.Slice<2, 2>(1, 1);
    using SubMatrixType  = decltype(slice);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(SubMatrixType);
    PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(SubMatrixType);
    slice.SetConstant(ScalarType(2));
    for (auto j = 1; j < kCols; ++j)
        for (auto i = 1; i < kRows; ++i)
            CHECK_EQ(A(i, j), ScalarType(2));
    CHECK_EQ(A(0, 0), ScalarType(1));
    CHECK_EQ(A(0, 1), ScalarType(1));
    CHECK_EQ(A(0, 2), ScalarType(1));
    CHECK_EQ(A(0, 0), ScalarType(1));
    CHECK_EQ(A(1, 0), ScalarType(1));
    CHECK_EQ(A(2, 0), ScalarType(1));
}