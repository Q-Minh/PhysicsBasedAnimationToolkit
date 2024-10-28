#include "Matrix.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Matrix")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;

    PBAT_MINI_CHECK_READABLE_CONCEPTS(MatrixType);
    PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(MatrixType);

    MatrixType M{};
    CHECK_EQ(M.Rows(), kRows);
    CHECK_EQ(M.Cols(), kCols);
    CHECK_EQ(M.Slice<2, 2>(1, 1)(0, 0), M(1, 1));
    CHECK_EQ(M.Col(2)(1), M(1, 2));
    CHECK_EQ(M.Row(2)(1), M(2, 1));

    using MatrixViewType = SMatrixView<ScalarType, kRows, kCols>;
    PBAT_MINI_CHECK_READABLE_CONCEPTS(MatrixViewType);
    PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(MatrixViewType);
}