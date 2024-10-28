#include "Transpose.h"

#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Transpose")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType     = pbat::Scalar;
    auto constexpr kRows = 3;
    auto constexpr kCols = 4;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType A;
    A.SetZero();
    A(0, 1)             = ScalarType(1);
    auto AT             = A.Transpose();
    using TransposeType = decltype(AT);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(TransposeType);
    PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(TransposeType);
    CHECK_EQ(AT(1, 0), ScalarType(1));
    AT(2, 3) = ScalarType(3);
    CHECK_EQ(A(3, 2), ScalarType(3));
}