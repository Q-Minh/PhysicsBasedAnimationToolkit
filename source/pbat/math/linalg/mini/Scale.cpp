#include "Scale.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Scale")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using ScalarType     = pbat::Scalar;
    auto A               = ScalarType(-2) * Ones<ScalarType, kRows, kCols>();
    using ScaleType      = decltype(A);
    CHECK_EQ(A.Rows(), kRows);
    CHECK_EQ(A.Cols(), kCols);
    for (auto j = 0; j < A.Rows(); ++j)
        for (auto i = 0; i < A.Cols(); ++i)
            CHECK_EQ(A(i, j), ScalarType(-2));
    PBAT_MINI_CHECK_READABLE_CONCEPTS(ScaleType);
}