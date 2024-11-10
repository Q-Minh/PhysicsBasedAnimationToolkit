#include "Geometry.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Geometry")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    using ScalarType     = pbat::Scalar;
    using VectorType     = SVector<ScalarType, kRows>;

    VectorType ex, ey;
    ex(0) = ScalarType(1);
    ex(1) = ScalarType(0);
    ex(2) = ScalarType(0);
    ey(0) = ScalarType(0);
    ey(1) = ScalarType(1);
    ey(2) = ScalarType(0);

    auto cross = Cross(ex, ey);
    CHECK_EQ(cross.Rows(), 3);
    CHECK_EQ(cross.Cols(), 1);
    CHECK_EQ(cross(0), ScalarType(0));
    CHECK_EQ(cross(1), ScalarType(0));
    CHECK_EQ(cross(2), ScalarType(1));

    using CrossType = decltype(cross);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(CrossType);
}