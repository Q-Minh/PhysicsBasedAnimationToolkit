#include "Norm.h"

#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Scale.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Norm")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 2;
    using ScalarType     = pbat::Scalar;
    using VectorType     = SVector<ScalarType, kRows>;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;

    VectorType v = ScalarType(2) * Ones<ScalarType, kRows, 1>();
    auto nv2     = SquaredNorm(v);
    CHECK_EQ(nv2, ScalarType(12));
    auto nv         = Norm(v);
    auto nvExpected = sqrt(ScalarType(12));
    CHECK_LE(std::abs(nv - nvExpected), ScalarType(1e-8));

    MatrixType A = ScalarType(2) * Ones<ScalarType, kRows, kCols>();
    auto nA2     = SquaredNorm(A);
    CHECK_EQ(nA2, ScalarType(24));
    auto nA         = Norm(A);
    auto nAExpected = sqrt(ScalarType(24));
    CHECK_LE(std::abs(nA - nAExpected), ScalarType(1e-8));
}