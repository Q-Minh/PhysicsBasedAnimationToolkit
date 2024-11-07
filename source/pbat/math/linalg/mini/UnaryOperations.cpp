#include "UnaryOperations.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <cmath>
#include <doctest/doctest.h>
#include <math.h>

TEST_CASE("[math][linalg][mini] UnaryOperations")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType A;
    A.SetConstant(ScalarType(3));
    auto A2 = Squared(A);
    CHECK_EQ(A2.Rows(), A.Rows());
    CHECK_EQ(A2.Cols(), A.Cols());
    for (auto j = 0; j < A2.Cols(); ++j)
        for (auto i = 0; i < A2.Rows(); ++i)
            CHECK_EQ(A2(i, j), ScalarType(9));
    using SquaredType = decltype(A2);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(SquaredType);

    using VectorType = SVector<ScalarType, kRows>;
    VectorType v     = Ones<ScalarType, kRows, 1>();
    auto unitv       = Normalized(v);
    auto viExpected  = ScalarType(1) / sqrt(ScalarType(3));
    CHECK_EQ(unitv.Rows(), kRows);
    CHECK_EQ(unitv.Cols(), 1);
    for (auto i = 0; i < kRows; ++i)
        CHECK_LE(std::abs(unitv(i) - viExpected), ScalarType(1e-8));
}