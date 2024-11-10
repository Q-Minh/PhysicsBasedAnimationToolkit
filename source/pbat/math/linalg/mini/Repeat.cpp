#include "Repeat.h"

#include "BinaryOperations.h"
#include "Matrix.h"
#include "Norm.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Repeat")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType A         = Ones<ScalarType, kRows, kCols>();
    auto AR              = Repeat<2, 3>(A);
    using RepeatType     = decltype(AR);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(RepeatType);

    CHECK_EQ(AR.Rows(), A.Rows() * 2);
    CHECK_EQ(AR.Cols(), A.Cols() * 3);
    for (auto br =0; br < 2; ++br)
        for (auto bc = 0; bc < 3; ++bc)
            CHECK_LT(SquaredNorm(AR.Slice<kRows, kCols>(br*kRows, bc*kCols) - A), 1e-10);
}