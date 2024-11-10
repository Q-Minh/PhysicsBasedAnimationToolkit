#include "Assign.h"

#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Assign")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType A{};
    MatrixType B{};
    B.SetConstant(ScalarType{1.});

    auto const check_equal_to = [&A](auto k) {
        for (auto j = 0; j < A.Cols(); ++j)
            for (auto i = 0; i < A.Rows(); ++i)
                CHECK_EQ(A(i, j), static_cast<ScalarType>(k));
    };
    Assign(A, B);
    check_equal_to(1.);
    AddAssign(A, B);
    check_equal_to(2.);
    SubtractAssign(A, B);
    check_equal_to(1.);
    AssignScalar(A, ScalarType{3.});
    check_equal_to(3.);
    DivideAssign(A, ScalarType{3.});
    check_equal_to(1.);
    MultiplyAssign(A, ScalarType{3.});
    check_equal_to(3.);
}