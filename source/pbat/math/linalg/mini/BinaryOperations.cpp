#include "BinaryOperations.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] BinaryOperations")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows      = 3;
    auto constexpr kCols      = 3;
    using ScalarType          = pbat::Scalar;
    using MatrixType          = SMatrix<ScalarType, kRows, kCols>;
    auto const check_equal_to = [](auto C, auto k) {
        for (auto j = 0; j < C.Cols(); ++j)
            for (auto i = 0; i < C.Rows(); ++i)
                CHECK_EQ(C(i, j), static_cast<ScalarType>(k));
    };

    MatrixType A, B;
    A.SetConstant(ScalarType{1.});
    B.SetConstant(ScalarType{2.});
    check_equal_to(A + B, 3.);
    check_equal_to(A - B, -1.);
    check_equal_to(Min(A, B), 1.);
    check_equal_to(Max(A, B), 2.);
    A += B;
    check_equal_to(A, 3.);
    A -= B;
    check_equal_to(A, 1.);

    using SumType         = Sum<MatrixType, MatrixType>;
    using SubtractionType = Subtraction<MatrixType, MatrixType>;
    using MinimumType     = Minimum<MatrixType, MatrixType>;
    using MaximumType     = Maximum<MatrixType, MatrixType>;
    PBAT_MINI_CHECK_READABLE_CONCEPTS(SumType);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(SubtractionType);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(MinimumType);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(MaximumType);
}