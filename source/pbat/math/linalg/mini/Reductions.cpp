#include "Reductions.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Reductions")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType A         = Identity<ScalarType, kRows, kCols>();
    auto Adiag           = Diag(A);
    CHECK_EQ(Adiag.Rows(), kRows);
    CHECK_EQ(Adiag.Cols(), 1);
    using DiagonalType = decltype(Adiag);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(DiagonalType);
    for (auto i = 0; i < Adiag.Rows(); ++i)
        CHECK_EQ(Adiag(i), ScalarType(1));
    Adiag.SetConstant(ScalarType(2));
    for (auto i = 0; i < Adiag.Rows(); ++i)
    {
        CHECK_EQ(Adiag(i), ScalarType(2));
        CHECK_EQ(A(i, i), ScalarType(2));
    }

    auto trA = Trace(A);
    CHECK_EQ(trA, ScalarType(6));
    auto AdotA = Dot(A, A);
    CHECK_EQ(AdotA, ScalarType(12));
}