#include "Determinant.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <Eigen/LU>
#include <cmath>
#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Determinant")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows              = 3;
    auto constexpr kCols              = 3;
    using ScalarType                  = pbat::Scalar;
    using MatrixType                  = SMatrix<ScalarType, kRows, kCols>;
    MatrixType Amini                  = Ones<ScalarType, kRows, kCols>();
    pbat::Matrix<kRows, kCols> Aeigen = pbat::Matrix<kRows, kCols>::Ones();
    auto detAmini                     = Determinant(Amini);
    auto detAeigen                    = Aeigen.determinant();
    CHECK_LE(std::abs(detAmini - detAeigen), ScalarType{1e-8});
}