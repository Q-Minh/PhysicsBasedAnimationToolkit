#include "Inverse.h"

#include "Matrix.h"
#include "Scale.h"
#include "pbat/Aliases.h"

#include <Eigen/LU>
#include <cmath>
#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Inverse")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows                 = 3;
    auto constexpr kCols                 = 3;
    using ScalarType                     = pbat::Scalar;
    using MatrixType                     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType Amini                     = ScalarType(5) * Identity<ScalarType, kRows, kCols>();
    pbat::Matrix<kRows, kCols> Aeigen    = ScalarType(5) * pbat::Matrix<kRows, kCols>::Identity();
    MatrixType AminiInv                  = Inverse(Amini);
    pbat::Matrix<kRows, kCols> AeigenInv = Aeigen.inverse();

    for (auto j = 0; j < Amini.Cols(); ++j)
        for (auto i = 0; i < Amini.Rows(); ++i)
            CHECK_LE(std::abs(AminiInv(i, j) - AeigenInv(i, j)), ScalarType{1e-8});
}