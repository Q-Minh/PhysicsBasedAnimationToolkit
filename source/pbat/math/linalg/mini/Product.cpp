#include "Product.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Product")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 4;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;

    MatrixType Amini;
    Amini.SetConstant(ScalarType(3));
    auto AATmini      = Amini * Amini.Transpose();
    using ProductType = decltype(AATmini);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(ProductType);

    pbat::Matrix<kRows, kCols> Aeigen   = pbat::Matrix<kRows, kCols>::Constant(ScalarType(3));
    pbat::Matrix<kRows, kRows> AATeigen = Aeigen * Aeigen.transpose();

    CHECK_EQ(AATmini.Rows(), AATeigen.rows());
    CHECK_EQ(AATmini.Cols(), AATeigen.cols());
    for (auto j = 0; j < AATmini.Cols(); ++j)
        for (auto i = 0; i < AATmini.Rows(); ++i)
            CHECK_EQ(AATmini(i, j), AATeigen(i, j));
}