#include "Eigen.h"

#include "Matrix.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Eigen")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 4;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;
    MatrixType Amini     = Ones<ScalarType, kRows, kCols>();
    auto Aeigen          = ToEigen(Amini);
    CHECK_EQ(Aeigen.rows(), Amini.Rows());
    CHECK_EQ(Aeigen.cols(), Amini.Cols());
    for (auto j = 0; j < Amini.Cols(); ++j)
        for (auto i = 0; i < Amini.Rows(); ++i)
            CHECK_EQ(Aeigen(i, j), Amini(i, j));

    auto AeigenToMini = FromEigen(Aeigen);
    CHECK_EQ(AeigenToMini.Rows(), Amini.Rows());
    CHECK_EQ(AeigenToMini.Cols(), Amini.Cols());
    for (auto j = 0; j < Amini.Cols(); ++j)
        for (auto i = 0; i < Amini.Rows(); ++i)
            CHECK_EQ(AeigenToMini(i, j), Amini(i, j));
    using EigenWrapperType = decltype(AeigenToMini);
    PBAT_MINI_CHECK_READABLE_CONCEPTS(EigenWrapperType);
    PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(EigenWrapperType);
}