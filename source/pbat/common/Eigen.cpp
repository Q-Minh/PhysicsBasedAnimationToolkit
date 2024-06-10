#include "Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[common] Conversions from STL to Eigen")
{
    using namespace pbat;
    SUBCASE("Arithmetic range")
    {
        std::vector<Scalar> v{1., 2., 3.};
        auto const veigen = common::ToEigen(v);
        CHECK_EQ(veigen.size(), v.size());
        for (auto i = 0u; i < veigen.size(); ++i)
            CHECK_EQ(v[i], veigen(i));
    }
    SUBCASE("Matrix arithmetic range")
    {
        std::vector<Vector<3>> v{{1., 2., 3.}, {4., 5., 6.}};
        auto const ve = common::ToEigen(v).reshaped(3, v.size());
        CHECK_EQ(ve.cols(), v.size());
        for (auto j = 0u; j < ve.cols(); ++j)
            for (auto i = 0u; i < ve.rows(); ++i)
                CHECK_EQ(v[j](i), ve(i, j));
    }
}
