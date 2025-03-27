#include "EdgeEdgeCcd.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] EdgeEdgeCcd")
{
    using ScalarType = pbat::Scalar;
    using namespace pbat::math::linalg::mini;
    SVector<ScalarType, 3> P1T, P1;
    SVector<ScalarType, 3> Q1T, Q1;
    SVector<ScalarType, 3> P2T, P2;
    SVector<ScalarType, 3> Q2T, Q2;
    SUBCASE("Point will intersect triangle")
    {
        P1T = SVector<ScalarType, 3>{0., 0., 0.};
        Q1T = SVector<ScalarType, 3>{0., 1., 1.};
        P2T = SVector<ScalarType, 3>{1., 0., 1.};
        Q2T = SVector<ScalarType, 3>{1., 1., 0.};
        P1  = P1T + SVector<ScalarType, 3>{1., 0., 0.};
        Q1  = Q1T + SVector<ScalarType, 3>{1., 0., 0.};
        P2  = P2T + SVector<ScalarType, 3>{-1., 0., 0.};
        Q2  = Q2T + SVector<ScalarType, 3>{-1., 0., 0.};
        SVector<ScalarType, 3> const r =
            pbat::geometry::EdgeEdgeCcd(P1T, Q1T, P2T, Q2T, P1, Q1, P2, Q2);
        auto t  = r[0];
        auto uv = r.Slice<2, 1>(1, 0);
        CHECK_EQ(t, doctest::Approx(0.5));
        CHECK_EQ(uv(0), doctest::Approx(0.5));
        CHECK_EQ(uv(1), doctest::Approx(0.5));
    }
}