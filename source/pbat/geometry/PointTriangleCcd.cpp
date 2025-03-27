#include "PointTriangleCcd.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] PointTriangleCcd")
{
    using ScalarType = pbat::Scalar;
    using namespace pbat::math::linalg::mini;
    SVector<ScalarType, 3> PT, P;
    SVector<ScalarType, 3> AT, A;
    SVector<ScalarType, 3> BT, B;
    SVector<ScalarType, 3> CT, C;
    SUBCASE("Point will intersect triangle")
    {
        PT = SVector<ScalarType, 3>{0., 0.25, 0.25};
        AT = SVector<ScalarType, 3>{1., 0., 0.};
        BT = SVector<ScalarType, 3>{1., 1., 0.};
        CT = SVector<ScalarType, 3>{1., 0., 1.};
        P  = PT + SVector<ScalarType, 3>{1., 0., 0.};
        A  = AT + SVector<ScalarType, 3>{-1., 0., 0.};
        B  = BT + SVector<ScalarType, 3>{-1., 0., 0.};
        C  = CT + SVector<ScalarType, 3>{-1., 0., 0.};
        SVector<ScalarType, 4> const r =
            pbat::geometry::PointTriangleCcd(PT, AT, BT, CT, P, A, B, C);
        auto t   = r[0];
        auto uvw = r.Slice<3, 1>(1, 0);
        CHECK_EQ(t, doctest::Approx(0.5));
        CHECK_EQ(uvw(0), doctest::Approx(0.5));
        CHECK_EQ(uvw(1), doctest::Approx(0.25));
        CHECK_EQ(uvw(2), doctest::Approx(0.25));
    }
}