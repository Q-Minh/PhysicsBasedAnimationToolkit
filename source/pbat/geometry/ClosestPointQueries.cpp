#include "ClosestPointQueries.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] Can obtain closest point on triangle ABC to point P")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SVector<ScalarType, 3> const A{0., 0., 0.};
    SVector<ScalarType, 3> const B{1., 0., 0.};
    SVector<ScalarType, 3> const C{0., 1., 1.};
    SVector<ScalarType, 3> const n = Normalized(Cross(B - A, C - A));
    ScalarType constexpr eps       = 1e-15;

    SUBCASE("Point is in triangle")
    {
        SVector<ScalarType, 3> const uvwExpected{0.11, 0.33, 0.56};
        SVector<ScalarType, 3> const Pexpected =
            uvwExpected(0) * A + uvwExpected(1) * B + uvwExpected(2) * C;
        SVector<ScalarType, 3> const uvw =
            pbat::geometry::ClosestPointQueries::UvwPointInTriangle(Pexpected, A, B, C);
        CHECK(ToEigen(uvw).isApprox(ToEigen(uvwExpected), eps));
        SVector<ScalarType, 3> const P =
            pbat::geometry::ClosestPointQueries::PointInTriangle(Pexpected, A, B, C);
        CHECK(ToEigen(P).isApprox(ToEigen(Pexpected), eps));
    }
    SUBCASE("Point is not in triangle and closest to face interior")
    {
        SVector<ScalarType, 3> const uvwExpected{0.11, 0.33, 0.56};
        SVector<ScalarType, 3> const Pexpected =
            uvwExpected(0) * A + uvwExpected(1) * B + uvwExpected(2) * C;
        ScalarType const d = 0.5;
        SVector<ScalarType, 3> const uvw =
            pbat::geometry::ClosestPointQueries::UvwPointInTriangle(Pexpected + d * n, A, B, C);
        CHECK(ToEigen(uvw).isApprox(ToEigen(uvwExpected), eps));
        SVector<ScalarType, 3> const P = uvw(0) * A + uvw(1) * B + uvw(2) * C;
        CHECK(ToEigen(P).isApprox(ToEigen(Pexpected), eps));
    }
    SUBCASE("Point is not in triangle and closest to a triangle edge")
    {
        SVector<ScalarType, 3> const uvwExpected{0.25, 0.75, 0.};
        SVector<ScalarType, 3> const Pexpected = uvwExpected(0) * A + uvwExpected(1) * B;
        SVector<ScalarType, 3> const s         = Normalized(Cross(B - A, n));
        ScalarType const d                     = 0.5;
        SVector<ScalarType, 3> const uvw =
            pbat::geometry::ClosestPointQueries::UvwPointInTriangle(Pexpected + d * s, A, B, C);
        CHECK(ToEigen(uvw).isApprox(ToEigen(uvwExpected), eps));
        SVector<ScalarType, 3> const P = uvw(0) * A + uvw(1) * B + uvw(2) * C;
        CHECK(ToEigen(P).isApprox(ToEigen(Pexpected), eps));
    }
}

TEST_CASE("[geometry] Can obtain closest point on tetrahedron ABCD to point P")
{
    using ScalarType = pbat::Scalar;
    using namespace pbat::math::linalg::mini;
    SVector<ScalarType, 3> const A{0., 0., 0.};
    SVector<ScalarType, 3> const B{1., 0., 0.};
    SVector<ScalarType, 3> const C{0., 1., 0.};
    SVector<ScalarType, 3> const D{0., 0., 1.};
    ScalarType constexpr eps = 1e-15;

    SUBCASE("Point is in tetrahedron")
    {
        SVector<ScalarType, 4> const u{0.25, 0.25, 0.25, 0.25};
        SVector<ScalarType, 3> const Pexpected = u(0) * A + u(1) * B + u(2) * C + u(3) * D;
        SVector<ScalarType, 3> const P =
            pbat::geometry::ClosestPointQueries::PointInTetrahedron(Pexpected, A, B, C, D);
        CHECK(ToEigen(P).isApprox(ToEigen(Pexpected), eps));
    }
    SUBCASE("Point is outside tetrahedron")
    {
        SVector<ScalarType, 4> const u{0.25, 0.25, 0., 0.5};
        SVector<ScalarType, 3> const Pexpected = u(0) * A + u(1) * B + u(2) * C + u(3) * D;
        SVector<ScalarType, 3> const n         = Normalized(Cross(B - A, D - A));
        ScalarType const t                     = 0.5;
        SVector<ScalarType, 3> const P =
            pbat::geometry::ClosestPointQueries::PointInTetrahedron(Pexpected + t * n, A, B, C, D);
        CHECK(ToEigen(P).isApprox(ToEigen(Pexpected), eps));
    }
}