#include "ClosestPointQueries.h"

#include <Eigen/Geometry>
#include <doctest/doctest.h>
#include <pbat/Aliases.h>

TEST_CASE("[geometry] Can obtain closest point on triangle ABC to point P")
{
    using namespace pbat;
    Vector<3> const A{0., 0., 0.};
    Vector<3> const B{1., 0., 0.};
    Vector<3> const C{0., 1., 1.};
    Vector<3> const n    = (B - A).cross(C - A).normalized();
    Scalar constexpr eps = 1e-15;

    SUBCASE("Point is in triangle")
    {
        Vector<3> const uvwExpected{0.11, 0.33, 0.56};
        Vector<3> const Pexpected = uvwExpected(0) * A + uvwExpected(1) * B + uvwExpected(2) * C;
        Vector<3> const uvw = geometry::ClosestPointQueries::UvwPointInTriangle(Pexpected, A, B, C);
        CHECK(uvw.isApprox(uvwExpected, eps));
        Vector<3> const P = geometry::ClosestPointQueries::PointInTriangle(Pexpected, A, B, C);
        CHECK(P.isApprox(Pexpected, eps));
    }
    SUBCASE("Point is not in triangle and closest to face interior")
    {
        Vector<3> const uvwExpected{0.11, 0.33, 0.56};
        Vector<3> const Pexpected = uvwExpected(0) * A + uvwExpected(1) * B + uvwExpected(2) * C;
        Scalar const d            = 0.5;
        Vector<3> const uvw =
            geometry::ClosestPointQueries::UvwPointInTriangle(Pexpected + d * n, A, B, C);
        CHECK(uvw.isApprox(uvwExpected, eps));
        Vector<3> const P = uvw(0) * A + uvw(1) * B + uvw(2) * C;
        CHECK(P.isApprox(Pexpected, eps));
    }
    SUBCASE("Point is not in triangle and closest to a triangle edge")
    {
        Vector<3> const uvwExpected{0.25, 0.75, 0.};
        Vector<3> const Pexpected = uvwExpected(0) * A + uvwExpected(1) * B;
        Vector<3> const s         = (B - A).cross(n).normalized();
        Scalar const d            = 0.5;
        Vector<3> const uvw =
            geometry::ClosestPointQueries::UvwPointInTriangle(Pexpected + d * s, A, B, C);
        CHECK(uvw.isApprox(uvwExpected, eps));
        Vector<3> const P = uvw(0) * A + uvw(1) * B + uvw(2) * C;
        CHECK(P.isApprox(Pexpected, eps));
    }
}

TEST_CASE("[geometry] Can obtain closest point on tetrahedron ABCD to point P")
{
    using namespace pbat;
    Vector<3> const A{0., 0., 0.};
    Vector<3> const B{1., 0., 0.};
    Vector<3> const C{0., 1., 0.};
    Vector<3> const D{0., 0., 1.};
    Scalar constexpr eps = 1e-15;

    SUBCASE("Point is in tetrahedron")
    {
        Vector<4> const u{0.25, 0.25, 0.25, 0.25};
        Vector<3> const Pexpected = u(0) * A + u(1) * B + u(2) * C + u(3) * D;
        Vector<3> const P =
            geometry::ClosestPointQueries::PointInTetrahedron(Pexpected, A, B, C, D);
        CHECK(P.isApprox(Pexpected, eps));
    }
    SUBCASE("Point is outside tetrahedron")
    {
        Vector<4> const u{0.25, 0.25, 0., 0.5};
        Vector<3> const Pexpected = u(0) * A + u(1) * B + u(2) * C + u(3) * D;
        Vector<3> const n         = (B - A).cross(D - A).normalized();
        Scalar const t            = 0.5;
        Vector<3> const P =
            geometry::ClosestPointQueries::PointInTetrahedron(Pexpected + t * n, A, B, C, D);
        CHECK(P.isApprox(Pexpected, eps));
    }
}