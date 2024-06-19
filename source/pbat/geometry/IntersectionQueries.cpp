#include "IntersectionQueries.h"

#include <doctest/doctest.h>
#include <pbat/Aliases.h>

TEST_CASE("[geometry] Line segment and sphere Intersection")
{
    using namespace pbat;
    SUBCASE("Non-intersecting line segment and sphere")
    {
        Vector<3> const P{0., 0., 0.};
        Vector<3> const Q{1., 1., 1.};
        Vector<3> const C{2., 2., 2.};
        Scalar const r = 1.;
        std::optional<Vector<3>> const intersection =
            geometry::IntersectionQueries::LineSegmentSphere(P, Q, C, r);
        CHECK_FALSE(intersection.has_value());
    }
    SUBCASE("Intersecting line segment and sphere")
    {
        Vector<3> const P{0., 0., 0.};
        Vector<3> const Q{2., 2., 2.};
        Vector<3> const C{1., 1., 1.};
        Scalar const r                        = 1.;
        Vector<3> const dQP                   = (P - Q).normalized();
        Vector<3> const expected_intersection = C + r * dQP;
        std::optional<Vector<3>> const intersection =
            geometry::IntersectionQueries::LineSegmentSphere(P, Q, C, r);
        CHECK(intersection.has_value());
        auto constexpr eps = 1e-15;
        CHECK(intersection.value().isApprox(expected_intersection, eps));
    }
}

TEST_CASE("[geometry] Lines intersecting triangles are detected")
{
    using namespace pbat;
    Vector<3> const P{0.25, 0.25, -1.};
    Vector<3> const Q{0.25, 0.25, 1.};
    Vector<3> const A{0., 0., 0.};
    Vector<3> const B{1., 0., 0.};
    Vector<3> const C{0., 1., 0.};

    SUBCASE("Line through PQ passes through the triangle T")
    {
        auto const do_assert = [&](std::optional<Vector<3>> const& uvwIntersection) {
            CHECK(uvwIntersection.has_value());
            Vector<3> const& uvw         = uvwIntersection.value();
            Vector<3> const intersection = uvw(0) * A + uvw(1) * B + uvw(2) * C;
            Vector<3> const expected_intersection{0.25, 0.25, 0.};
            auto constexpr eps = 1e-15;
            CHECK(intersection.isApprox(expected_intersection, eps));
        };
        auto const uvwIntersection1 = geometry::IntersectionQueries::UvwLineTriangle(P, Q, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 = geometry::IntersectionQueries::UvwLineTriangle(Q, P, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 = geometry::IntersectionQueries::UvwLineTriangle(P, Q, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 = geometry::IntersectionQueries::UvwLineTriangle(Q, P, A, C, B);
        do_assert(uvwIntersection4);
    }
    SUBCASE("Line through PQ translated in triangle normal direction passes through the triangle T")
    {
        auto const do_assert = [&](std::optional<Vector<3>> const& uvwIntersection) {
            CHECK(uvwIntersection.has_value());
            Vector<3> const& uvw         = uvwIntersection.value();
            Vector<3> const intersection = uvw(0) * A + uvw(1) * B + uvw(2) * C;
            Vector<3> const expected_intersection{0.25, 0.25, 0.};
            auto constexpr eps = 1e-15;
            CHECK(intersection.isApprox(expected_intersection, eps));
        };
        Vector<3> const t{0., 0., 1.01};
        Vector<3> const P2 = P + t;
        Vector<3> const Q2 = Q + t;
        auto const uvwIntersection1 =
            geometry::IntersectionQueries::UvwLineTriangle(P2, Q2, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            geometry::IntersectionQueries::UvwLineTriangle(Q2, P2, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            geometry::IntersectionQueries::UvwLineTriangle(P2, Q2, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            geometry::IntersectionQueries::UvwLineTriangle(Q2, P2, A, C, B);
        do_assert(uvwIntersection4);
    }
    SUBCASE(
        "Line through PQ translated in direction perpendicular to triangle normal direction does "
        "not pass through the triangle T")
    {
        auto const do_assert = [](std::optional<Vector<3>> const& intersection) {
            CHECK_FALSE(intersection.has_value());
        };
        Vector<3> const t{2., 0., 0.};
        Vector<3> const P2 = P + t;
        Vector<3> const Q2 = Q + t;
        auto const uvwIntersection1 =
            geometry::IntersectionQueries::UvwLineTriangle(P2, Q2, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            geometry::IntersectionQueries::UvwLineTriangle(Q2, P2, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            geometry::IntersectionQueries::UvwLineTriangle(P2, Q2, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            geometry::IntersectionQueries::UvwLineTriangle(Q2, P2, A, C, B);
        do_assert(uvwIntersection4);
    }
}

TEST_CASE("[geometry] Line segments intersecting triangles are detected")
{
    using namespace pbat;
    Vector<3> const P{0.25, 0.25, -1.};
    Vector<3> const Q{0.25, 0.25, 1.};
    Vector<3> const A{0., 0., 0.};
    Vector<3> const B{1., 0., 0.};
    Vector<3> const C{0., 1., 0.};

    SUBCASE("Line segment PQ passes through the triangle T")
    {
        auto const do_assert = [&](std::optional<Vector<3>> const& uvwIntersection) {
            CHECK(uvwIntersection.has_value());
            Vector<3> const& uvw         = uvwIntersection.value();
            Vector<3> const intersection = uvw(0) * A + uvw(1) * B + uvw(2) * C;
            Vector<3> const expected_intersection{0.25, 0.25, 0.};
            auto constexpr eps = 1e-15;
            CHECK(intersection.isApprox(expected_intersection, eps));
        };
        auto const uvwIntersection1 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(P, Q, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(Q, P, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(P, Q, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(Q, P, A, C, B);
        do_assert(uvwIntersection4);
    }
    SUBCASE("Line segment PQ does not pass through the triangle T")
    {
        auto const do_assert = [](std::optional<Vector<3>> const& intersection) {
            CHECK_FALSE(intersection.has_value());
        };
        Vector<3> const t{0., 0., 1.01};
        Vector<3> const P2 = P + t;
        Vector<3> const Q2 = Q + t;
        auto const uvwIntersection1 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(P2, Q2, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(Q2, P2, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(P2, Q2, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            geometry::IntersectionQueries::UvwLineSegmentTriangle(Q2, P2, A, C, B);
        do_assert(uvwIntersection4);
    }
}
