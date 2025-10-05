#include "IntersectionQueries.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] Line segment and sphere Intersection")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Non-intersecting line segment and sphere")
    {
        SVector<ScalarType, 3> const P{0., 0., 0.};
        SVector<ScalarType, 3> const Q{1., 1., 1.};
        SVector<ScalarType, 3> const C{2., 2., 2.};
        ScalarType const r = 1.;
        std::optional<SVector<ScalarType, 3>> const intersection =
            pbat::geometry::IntersectionQueries::LineSegmentSphere(P, Q, C, r);
        CHECK_FALSE(intersection.has_value());
    }
    SUBCASE("Intersecting line segment and sphere")
    {
        SVector<ScalarType, 3> const P{0., 0., 0.};
        SVector<ScalarType, 3> const Q{2., 2., 2.};
        SVector<ScalarType, 3> const C{1., 1., 1.};
        ScalarType const r                                 = 1.;
        SVector<ScalarType, 3> const dQP                   = Normalized(P - Q);
        SVector<ScalarType, 3> const expected_intersection = C + r * dQP;
        std::optional<SVector<ScalarType, 3>> const intersection =
            pbat::geometry::IntersectionQueries::LineSegmentSphere(P, Q, C, r);
        CHECK(intersection.has_value());
        auto constexpr eps = 1e-15;
        bool bAreEqual     = All(Abs(intersection.value() - expected_intersection) < eps);
        CHECK(bAreEqual);
    }
}

TEST_CASE("[geometry] Lines intersecting triangles are detected")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SVector<ScalarType, 3> const P{0.25, 0.25, -1.};
    SVector<ScalarType, 3> const Q{0.25, 0.25, 1.};
    SVector<ScalarType, 3> const A{0., 0., 0.};
    SVector<ScalarType, 3> const B{1., 0., 0.};
    SVector<ScalarType, 3> const C{0., 1., 0.};

    SUBCASE("Line through PQ passes through the triangle T")
    {
        auto const do_assert = [&](std::optional<SVector<ScalarType, 3>> const& uvwIntersection) {
            CHECK(uvwIntersection.has_value());
            SVector<ScalarType, 3> const& uvw         = uvwIntersection.value();
            SVector<ScalarType, 3> const intersection = uvw(0) * A + uvw(1) * B + uvw(2) * C;
            SVector<ScalarType, 3> const expected_intersection{0.25, 0.25, 0.};
            auto constexpr eps = 1e-15;
            CHECK(All(Abs(intersection - expected_intersection) < eps));
        };
        auto const uvwIntersection1 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(P, Q, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(Q, P, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(P, Q, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(Q, P, A, C, B);
        do_assert(uvwIntersection4);
    }
    SUBCASE("Line through PQ translated in triangle normal direction passes through the triangle T")
    {
        auto const do_assert = [&](std::optional<SVector<ScalarType, 3>> const& uvwIntersection) {
            CHECK(uvwIntersection.has_value());
            SVector<ScalarType, 3> const& uvw         = uvwIntersection.value();
            SVector<ScalarType, 3> const intersection = uvw(0) * A + uvw(1) * B + uvw(2) * C;
            SVector<ScalarType, 3> const expected_intersection{0.25, 0.25, 0.};
            auto constexpr eps = 1e-15;
            CHECK(All(Abs(intersection - expected_intersection) < eps));
        };
        SVector<ScalarType, 3> const t{0., 0., 1.01};
        SVector<ScalarType, 3> const P2 = P + t;
        SVector<ScalarType, 3> const Q2 = Q + t;
        auto const uvwIntersection1 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(P2, Q2, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(Q2, P2, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(P2, Q2, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(Q2, P2, A, C, B);
        do_assert(uvwIntersection4);
    }
    SUBCASE(
        "Line through PQ translated in direction perpendicular to triangle normal direction does "
        "not pass through the triangle T")
    {
        auto const do_assert = [](std::optional<SVector<ScalarType, 3>> const& intersection) {
            CHECK_FALSE(intersection.has_value());
        };
        SVector<ScalarType, 3> const t{2., 0., 0.};
        SVector<ScalarType, 3> const P2 = P + t;
        SVector<ScalarType, 3> const Q2 = Q + t;
        auto const uvwIntersection1 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(P2, Q2, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(Q2, P2, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(P2, Q2, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            pbat::geometry::IntersectionQueries::UvwLineTriangle3D(Q2, P2, A, C, B);
        do_assert(uvwIntersection4);
    }
}

TEST_CASE("[geometry] Line segments intersecting triangles are detected")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SVector<ScalarType, 3> const P{0.25, 0.25, -1.};
    SVector<ScalarType, 3> const Q{0.25, 0.25, 1.};
    SVector<ScalarType, 3> const A{0., 0., 0.};
    SVector<ScalarType, 3> const B{1., 0., 0.};
    SVector<ScalarType, 3> const C{0., 1., 0.};

    SUBCASE("Line segment PQ passes through the triangle T")
    {
        auto const do_assert = [&](std::optional<SVector<ScalarType, 4>> const& uvwIntersection) {
            CHECK(uvwIntersection.has_value());
            SVector<ScalarType, 3> const& uvw         = uvwIntersection->Slice<3, 1>(1, 0);
            SVector<ScalarType, 3> const intersection = uvw(0) * A + uvw(1) * B + uvw(2) * C;
            SVector<ScalarType, 3> const expected_intersection{0.25, 0.25, 0.};
            auto constexpr eps = 1e-15;
            CHECK(All(Abs(intersection - expected_intersection) < eps));
        };
        auto const uvwIntersection1 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(P, Q, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(Q, P, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(P, Q, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(Q, P, A, C, B);
        do_assert(uvwIntersection4);
    }
    SUBCASE("Line segment PQ does not pass through the triangle T")
    {
        auto const do_assert = [](std::optional<SVector<ScalarType, 4>> const& intersection) {
            CHECK_FALSE(intersection.has_value());
        };
        SVector<ScalarType, 3> const t{0., 0., 1.01};
        SVector<ScalarType, 3> const P2 = P + t;
        SVector<ScalarType, 3> const Q2 = Q + t;
        auto const uvwIntersection1 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(P2, Q2, A, B, C);
        do_assert(uvwIntersection1);
        auto const uvwIntersection2 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(Q2, P2, A, B, C);
        do_assert(uvwIntersection2);
        auto const uvwIntersection3 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(P2, Q2, A, C, B);
        do_assert(uvwIntersection3);
        auto const uvwIntersection4 =
            pbat::geometry::IntersectionQueries::UvwLineSegmentTriangle3D(Q2, P2, A, C, B);
        do_assert(uvwIntersection4);
    }
}
