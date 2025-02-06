#include "OverlapQueries.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] Point overlap with tetrahedron can be obtained")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    // Arrange
    SVector<ScalarType, 3> const A{0., 0., 0.};
    SVector<ScalarType, 3> const B{1., 0., 0.};
    SVector<ScalarType, 3> const C{0., 1., 0.};
    SVector<ScalarType, 3> const D{0., 0., 1.};

    SUBCASE("Point is in tetrahedron")
    {
        SVector<ScalarType, 3> const P{0.2, 0.2, 0.2};
        // Act
        bool const PContainedInABCD =
            pbat::geometry::OverlapQueries::PointTetrahedron3D(P, A, B, C, D);
        // Assert
        CHECK(PContainedInABCD);
    }
    SUBCASE("Point is not in tetrahedron")
    {
        SVector<ScalarType, 3> const P{1., 1., 1.};
        // Act
        bool const PContainedInABCD =
            pbat::geometry::OverlapQueries::PointTetrahedron3D(P, A, B, C, D);
        // Assert
        CHECK_FALSE(PContainedInABCD);
    }
}

TEST_CASE("[geometry] Sphere overlap predicate can be obtained")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Spheres are overlapping")
    {
        SVector<ScalarType, 3> const c1{0.0, 0.0, 0.0};
        ScalarType const r1 = 1.;
        SVector<ScalarType, 3> const c2{2.0, 0.0, 0.0};
        ScalarType const r2  = 1.;
        bool const bOverlaps = pbat::geometry::OverlapQueries::Spheres(c1, r1, c2, r2);
        CHECK(bOverlaps);
    }
    SUBCASE("Spheres are not overlapping")
    {
        SVector<ScalarType, 3> const c1{0.0, 0.0, 0.0};
        ScalarType const r1 = 1.;
        SVector<ScalarType, 3> const c2{4.0, 0.0, 0.0};
        ScalarType const r2  = 1.;
        bool const bOverlaps = pbat::geometry::OverlapQueries::Spheres(c1, r1, c2, r2);
        CHECK_FALSE(bOverlaps);
    }
}

TEST_CASE("[geometry] AABB overlap predicate can be obtained")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("AABBs are overlapping")
    {
        SVector<ScalarType, 3> const min1{0.0, 0.0, 0.0};
        SVector<ScalarType, 3> const max1{2.0, 2.0, 2.0};
        SVector<ScalarType, 3> const min2{1.0, 1.0, 1.0};
        SVector<ScalarType, 3> const max2{3.0, 3.0, 3.0};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::AxisAlignedBoundingBoxes(min1, max1, min2, max2);
        CHECK(bOverlaps);
    }
    SUBCASE("AABBs are not overlapping")
    {
        SVector<ScalarType, 3> const min1{0.0, 0.0, 0.0};
        SVector<ScalarType, 3> const max1{1.0, 1.0, 1.0};
        SVector<ScalarType, 3> const min2{2.0, 2.0, 2.0};
        SVector<ScalarType, 3> const max2{3.0, 3.0, 3.0};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::AxisAlignedBoundingBoxes(min1, max1, min2, max2);
        CHECK_FALSE(bOverlaps);
    }
}

TEST_CASE("[geometry] AABB against sphere overlap predicate can be obtained")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("AABB and sphere are overlapping")
    {
        SVector<ScalarType, 3> const C{0.5, 0.5, 0.5};
        ScalarType const r = 1.;
        SVector<ScalarType, 3> const min{0., 0., 0.};
        SVector<ScalarType, 3> const max{1., 1., 1.};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::SphereAxisAlignedBoundingBox(C, r, min, max);
        CHECK(bOverlaps);
    }
    SUBCASE("AABB and sphere are not overlapping")
    {
        SVector<ScalarType, 3> const C{2., 2., 2.};
        ScalarType const r = 1.;
        SVector<ScalarType, 3> const min{0., 0., 0.};
        SVector<ScalarType, 3> const max{1., 1., 1.};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::SphereAxisAlignedBoundingBox(C, r, min, max);
        CHECK_FALSE(bOverlaps);
    }
}

TEST_CASE("[geometry] Line segment swept triangle overlap predicate can be obtained")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Line segment intersects swept triangle")
    {
        SUBCASE("Swept triangle is non-degenerate")
        {
            SVector<ScalarType, 3> const A1{0., 0., 0.};
            SVector<ScalarType, 3> const B1{1., 0., 0.};
            SVector<ScalarType, 3> const C1{0., 1., 0.};
            SVector<ScalarType, 3> const A2{0., 0., 1.};
            SVector<ScalarType, 3> const B2{1., 0., 1.};
            SVector<ScalarType, 3> const C2{0., 1., 1.};
            SVector<ScalarType, 3> const P{0.2, 0.2, -1.};
            SVector<ScalarType, 3> const Q{0.2, 0.2, 1.};
            bool const bOverlaps = pbat::geometry::OverlapQueries::LineSegmentSweptTriangle3D(
                P,
                Q,
                A1,
                B1,
                C1,
                A2,
                B2,
                C2);
            CHECK(bOverlaps);
        }
        SUBCASE("Swept triangle is non-degenerate and inverted")
        {
            SVector<ScalarType, 3> const A1{0., 0., 0.};
            SVector<ScalarType, 3> const B1{1., 0., 0.};
            SVector<ScalarType, 3> const C1{0., 1., 0.};
            SVector<ScalarType, 3> const A2{0., 0., -1.};
            SVector<ScalarType, 3> const B2{1., 0., -1.};
            SVector<ScalarType, 3> const C2{0., 1., -1.};
            SVector<ScalarType, 3> const P{0.2, 0.2, -1.};
            SVector<ScalarType, 3> const Q{0.2, 0.2, 1.};
            bool const bOverlaps = pbat::geometry::OverlapQueries::LineSegmentSweptTriangle3D(
                P,
                Q,
                A1,
                B1,
                C1,
                A2,
                B2,
                C2);
            CHECK(bOverlaps);
        }
        SUBCASE("Swept triangle is degenerate")
        {
            SVector<ScalarType, 3> const A1{0., 0., 0.};
            SVector<ScalarType, 3> const B1{1., 0., 0.};
            SVector<ScalarType, 3> const C1{0., 1., 0.};
            SVector<ScalarType, 3> const P{0.2, 0.2, -1.};
            SVector<ScalarType, 3> const Q{0.2, 0.2, 1.};
            bool const bOverlaps = pbat::geometry::OverlapQueries::LineSegmentSweptTriangle3D(
                P,
                Q,
                A1,
                B1,
                C1,
                A1,
                B1,
                C1);
            CHECK(bOverlaps);
        }
    }
    SUBCASE("Line segment does not intersect non-degenerate triangle")
    {
        SUBCASE("Swept triangle is non-degenerate")
        {
            SVector<ScalarType, 3> const A1{0., 0., 0.};
            SVector<ScalarType, 3> const B1{1., 0., 0.};
            SVector<ScalarType, 3> const C1{0., 1., 0.};
            SVector<ScalarType, 3> const A2{0., 0., 1.};
            SVector<ScalarType, 3> const B2{1., 0., 1.};
            SVector<ScalarType, 3> const C2{0., 1., 1.};
            SVector<ScalarType, 3> const P{2., 2., -1.};
            SVector<ScalarType, 3> const Q{2., 2., 1.};
            bool const bOverlaps = pbat::geometry::OverlapQueries::LineSegmentSweptTriangle3D(
                P,
                Q,
                A1,
                B1,
                C1,
                A2,
                B2,
                C2);
            CHECK_FALSE(bOverlaps);
        }
        SUBCASE("Swept triangle is non-degenerate and inverted")
        {
            SVector<ScalarType, 3> const A1{0., 0., 0.};
            SVector<ScalarType, 3> const B1{1., 0., 0.};
            SVector<ScalarType, 3> const C1{0., 1., 0.};
            SVector<ScalarType, 3> const A2{0., 0., -1.};
            SVector<ScalarType, 3> const B2{1., 0., -1.};
            SVector<ScalarType, 3> const C2{0., 1., -1.};
            SVector<ScalarType, 3> const P{2., 2., -1.};
            SVector<ScalarType, 3> const Q{2., 2., 0.};
            bool const bOverlaps = pbat::geometry::OverlapQueries::LineSegmentSweptTriangle3D(
                P,
                Q,
                A1,
                B1,
                C1,
                A2,
                B2,
                C2);
            CHECK_FALSE(bOverlaps);
        }
        SUBCASE("Swept triangle is degenerate")
        {
            SVector<ScalarType, 3> const A1{0., 0., 0.};
            SVector<ScalarType, 3> const B1{1., 0., 0.};
            SVector<ScalarType, 3> const C1{0., 1., 0.};
            SVector<ScalarType, 3> const P{2., 2., -1.};
            SVector<ScalarType, 3> const Q{2., 2., 1.};
            bool const bOverlaps = pbat::geometry::OverlapQueries::LineSegmentSweptTriangle3D(
                P,
                Q,
                A1,
                B1,
                C1,
                A1,
                B1,
                C1);
            CHECK_FALSE(bOverlaps);
        }
    }
}

TEST_CASE("[geometry] Plane against AABB overlap predicate can be obtained")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Plane and AABB are overlapping")
    {
        SVector<ScalarType, 3> const P{0., 0., 0.};
        SVector<ScalarType, 3> const n{0., 0., 1.};
        SVector<ScalarType, 3> const min{-1., -1., -1.};
        SVector<ScalarType, 3> const max{1., 1., 1.};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::PlaneAxisAlignedBoundingBox(P, n, min, max);
        CHECK(bOverlaps);
    }
    SUBCASE("Plane and AABB are not overlapping")
    {
        SVector<ScalarType, 3> const P{0., 0., 0.};
        SVector<ScalarType, 3> const n{0., 0., 1.};
        SVector<ScalarType, 3> const min{2., 2., 2.};
        SVector<ScalarType, 3> const max{3., 3., 3.};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::PlaneAxisAlignedBoundingBox(P, n, min, max);
        CHECK_FALSE(bOverlaps);
    }
}

TEST_CASE("[geometry] Can detect overlap between triangle and AABB")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SVector<ScalarType, 3> const min{0., 0., 0.};
    SVector<ScalarType, 3> const max{1., 1., 1.};

    SUBCASE("Triangle is on AABB face")
    {
        SVector<ScalarType, 3> const A{0., 0., 0.};
        SVector<ScalarType, 3> const B{1., 0., 0.};
        SVector<ScalarType, 3> const C{0., 1., 0.};
        bool const bIntersects =
            pbat::geometry::OverlapQueries::TriangleAxisAlignedBoundingBox(A, B, C, min, max);
        CHECK(bIntersects);
    }
    SUBCASE("Triangle is in AABB")
    {
        SVector<ScalarType, 3> const A{0.1, 0.1, 0.1};
        SVector<ScalarType, 3> const B{.5, 0.1, 0.1};
        SVector<ScalarType, 3> const C{0.1, .5, 0.1};
        bool const bIntersects =
            pbat::geometry::OverlapQueries::TriangleAxisAlignedBoundingBox(A, B, C, min, max);
        CHECK(bIntersects);
    }
    SUBCASE("Triangle partially passes through AABB")
    {
        SVector<ScalarType, 3> const A{0.1, 0., 0.};
        SVector<ScalarType, 3> const B{2., 0., 0.};
        SVector<ScalarType, 3> const C{0.1, 2., 0.};
        bool const bIntersects =
            pbat::geometry::OverlapQueries::TriangleAxisAlignedBoundingBox(A, B, C, min, max);
        CHECK(bIntersects);
    }
    SUBCASE("Triangle is outside of AABB")
    {
        SVector<ScalarType, 3> const A{1.1, 0., 0.};
        SVector<ScalarType, 3> const B{2., 0., 0.};
        SVector<ScalarType, 3> const C{1.1, 2., 0.};
        bool const bIntersects =
            pbat::geometry::OverlapQueries::TriangleAxisAlignedBoundingBox(A, B, C, min, max);
        CHECK_FALSE(bIntersects);
    }
}

TEST_CASE("[geometry] Can detect overlap between tetrahedron and AABB")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Non-overlapping tetrahedron and AABB")
    {
        SVector<ScalarType, 3> const A{0., 0., 0.};
        SVector<ScalarType, 3> const B{1., 0., 0.};
        SVector<ScalarType, 3> const C{0., 1., 0.};
        SVector<ScalarType, 3> const D{0., 0., 1.};
        SVector<ScalarType, 3> const min{2., 2., 2.};
        SVector<ScalarType, 3> const max{3., 3., 3.};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TetrahedronAxisAlignedBoundingBox(A, B, C, D, min, max);
        CHECK_FALSE(bOverlaps);
    }
    SUBCASE("Tetrahedron entirely contained in AABB")
    {
        SVector<ScalarType, 3> const A{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const D{0.2, 0.2, 0.3};
        SVector<ScalarType, 3> const min{0.1, 0.1, 0.1};
        SVector<ScalarType, 3> const max{0.4, 0.4, 0.4};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TetrahedronAxisAlignedBoundingBox(A, B, C, D, min, max);
        CHECK(bOverlaps);
    }
    SUBCASE("Tetrahedron partially contained in AABB")
    {
        SVector<ScalarType, 3> const A{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const D{0.2, 0.2, 0.3};
        SVector<ScalarType, 3> const min{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const max{0.3, 0.3, 0.3};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TetrahedronAxisAlignedBoundingBox(A, B, C, D, min, max);
        CHECK(bOverlaps);
    }
}

TEST_CASE("[geometry] Can detect overlap between triangle and tetrahedron")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Non-overlapping tetrahedron and triangle")
    {
        SVector<ScalarType, 3> const A1{0., 0., 0.};
        SVector<ScalarType, 3> const B1{1., 0., 0.};
        SVector<ScalarType, 3> const C1{0., 1., 0.};
        SVector<ScalarType, 3> const A2{1., 1., 1.};
        SVector<ScalarType, 3> const B2{2., 1., 1.};
        SVector<ScalarType, 3> const C2{1., 2., 1.};
        SVector<ScalarType, 3> const D2{1., 1., 2.};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TriangleTetrahedron(A1, B1, C1, A2, B2, C2, D2);
        CHECK_FALSE(bOverlaps);
    }
    SUBCASE("Triangle entirely contained in tetrahedron")
    {
        SVector<ScalarType, 3> const A1{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B1{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C1{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const A2{0.0, 0.0, 0.0};
        SVector<ScalarType, 3> const B2{1.0, 0.0, 0.0};
        SVector<ScalarType, 3> const C2{0.0, 1.0, 0.0};
        SVector<ScalarType, 3> const D2{0.0, 0.0, 1.0};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TriangleTetrahedron(A1, B1, C1, A2, B2, C2, D2);
        CHECK(bOverlaps);
    }
    SUBCASE("Triangle partially contained in tetrahedron")
    {
        SVector<ScalarType, 3> const A1{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B1{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C1{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const A2{0.0, 0.0, 0.0};
        SVector<ScalarType, 3> const B2{1.0, 0.0, 0.0};
        SVector<ScalarType, 3> const C2{0.0, 1.0, 0.0};
        SVector<ScalarType, 3> const D2{1.0, 1.0, 1.0};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TriangleTetrahedron(A1, B1, C1, A2, B2, C2, D2);
        CHECK(bOverlaps);
    }
}

TEST_CASE("[geometry] Can detect overlap between tetrahedron and tetrahedron")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Non-overlapping tetrahedra")
    {
        SVector<ScalarType, 3> const A1{0., 0., 0.};
        SVector<ScalarType, 3> const B1{1., 0., 0.};
        SVector<ScalarType, 3> const C1{0., 1., 0.};
        SVector<ScalarType, 3> const D1{0., 0., 1.};
        SVector<ScalarType, 3> const A2{2., 2., 2.};
        SVector<ScalarType, 3> const B2{3., 2., 2.};
        SVector<ScalarType, 3> const C2{2., 3., 2.};
        SVector<ScalarType, 3> const D2{2., 2., 3.};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::Tetrahedra(A1, B1, C1, D1, A2, B2, C2, D2);
        CHECK_FALSE(bOverlaps);
    }
    SUBCASE("One tetrahedron entirely contained in the other")
    {
        SVector<ScalarType, 3> const A1{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B1{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C1{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const D1{0.2, 0.2, 0.3};
        SVector<ScalarType, 3> const A2{0.1, 0.1, 0.1};
        SVector<ScalarType, 3> const B2{0.4, 0.1, 0.1};
        SVector<ScalarType, 3> const C2{0.1, 0.4, 0.1};
        SVector<ScalarType, 3> const D2{0.1, 0.1, 0.4};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::Tetrahedra(A1, B1, C1, D1, A2, B2, C2, D2);
        CHECK(bOverlaps);
    }
    SUBCASE("One tetrahedron partially contained in the other")
    {
        SVector<ScalarType, 3> const A1{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B1{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C1{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const D1{0.2, 0.2, 0.3};
        SVector<ScalarType, 3> const A2{0.0, 0.0, 0.0};
        SVector<ScalarType, 3> const B2{1.0, 0.0, 0.0};
        SVector<ScalarType, 3> const C2{0.0, 1.0, 0.0};
        SVector<ScalarType, 3> const D2{0.0, 0.0, 1.0};
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::Tetrahedra(A1, B1, C1, D1, A2, B2, C2, D2);
        CHECK(bOverlaps);
    }
}

TEST_CASE("[geometry] Can detect overlap between triangle and sphere")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Non-overlapping triangle and sphere")
    {
        SVector<ScalarType, 3> const A{0., 0., 0.};
        SVector<ScalarType, 3> const B{1., 0., 0.};
        SVector<ScalarType, 3> const C{0., 1., 0.};
        SVector<ScalarType, 3> const center{2., 2., 2.};
        ScalarType const radius = 1.;
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TriangleSphere(A, B, C, center, radius);
        CHECK_FALSE(bOverlaps);
    }
    SUBCASE("Triangle entirely contained in sphere")
    {
        SVector<ScalarType, 3> const A{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const center{0.25, 0.25, 0.25};
        ScalarType const radius = 1.;
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TriangleSphere(A, B, C, center, radius);
        CHECK(bOverlaps);
    }
    SUBCASE("Triangle partially contained in sphere")
    {
        SVector<ScalarType, 3> const A{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const center{0.1, 0.1, 0.1};
        ScalarType const radius = 1.;
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TriangleSphere(A, B, C, center, radius);
        CHECK(bOverlaps);
    }
}

TEST_CASE("[geometry] Can detect overlap between tetrahedron and sphere")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("Non-overlapping tetrahedron and sphere")
    {
        SVector<ScalarType, 3> const A{0., 0., 0.};
        SVector<ScalarType, 3> const B{1., 0., 0.};
        SVector<ScalarType, 3> const C{0., 1., 0.};
        SVector<ScalarType, 3> const D{0., 0., 1.};
        SVector<ScalarType, 3> const center{2., 2., 2.};
        ScalarType const radius = 1.;
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TetrahedronSphere(A, B, C, D, center, radius);
        CHECK_FALSE(bOverlaps);
    }

    SUBCASE("Tetrahedron entirely contained in sphere")
    {
        SVector<ScalarType, 3> const A{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const D{0.2, 0.2, 0.3};
        SVector<ScalarType, 3> const center{0.25, 0.25, 0.25};
        ScalarType const radius = 1.;
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TetrahedronSphere(A, B, C, D, center, radius);
        CHECK(bOverlaps);
    }

    SUBCASE("Tetrahedron partially contained in sphere")
    {
        SVector<ScalarType, 3> const A{0.2, 0.2, 0.2};
        SVector<ScalarType, 3> const B{0.3, 0.2, 0.2};
        SVector<ScalarType, 3> const C{0.2, 0.3, 0.2};
        SVector<ScalarType, 3> const D{0.2, 0.2, 0.3};
        SVector<ScalarType, 3> const center{0.1, 0.1, 0.1};
        ScalarType const radius = 1.;
        bool const bOverlaps =
            pbat::geometry::OverlapQueries::TetrahedronSphere(A, B, C, D, center, radius);
        CHECK(bOverlaps);
    }
}
