#include "DistanceQueries.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] Squared distance between axis-aligned bounding boxes can be computed")
{
    using namespace pbat;
    SUBCASE("AABBs overlap")
    {
        // Arrange
        Vector<3> const Amin{-1., -1., -1.};
        Vector<3> const Amax{1., 1., 1.};
        Vector<3> const Bmin{-0.5, -0.5, -0.5};
        Vector<3> const Bmax{1., 1., 1.};
        // Act
        Scalar const d2 =
            geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        Scalar const d2sym =
            geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        // Assert
        Scalar constexpr d2Expected = 0.;
        CHECK_EQ(d2, d2Expected);
        CHECK_EQ(d2sym, d2Expected);
    }
    SUBCASE("AABBs do not overlap in x-axis")
    {
        // Arrange
        Vector<3> const Amin{-1., -1., -1.};
        Vector<3> const Amax{1., 1., 1.};
        Vector<3> const Bmin{2., -0.5, -0.5};
        Vector<3> const Bmax{3., 1., 1.};
        // Act
        Scalar const d2 =
            geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        Scalar const d2sym =
            geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        // Assert
        Scalar constexpr d2Expected = 1.;
        CHECK_EQ(d2, d2Expected);
        CHECK_EQ(d2sym, d2Expected);
    }
    SUBCASE("AABBs overlap in all axis'")
    {
        // Arrange
        Vector<3> const Amin{-1., -1., -1.};
        Vector<3> const Amax{1., 1., 1.};
        Vector<3> const Bmin{2., 2., 2.};
        Vector<3> const Bmax{3., 3., 3.};
        // Act
        Scalar const d2 =
            geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        Scalar const d2sym =
            geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        // Assert
        Scalar const d2Expected = (Bmin - Amax).squaredNorm();
        CHECK_EQ(d2, d2Expected);
        CHECK_EQ(d2sym, d2Expected);
    }
}