#include "DistanceQueries.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] Squared distance between axis-aligned bounding boxes can be computed")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    SUBCASE("AABBs overlap")
    {
        // Arrange
        SVector<ScalarType, 3> const Amin{-1., -1., -1.};
        SVector<ScalarType, 3> const Amax{1., 1., 1.};
        SVector<ScalarType, 3> const Bmin{-0.5, -0.5, -0.5};
        SVector<ScalarType, 3> const Bmax{1., 1., 1.};
        // Act
        ScalarType const d2 =
            pbat::geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        ScalarType const d2sym =
            pbat::geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        // Assert
        ScalarType constexpr d2Expected = 0.;
        CHECK_EQ(d2, d2Expected);
        CHECK_EQ(d2sym, d2Expected);
    }
    SUBCASE("AABBs do not overlap in x-axis")
    {
        // Arrange
        SVector<ScalarType, 3> const Amin{-1., -1., -1.};
        SVector<ScalarType, 3> const Amax{1., 1., 1.};
        SVector<ScalarType, 3> const Bmin{2., -0.5, -0.5};
        SVector<ScalarType, 3> const Bmax{3., 1., 1.};
        // Act
        ScalarType const d2 =
            pbat::geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        ScalarType const d2sym =
            pbat::geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        // Assert
        ScalarType constexpr d2Expected = 1.;
        CHECK_EQ(d2, d2Expected);
        CHECK_EQ(d2sym, d2Expected);
    }
    SUBCASE("AABBs overlap in all axis'")
    {
        // Arrange
        SVector<ScalarType, 3> const Amin{-1., -1., -1.};
        SVector<ScalarType, 3> const Amax{1., 1., 1.};
        SVector<ScalarType, 3> const Bmin{2., 2., 2.};
        SVector<ScalarType, 3> const Bmax{3., 3., 3.};
        // Act
        ScalarType const d2 =
            pbat::geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        ScalarType const d2sym =
            pbat::geometry::DistanceQueries::AxisAlignedBoundingBoxes(Amin, Amax, Bmin, Bmax);
        // Assert
        ScalarType const d2Expected = SquaredNorm(Bmin - Amax);
        CHECK_EQ(d2, d2Expected);
        CHECK_EQ(d2sym, d2Expected);
    }
}