#include "MeshDistance.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] Point-point distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Coincident points")
    {
        // Arrange
        SVector<ScalarType, 6> x;
        x(0) = 1.;
        x(1) = 2.;
        x(2) = 3.;
        x(3) = 1.;
        x(4) = 2.;
        x(5) = 3.;
        // Act
        ScalarType const d = PointPointDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Unit distance")
    {
        // Arrange
        SVector<ScalarType, 6> x;
        x(0) = 0.;
        x(1) = 0.;
        x(2) = 0.;
        x(3) = 1.;
        x(4) = 0.;
        x(5) = 0.;
        // Act
        ScalarType const d = PointPointDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-10));
    }
}

TEST_CASE("[geometry] Point-edge distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Point on edge")
    {
        // Arrange
        // Edge from (0,0,0) to (2,0,0), point at (1,0,0)
        SVector<ScalarType, 9> x;
        x(0) = 0.;
        x(1) = 0.;
        x(2) = 0.; // a
        x(3) = 2.;
        x(4) = 0.;
        x(5) = 0.; // b
        x(6) = 1.;
        x(7) = 0.;
        x(8) = 0.; // x
        // Act
        ScalarType const d = PointEdgeDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Point perpendicular to edge")
    {
        // Arrange
        // Edge from (0,0,0) to (2,0,0), point at (1,1,0)
        SVector<ScalarType, 9> x;
        x(0) = 0.;
        x(1) = 0.;
        x(2) = 0.; // a
        x(3) = 2.;
        x(4) = 0.;
        x(5) = 0.; // b
        x(6) = 1.;
        x(7) = 1.;
        x(8) = 0.; // x
        // Act
        ScalarType const d = PointEdgeDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-10));
    }
}

TEST_CASE("[geometry] Point-triangle distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Point on triangle plane")
    {
        // Arrange
        // Triangle with vertices at (0,0,0), (1,0,0), (0,1,0)
        // Point at (0.25, 0.25, 0)
        SVector<ScalarType, 12> x;
        x(0)  = 0.;
        x(1)  = 0.;
        x(2)  = 0.; // a
        x(3)  = 1.;
        x(4)  = 0.;
        x(5)  = 0.; // b
        x(6)  = 0.;
        x(7)  = 1.;
        x(8)  = 0.; // c
        x(9)  = 0.25;
        x(10) = 0.25;
        x(11) = 0.; // x
        // Act
        ScalarType const d = PointTriangleDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Point above triangle")
    {
        // Arrange
        // Triangle with vertices at (0,0,0), (1,0,0), (0,1,0)
        // Point at (0.25, 0.25, 1)
        SVector<ScalarType, 12> x;
        x(0)  = 0.;
        x(1)  = 0.;
        x(2)  = 0.; // a
        x(3)  = 1.;
        x(4)  = 0.;
        x(5)  = 0.; // b
        x(6)  = 0.;
        x(7)  = 1.;
        x(8)  = 0.; // c
        x(9)  = 0.25;
        x(10) = 0.25;
        x(11) = 1.; // x
        // Act
        ScalarType const d = PointTriangleDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-10));
    }
}

TEST_CASE("[geometry] Edge-edge distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Intersecting edges")
    {
        // Arrange
        // Edge 1 from (-1,0,0) to (1,0,0)
        // Edge 2 from (0,-1,0) to (0,1,0)
        SVector<ScalarType, 12> x;
        x(0)                 = -1.;
        x(1)                 = 0.;
        x(2)                 = 0.; // a
        x(3)                 = 1.;
        x(4)                 = 0.;
        x(5)                 = 0.; // b
        x(6)                 = 0.;
        x(7)                 = -1.;
        x(8)                 = 0.; // c
        x(9)                 = 0.;
        x(10)                = 1.;
        x(11)                = 0.; // d
        ScalarType const eps = 1e-6;
        // Act
        ScalarType const d = EdgeEdgeDistance<ScalarType>{}.Eval(x, eps);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-5));
    }
    SUBCASE("Parallel edges")
    {
        // Arrange
        // Edge 1 from (0,0,0) to (1,0,0)
        // Edge 2 from (0,1,0) to (1,1,0)
        SVector<ScalarType, 12> x;
        x(0)                 = 0.;
        x(1)                 = 0.;
        x(2)                 = 0.; // a
        x(3)                 = 1.;
        x(4)                 = 0.;
        x(5)                 = 0.; // b
        x(6)                 = 0.;
        x(7)                 = 1.;
        x(8)                 = 0.; // c
        x(9)                 = 1.;
        x(10)                = 1.;
        x(11)                = 0.; // d
        ScalarType const eps = 1e-6;
        // Act
        ScalarType const d = EdgeEdgeDistance<ScalarType>{}.Eval(x, eps);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-5));
    }
}
