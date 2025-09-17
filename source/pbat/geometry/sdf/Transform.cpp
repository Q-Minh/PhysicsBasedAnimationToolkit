#include "Transform.h"

#include "pbat/math/linalg/mini/Eigen.h"

#include <Eigen/Geometry>
#include <doctest/doctest.h>
#include <numbers>

TEST_CASE("[geometry][sdf] Transform")
{
    using namespace pbat::math::linalg::mini;
    using ScalarType = pbat::Scalar;
    using namespace pbat::geometry::sdf;
    SUBCASE("A point can be transformed")
    {
        // Arrange
        SVector<ScalarType, 3> const p = FromEigen(Eigen::Vector<ScalarType, 3>::Random());
        SVector<ScalarType, 3> const t = FromEigen(Eigen::Vector<ScalarType, 3>::Random());
        Eigen::AngleAxis<ScalarType> v(
            std::numbers::pi_v<ScalarType> / 2,
            Eigen::Vector3d::UnitZ());
        SMatrix<ScalarType, 3, 3> const R = FromEigen(v.toRotationMatrix());
        Transform<ScalarType> const T{R, t};
        // Act
        SVector<ScalarType, 3> const Rpt = T(p);
        SVector<ScalarType, 3> const pp  = T / Rpt;
        // Assert
        bool const bCanTransform       = not ToEigen(Rpt).isApprox(ToEigen(p));
        bool const bCanInvertTransform = ToEigen(pp).isApprox(ToEigen(p));
        CHECK(bCanTransform);
        CHECK(bCanInvertTransform);
    }
}