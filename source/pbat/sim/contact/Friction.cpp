#include "Friction.h"

#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <doctest/doctest.h>

TEST_CASE("[sim][contact] PointPointTangentialBasis")
{
    using namespace pbat::sim::contact;
    namespace mini = pbat::math::linalg::mini;
    using pbat::Scalar;

    auto const fComputeOrthogonalFrame = [](Eigen::Vector<Scalar, 3> const& n, Scalar eps) {
        Eigen::Matrix<Scalar, 3, 2> T;
        auto t1 = T.col(0);
        auto t2 = T.col(1);
        if (std::abs(n.dot(Eigen::Vector<Scalar, 3>::UnitX())) >= Scalar(1) - eps)
        {
            // n is colinear with +x axis
            t1 = n.cross(Eigen::Vector<Scalar, 3>::UnitY()).normalized();
        }
        else
        {
            t1 = n.cross(Eigen::Vector<Scalar, 3>::UnitX()).normalized();
        }
        t2 = n.cross(t1);
        return T;
    };

    // We'll test a selection of relative directions: along x, generic, and near-colinear with x.
    auto test_direction = [&](Eigen::Vector<Scalar, 3> x, Eigen::Vector<Scalar, 3> y) {
        Scalar constexpr eps = 1e-6;
        auto Bopt =
            PointPointTangentialBasis(mini::FromEigen(x), mini::FromEigen(y), eps); // optimized
        // Naive implementation mirroring the commented code logic
        Eigen::Vector<Scalar, 3> xrel      = (x - y).stableNormalized();
        Eigen::Matrix<Scalar, 3, 2> Bnaive = fComputeOrthogonalFrame(xrel, eps);
        CHECK(ToEigen(Bopt).isApprox(Bnaive, Scalar(1e-6)));
    };

    test_direction(
        Eigen::Vector<Scalar, 3>{Scalar(1), Scalar(0), Scalar(0)},
        Eigen::Vector<Scalar, 3>{Scalar(0), Scalar(0), Scalar(0)});
    test_direction(
        Eigen::Vector<Scalar, 3>{Scalar(2), Scalar(1), Scalar(0)},
        Eigen::Vector<Scalar, 3>{Scalar(1), Scalar(0.2), Scalar(0)});
    test_direction(
        Eigen::Vector<Scalar, 3>{Scalar(3), Scalar(1e-9), Scalar(1e-9)},
        Eigen::Vector<Scalar, 3>{Scalar(0), Scalar(0), Scalar(0)});
}

#include "Friction.h"