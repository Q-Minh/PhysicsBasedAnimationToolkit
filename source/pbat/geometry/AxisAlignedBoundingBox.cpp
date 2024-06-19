#include "AxisAlignedBoundingBox.h"

#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>

TEST_CASE("[geometry] AxisAlignedBoundingBox")
{
    using namespace pbat;
    common::ForValues<1, 2, 3>([]<auto Dims>() {
        auto constexpr N        = 10u;
        Matrix<Dims, N> const P = Matrix<Dims, N>::Random();
        Vector<Dims> const min  = P.rowwise().minCoeff();
        Vector<Dims> const max  = P.rowwise().maxCoeff();
        geometry::AxisAlignedBoundingBox<Dims> const aabb(P);
        Scalar const minError = (aabb.min() - min).squaredNorm();
        Scalar const maxError = (aabb.max() - max).squaredNorm();
        auto constexpr zero   = 1e-15;
        CHECK_LE(minError, zero);
        CHECK_LE(maxError, zero);
    });
}