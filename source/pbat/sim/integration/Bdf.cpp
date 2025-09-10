#include "Bdf.h"

namespace pbat::sim::integration {

} // namespace pbat::sim::integration

#include <doctest/doctest.h>

TEST_CASE("[sim][integration] Bdf")
{
    using namespace pbat;
    using namespace pbat::sim::integration;

    auto constexpr n = 5;
    VectorX f(n);
    f.setOnes();

    SUBCASE("2nd order BDF1")
    {
        // Arrange
        auto constexpr order  = 2;
        auto constexpr step   = 1;
        auto constexpr nsteps = 5;
        Scalar constexpr h    = 0.01;
        MatrixX x0(n, order);
        x0.setRandom();

        VectorX x(n);
        VectorX v(n);
        x = x0.col(0);
        v = x0.col(1);
        for (auto i = 0; i < nsteps; ++i)
        {
            v += h * f;
            x += h * v;
        }

        // Act
        VectorX xn(n);
        VectorX vn(n);
        Bdf bdf(step, order);
        bdf.SetTimeStep(h);
        bdf.SetInitialConditions(x0);
        for (auto i = 0; i < nsteps; ++i)
        {
            bdf.ConstructEquations();
            vn = bdf.BetaTilde() * f - bdf.Inertia(1);
            xn = bdf.BetaTilde() * vn - bdf.Inertia(0);
            bdf.Step(xn, vn);
        }

        // Assert
        CHECK(xn.isApprox(x));
        CHECK(vn.isApprox(v));
    }
    SUBCASE("2nd order BDF2")
    {
        // Arrange
        auto constexpr order  = 2;
        auto constexpr step   = 2;
        auto constexpr nsteps = 4;
        Scalar constexpr h    = 0.01;
        MatrixX x0(n, order);
        x0.setRandom();

        VectorX x(n), xtm1(n), xtm2(n);
        VectorX v(n), vtm1(n), vtm2(n);
        x = xtm1 = xtm2 = x0.col(0);
        v = vtm1 = vtm2 = x0.col(1);
        for (auto i = 0; i < nsteps; ++i)
        {
            v    = (Scalar(4) / 3) * vtm1 - (Scalar(1) / 3) * vtm2 + (Scalar(2) / 3) * h * f;
            x    = (Scalar(4) / 3) * xtm1 - (Scalar(1) / 3) * xtm2 + (Scalar(2) / 3) * h * v;
            vtm2 = vtm1;
            vtm1 = v;
            xtm2 = xtm1;
            xtm1 = x;
        }

        // Act
        VectorX xn(n);
        VectorX vn(n);
        Bdf bdf(step, order);
        bdf.SetTimeStep(h);
        bdf.SetInitialConditions(x0);
        for (auto i = 0; i < nsteps; ++i)
        {
            bdf.ConstructEquations();
            vn = bdf.BetaTilde() * f - bdf.Inertia(1);
            xn = bdf.BetaTilde() * vn - bdf.Inertia(0);
            bdf.Step(xn, vn);
        }

        // Assert
        CHECK(xn.isApprox(x));
        CHECK(vn.isApprox(v));
    }
}