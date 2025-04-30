#include "Bdf.h"

#include <algorithm>
#include <exception>

namespace pbat::sim::integration {

Bdf::Bdf(int step, int order)
    : xt(), xtilde(), ti(0), h(Scalar(0.01)), mOrder(), mStep(), mAlpha(), mBeta()
{
    SetStep(step);
    SetOrder(order);
}

auto Bdf::State(int k, int o) const
{
    if (k < 0 || k > mStep)
    {
        throw std::out_of_range("0 <= k <= s");
    }
    if (o < 0 || o >= mOrder)
    {
        throw std::out_of_range("0 <= o < order");
    }
    auto kt = common::Modulo(ti /*- mStep*/ + k, mStep);
    return xt.col(o * mStep + kt);
}

auto Bdf::State(int k, int o)
{
    if (k < 0 || k > mStep)
    {
        throw std::out_of_range("0 <= k <= s");
    }
    if (o < 0 || o >= mOrder)
    {
        throw std::out_of_range("0 <= o < order");
    }
    auto kt = common::Modulo(ti /*- mStep*/ + k, mStep);
    return xt.col(o * mStep + kt);
}

auto Bdf::CurrentState(int o) const
{
    return State(mStep - 1, o);
}

auto Bdf::CurrentState(int o)
{
    return State(mStep - 1, o);
}

void Bdf::SetOrder(int order)
{
    if (order <= 0)
    {
        throw std::invalid_argument("order > 0");
    }
    mOrder = order;
}

void Bdf::SetStep(int step)
{
    if (step < 1 || step > 6)
    {
        throw std::invalid_argument("0 < s < 7.");
    }
    mStep = step;
    mAlpha.setZero();
    switch (mStep)
    {
        case 1:
            mAlpha(0) = Scalar(-1);
            mBeta     = Scalar(1);
            break;
        case 2:
            mAlpha(0) = Scalar(1) / 3;
            mAlpha(1) = Scalar(-4) / 3;
            mBeta     = Scalar(2) / 3;
            break;
        case 3:
            mAlpha(0) = Scalar(-2) / 11;
            mAlpha(1) = Scalar(9) / 11;
            mAlpha(2) = Scalar(-18) / 11;
            mBeta     = Scalar(6) / 11;
            break;
        case 4:
            mAlpha(0) = Scalar(3) / 25;
            mAlpha(1) = Scalar(-16) / 25;
            mAlpha(2) = Scalar(36) / 25;
            mAlpha(3) = Scalar(-48) / 25;
            mBeta     = Scalar(12) / 25;
            break;
        case 5:
            mAlpha(0) = Scalar(-12) / 137;
            mAlpha(1) = Scalar(75) / 137;
            mAlpha(2) = Scalar(-200) / 137;
            mAlpha(3) = Scalar(300) / 137;
            mAlpha(4) = Scalar(-300) / 137;
            mBeta     = Scalar(60) / 137;
            break;
        case 6:
            mAlpha(0) = Scalar(10) / 147;
            mAlpha(1) = Scalar(-72) / 147;
            mAlpha(2) = Scalar(225) / 147;
            mAlpha(3) = Scalar(-400) / 147;
            mAlpha(4) = Scalar(450) / 147;
            mAlpha(5) = Scalar(-360) / 147;
            mBeta     = Scalar(60) / 147;
            break;
    }
}

void Bdf::SetTimeStep(Scalar dt)
{
    if (dt <= 0)
    {
        throw std::invalid_argument("dt > 0");
    }
    h = dt;
}

void Bdf::ConstructEquations()
{
    xtilde.setZero();
    for (auto o = 0; o < mOrder; ++o)
        for (auto k = 0; k < mStep; ++k)
            xtilde.col(o) += mAlpha(k) * State(k, o);
}

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