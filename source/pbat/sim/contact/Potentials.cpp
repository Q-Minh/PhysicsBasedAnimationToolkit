#include "Potentials.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact][potentials] SignedDistanceQuadraticPenalty")
{
    using namespace pbat::sim::contact::potentials;
    using pbat::Scalar;
    Scalar kc  = Scalar(10);
    Scalar r   = Scalar(0.5);
    Scalar sd1 = Scalar(0.25); // inside (sd < r)
    auto dE1   = SignedDistanceQuadraticPenalty<2>(sd1, kc, r);
    Scalar rd1 = sd1 - r; // -0.25
    CHECK_EQ(dE1(0), Scalar(0.5) * kc * rd1 * rd1);
    CHECK_EQ(dE1(1), kc * rd1);
    CHECK_EQ(dE1(2), kc);

    Scalar sd2 = r; // boundary
    auto dE2   = SignedDistanceQuadraticPenalty<2>(sd2, kc, r);
    CHECK_EQ(dE2(0), Scalar(0));
    CHECK_EQ(dE2(1), Scalar(0));
    CHECK_EQ(dE2(2), kc);

    Scalar sd3 = Scalar(0.75); // outside region per doc (implementation still evaluates)
    auto dE3   = SignedDistanceQuadraticPenalty<2>(sd3, kc, r);
    Scalar rd3 = sd3 - r; // 0.25
    CHECK_EQ(dE3(0), Scalar(0.5) * kc * rd3 * rd3);
    CHECK_EQ(dE3(1), kc * rd3);
    CHECK_EQ(dE3(2), kc);
}

TEST_CASE("[sim][contact][potentials] QuadraticToLogBarrierTwoStageActivation")
{
    using namespace pbat::sim::contact::potentials;
    using pbat::Scalar;
    Scalar r   = Scalar(2);
    Scalar kc  = Scalar(10);
    Scalar tau = r / Scalar(2);
    Scalar kcp = tau * kc * (tau - r) * (tau - r); // tau*kc*(tau-r)^2
    Scalar b   = kc / Scalar(2) * (r - tau) * (r - tau) + kcp * std::log(tau);

    // Quadratic branch: d >= tau
    Scalar dquad = Scalar(1.5); // >= r/2
    auto dq      = QuadraticToLogBarrierTwoStageActivation<2>(dquad, r, kc, kcp, b);
    Scalar rd    = r - dquad; // 0.5
    CHECK_EQ(dq(0), Scalar(0.5) * kc * rd * rd);
    CHECK_EQ(dq(1), -kc * rd);
    CHECK_EQ(dq(2), kc);

    // Log barrier branch: d < tau
    Scalar dlog = Scalar(0.5); // < r/2
    auto dl     = QuadraticToLogBarrierTwoStageActivation<2>(dlog, r, kc, kcp, b);
    CHECK_LE(std::abs(dl(0) - (-kcp * std::log(dlog) + b)), Scalar(1e-12));
    CHECK_LE(std::abs(dl(1) - (-kcp / dlog)), Scalar(1e-12));
    CHECK_LE(std::abs(dl(2) - (kcp / (dlog * dlog))), Scalar(1e-12));
}

TEST_CASE("[sim][contact][potentials] ClosestPointsGradientsAndHessians")
{
    using namespace pbat::sim::contact::potentials;
    using namespace pbat::math::linalg::mini;
    using pbat::Scalar;
    // Points in 2D
    SVector<Scalar, 2> x{Scalar(1), Scalar(0)};
    SVector<Scalar, 2> y{Scalar(0), Scalar(0)};
    Scalar d     = Scalar(1); // ||x - y||
    Scalar dEdd  = Scalar(2);
    Scalar d2Edd = Scalar(5);
    // Gradient wrt closest points
    auto gxy = GradientWrtClosestPoints(x, y, d, dEdd);
    CHECK_EQ(gxy(0), Scalar(2));
    CHECK_EQ(gxy(1), Scalar(0));
    CHECK_EQ(gxy(2), Scalar(-2));
    CHECK_EQ(gxy(3), Scalar(0));
    // Segment gradient matches slices
    auto gx = GradientSegmentWrtClosestPoints(x, y, d, dEdd, 0);
    auto gy = GradientSegmentWrtClosestPoints(x, y, d, dEdd, 1);
    CHECK_EQ(gx(0), gxy(0));
    CHECK_EQ(gx(1), gxy(1));
    CHECK_EQ(gy(0), gxy(2));
    CHECK_EQ(gy(1), gxy(3));

    // Hessian wrt closest points
    auto H = HessianWrtClosestPoints(x, y, d, dEdd, d2Edd);
    // Expected block Hxx = d2Edd*gxgxT + abs(dEdd) * Norm(d2ddxx) * I
    // gx = (1,0); gxgxT = [[1,0],[0,0]]; d2ddxx = [[0,0],[0,1]]
    auto const symmetry = SquaredNorm(H - H.Transpose());
    CHECK_LE(symmetry, Scalar(1e-12));
    // Check block API
    for (auto i = 0; i < 2; ++i)
    {
        for (auto j = 0; j < 2; ++j)
        {
            auto Hij         = HessianBlockWrtClosestPoints(x, y, d, dEdd, d2Edd, i, j);
            auto const error = SquaredNorm(Hij - H.template Slice<2, 2>(i * 2, j * 2));
            CHECK_LE(error, Scalar(1e-12));
        }
    }
}

TEST_CASE("[sim][contact][potentials] LinearlyInterpolatedClosestPoints")
{
    using namespace pbat::sim::contact::potentials;
    using namespace pbat::math::linalg::mini;
    using pbat::Scalar;
    // Two vertices each side, 2D
    SVector<Scalar, 2> x{Scalar(1), Scalar(0)};
    SVector<Scalar, 2> y{Scalar(0), Scalar(0)};
    SVector<Scalar, 2> a{Scalar(0.3), Scalar(0.7)}; // weights for x's primitive
    SVector<Scalar, 2> b{Scalar(0.4), Scalar(0.6)}; // weights for y's primitive
    Scalar d     = Scalar(1);
    Scalar dEdd  = Scalar(2);
    Scalar d2Edd = Scalar(5);
    auto gxy     = GradientWrtClosestPoints(x, y, d, dEdd);
    auto guv     = GradientWrtLinearlyInterpolatedClosestPoints(a, b, x, y, d, dEdd);
    // gu part: each vertex gets weighted gx
    CHECK_EQ(guv(0), gxy(0) * a(0));
    CHECK_EQ(guv(1), gxy(1) * a(0));
    CHECK_EQ(guv(2), gxy(0) * a(1));
    CHECK_EQ(guv(3), gxy(1) * a(1));
    // gv part
    CHECK_EQ(guv(4), gxy(2) * b(0));
    CHECK_EQ(guv(5), gxy(3) * b(0));
    CHECK_EQ(guv(6), gxy(2) * b(1));
    CHECK_EQ(guv(7), gxy(3) * b(1));

    // Segment gradient (ib selects x or y)
    auto gv0_0 = GradientSegmentWrtLinearlyInterpolatedClosestPoints(a, b, x, y, d, dEdd, 0, 0);
    CHECK_EQ(gv0_0(0), gxy(0) * a(0));
    auto gv1_1 = GradientSegmentWrtLinearlyInterpolatedClosestPoints(a, b, x, y, d, dEdd, 1, 1);
    CHECK_EQ(gv1_1(0), gxy(2) * b(1));
    // Hessian wrt interpolated
    auto Huv = HessianWrtLinearlyInterpolatedClosestPoints(a, b, x, y, d, dEdd, d2Edd);
    // Hessian block API for interpolated points
    {
        auto Hblk = HessianBlockWrtLinearlyInterpolatedClosestPoints(
            a,
            b,
            x,
            y,
            d,
            dEdd,
            d2Edd,
            0,
            0,
            0,
            0);
        auto const error = SquaredNorm(Hblk - Huv.template Slice<2, 2>(0 * 2, 0 * 2));
        CHECK_LE(error, Scalar(1e-12));
    }
}