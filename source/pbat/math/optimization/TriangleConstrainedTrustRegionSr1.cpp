#include "TriangleConstrainedTrustRegionSr1.h"

#include <doctest/doctest.h>

namespace pbat::math::optimization {

TEST_CASE("[math][optimization] TriangleConstrainedTrustRegionSr1 scenarios")
{
    namespace mini = pbat::math::linalg::mini;
    using Scalar   = double;
    TriangleConstrainedTrustRegionSr1Params<Scalar> params{};
    params.R0        = Scalar(0.25);
    params.eta       = Scalar(1e-4);
    params.trlo      = Scalar(0.1);
    params.trhi      = Scalar(0.9);
    params.trbound   = Scalar(0.9);
    params.trgrow    = Scalar(2.0);
    params.trshrink  = Scalar(0.5);
    params.sigmaB    = Scalar(1.0);
    params.deltaf    = Scalar(1e-12);
    params.deltas    = Scalar(1e-12);
    params.nMaxIters = 50;
    params.gzero     = Scalar(1e-8);
    SUBCASE("Center inside triangle converges to center")
    {
        mini::SVector<Scalar, 2> xcenter{Scalar(0.2), Scalar(0.3)};
        auto f = [&](mini::SVector<Scalar, 2> const& x) {
            return Scalar(0.5) * mini::SquaredNorm(x - xcenter);
        };
        auto gradf = [&](mini::SVector<Scalar, 2> const& x) {
            return x - xcenter;
        };
        mini::SVector<Scalar, 2> xk{Scalar(0.8), Scalar(0.1)};
        bool converged = TriangleConstrainedTrustRegionSr1(f, gradf, xk, params);
        CHECK(converged);
        CHECK_LE(mini::Norm(xk - xcenter), Scalar(1e-6));
    }
    SUBCASE("Center outside (sum>1) projects onto edge")
    {
        mini::SVector<Scalar, 2> xcenter{Scalar(0.7), Scalar(0.7)};
        auto f = [&](mini::SVector<Scalar, 2> const& x) {
            return Scalar(0.5) * mini::SquaredNorm(x - xcenter);
        };
        auto gradf = [&](mini::SVector<Scalar, 2> const& x) {
            return x - xcenter;
        };
        mini::SVector<Scalar, 2> xk{Scalar(0.2), Scalar(0.2)};
        bool converged = TriangleConstrainedTrustRegionSr1(f, gradf, xk, params);
        CHECK(converged);
        mini::SVector<Scalar, 2> xproj{Scalar(0.5), Scalar(0.5)};
        CHECK_LE(mini::Norm(xk - xproj), Scalar(1e-6));
    }
    SUBCASE("Center outside near vertex projects onto vertex")
    {
        mini::SVector<Scalar, 2> xcenter{Scalar(2.0), Scalar(0.05)};
        auto f = [&](mini::SVector<Scalar, 2> const& x) {
            return Scalar(0.5) * mini::SquaredNorm(x - xcenter);
        };
        auto gradf = [&](mini::SVector<Scalar, 2> const& x) {
            return x - xcenter;
        };
        mini::SVector<Scalar, 2> xk{Scalar(0.3), Scalar(0.3)};
        bool converged = TriangleConstrainedTrustRegionSr1(f, gradf, xk, params);
        CHECK(converged);
        mini::SVector<Scalar, 2> xproj{Scalar(1.0), Scalar(0.0)};
        CHECK_LE(mini::Norm(xk - xproj), Scalar(1e-6));
    }
}

} // namespace pbat::math::optimization