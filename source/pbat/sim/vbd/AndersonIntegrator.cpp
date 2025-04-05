#include "AndersonIntegrator.h"

#include "pbat/profiling/Profiling.h"

#include <Eigen/QR>
#include <algorithm>
#include <exception>

namespace pbat::sim::vbd {

AndersonIntegrator::AndersonIntegrator(Data dataIn)
    : Integrator(std::move(dataIn)),
      FK(data.x.size(), data.mAndersonWindowSize),
      GK(data.x.size(), data.mAndersonWindowSize),
      DFK(data.x.size(), data.mAndersonWindowSize),
      DGK(data.x.size(), data.mAndersonWindowSize),
      xkm1(data.x.rows(), data.x.cols()),
      alpha(data.mAndersonWindowSize)
{
}

void AndersonIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    auto const mod = [](auto a, auto b) {
        return (a % b + b) % b;
    };
    Eigen::CompleteOrthogonalDecomposition<MatrixX> QR{};
    QR.setThreshold(1e-10);

    auto m = GK.cols();
    xkm1   = data.x;
    RunVbdIteration(sdt, sdt2);
    FK.col(0) = data.x.reshaped() - xkm1.reshaped();
    GK.col(0) = data.x.reshaped();
    for (Index k = 1; k < iterations; ++k)
    {
        // Vanilla VBD iteration
        xkm1 = data.x;
        RunVbdIteration(sdt, sdt2);
        // Update window
        auto kl      = mod(k, m);
        auto dkl     = mod(k - 1, m);
        GK.col(kl)   = data.x.reshaped();
        FK.col(kl)   = GK.col(kl) - xkm1.reshaped();
        DGK.col(dkl) = GK.col(kl) - GK.col(dkl);
        DFK.col(dkl) = FK.col(kl) - FK.col(dkl);
        // Anderson Update
        auto mk = std::min(m, k);
        QR.compute(DFK.leftCols(mk));
        if (QR.info() != Eigen::ComputationInfo::Success)
        {
            throw std::runtime_error("QR decomposition failed");
        }
        alpha.segment(0, mk) = QR.solve(FK.col(kl));
        auto x               = data.x.reshaped();
        x                    = GK.col(kl);
        for (auto j = 0; j < mk; ++j)
        {
            auto dklj = mod(dkl - j, m);
            x -= alpha(dklj) * DGK.col(dklj);
        }
    }
}

} // namespace pbat::sim::vbd

#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd] AndersonIntegrator")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    MatrixX P(3, 8);
    IndexMatrixX V(1, 8);
    IndexMatrixX T(4, 5);
    IndexMatrixX F(3, 12);
    // clang-format off
    P << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
         1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
         4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // clang-format on
    V.reshaped().setLinSpaced(0, static_cast<Index>(P.cols() - 1));
    // Problem parameters
    auto constexpr dt         = Scalar{1e-2};
    auto constexpr substeps   = 1;
    auto constexpr iterations = 10;
    Index constexpr m         = 5;
    using pbat::common::ToEigen;
    using pbat::sim::vbd::AndersonIntegrator;
    AndersonIntegrator avbd{sim::vbd::Data()
                                .WithVolumeMesh(P, T)
                                .WithSurfaceMesh(V, F)
                                .WithAndersonAcceleration(m)
                                .Construct()};
    MatrixX xtilde = avbd.data.x + dt * avbd.data.v + dt * dt * avbd.data.aext;
    Scalar f0      = avbd.ObjectiveFunction(avbd.data.x, xtilde, dt);
    VectorX g0     = avbd.ObjectiveFunctionGradient(avbd.data.x, xtilde, dt);

    // Act
    avbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = avbd.data.x - P;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
    VectorX g    = avbd.ObjectiveFunctionGradient(avbd.data.x, xtilde, dt);
    Scalar gnorm = g.norm();
    CHECK_LT(gnorm, zero);
    Scalar f = avbd.ObjectiveFunction(avbd.data.x, xtilde, dt);
    CHECK_LT(f, f0);
}