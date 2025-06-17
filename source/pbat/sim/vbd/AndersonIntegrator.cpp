#include "AndersonIntegrator.h"

#include "pbat/common/Modulo.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/QR>
#include <algorithm>
#include <exception>

namespace pbat::sim::vbd {

AndersonIntegrator::AndersonIntegrator(Data dataIn)
    : Integrator(std::move(dataIn)),
      Fk(data.x.size()),
      Fkm1(data.x.size()),
      Gkm1(data.x.size()),
      xkm1(data.x.size()),
      DFK(data.x.size(), data.mWindowSize),
      DGK(data.x.size(), data.mWindowSize),
      alpha(data.mWindowSize)
{
}

void AndersonIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    Eigen::CompleteOrthogonalDecomposition<MatrixX> QR{};
    QR.setThreshold(1e-10);
    auto m = DGK.cols();

    xkm1 = data.x.reshaped();
    RunVbdIteration(sdt, sdt2);
    Gkm1 = data.x.reshaped();
    Fkm1 = Gkm1 - xkm1;
    for (Index k = 1; k < iterations; ++k)
    {
        // Vanilla VBD iteration
        xkm1 = data.x.reshaped();
        RunVbdIteration(sdt, sdt2);
        // Update window
        auto dkl     = common::Modulo(k - 1, m);
        auto Gk      = data.x.reshaped();
        Fk           = Gk - xkm1;
        DGK.col(dkl) = Gk - Gkm1;
        DFK.col(dkl) = Fk - Fkm1;
        Gkm1         = Gk;
        Fkm1         = Fk;
        // Anderson update
        auto mk = std::min(m, k);
        QR.compute(DFK.leftCols(mk));
        if (QR.info() != Eigen::ComputationInfo::Success)
        {
            throw std::runtime_error("QR decomposition failed");
        }
        alpha.head(mk) = QR.solve(Fk);
        Gk -= DGK.leftCols(mk) * alpha.head(mk);
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