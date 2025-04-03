#include "NesterovIntegrator.h"

#include "pbat/profiling/Profiling.h"

#include <cmath>

namespace pbat::sim::vbd {

NesterovIntegrator::NesterovIntegrator(Data dataIn)
    : Integrator(std::move(dataIn)),
      xkm1(data.x.rows(), data.x.cols()),
      yk(data.x.rows(), data.x.cols()),
      L(data.mNesterovLipschitzConstant),
      start(data.mNesterovAccelerationStart)
{
}

void NesterovIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.NesterovIntegrator.Solve");
    xkm1                  = data.x;
    Scalar alpha          = 1. / L;
    Scalar lambda         = 0.;
    Scalar beta           = 0.;
    Scalar constexpr one  = 1.0;
    Scalar constexpr two  = 2.0;
    Scalar constexpr four = 4.0;
    for (auto k = 0; k < iterations; ++k)
    {
        if (start < k)
        {
            yk = data.x + beta * (data.x - xkm1);
        }
        RunVbdIteration(sdt, sdt2);
        if (start < k)
        {
            data.x         = yk - alpha * (data.x - xkm1);
            Scalar lambdak = lambda;
            lambda         = (one + std::sqrt(one + four * lambda * lambda)) / two;
            beta           = (lambdak - one) / lambda;
        }
    }
}

} // namespace pbat::sim::vbd

#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd] NesterovIntegrator")
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
    auto constexpr L          = Scalar{1};
    using pbat::common::ToEigen;
    using pbat::sim::vbd::NesterovIntegrator;
    NesterovIntegrator nvbd{sim::vbd::Data()
                                .WithVolumeMesh(P, T)
                                .WithSurfaceMesh(V, F)
                                .WithNesterovAcceleration(L)
                                .Construct()};
    MatrixX xtilde = nvbd.data.x + dt * nvbd.data.v + dt * dt * nvbd.data.aext;
    Scalar f0      = nvbd.ObjectiveFunction(nvbd.data.x, xtilde, dt);
    VectorX g0     = nvbd.ObjectiveFunctionGradient(nvbd.data.x, xtilde, dt);

    // Act
    nvbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = nvbd.data.x - P;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
    VectorX g    = nvbd.ObjectiveFunctionGradient(nvbd.data.x, xtilde, dt);
    Scalar gnorm = g.norm();
    CHECK_LT(gnorm, zero);
    Scalar f = nvbd.ObjectiveFunction(nvbd.data.x, xtilde, dt);
    CHECK_LT(f, f0);
}