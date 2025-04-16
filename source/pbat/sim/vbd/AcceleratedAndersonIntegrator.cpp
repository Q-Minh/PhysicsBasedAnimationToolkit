#include "AcceleratedAndersonIntegrator.h"

#include "pbat/profiling/Profiling.h"

#include <Eigen/QR>
#include <algorithm>
#include <exception>
#include <limits>
#include <tbb/parallel_for.h>

namespace pbat::sim::vbd {

AcceleratedAndersonIntegrator::AcceleratedAndersonIntegrator(Data dataIn)
    : Integrator(std::move(dataIn)),
      U(data.x.size(), data.mAndersonWindowSize),
      V(data.x.size(), data.mAndersonWindowSize),
      xkm1(data.x.size()),
      dx(data.x.size()),
      Fk(data.x.size()),
      Fkm1(data.x.size()),
      GdFk(data.x.size()),
      GTdx(data.x.size())
{
}

void AcceleratedAndersonIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    auto const mod = [](auto a, auto b) {
        return (a % b + b) % b;
    };
    auto m  = U.cols();
    auto n  = U.rows();
    auto G0 = -MatrixX::Identity(n, n);
    Index mk{0};
    auto const Gmul = [&](auto const& x) {
        return G0 * x + U.leftCols(mk) * (V.leftCols(mk).transpose() * x);
    };
    auto const GTmul = [&](auto const& x) {
        return G0 * x + V.leftCols(mk) * (U.leftCols(mk).transpose() * x);
    };
    Scalar constexpr eps = std::numeric_limits<Scalar>::epsilon();

    xkm1 = data.x.reshaped();
    RunVbdIteration(sdt, sdt2);
    Fk = data.x.reshaped() - xkm1;
    for (Index k = 0; k < iterations; ++k)
    {
        // Broyden step
        mk = std::min(m, k);
        dx = -Gmul(Fk);
        data.x.reshaped() += dx;
        // Compute f(x) = g(x) - x, i.e. the VBD iteration
        Fkm1 = Fk;
        xkm1 = data.x.reshaped();
        RunVbdIteration(sdt, sdt2);
        Fk       = data.x.reshaped() - xkm1;
        auto dFk = Fk - Fkm1;
        GdFk     = Gmul(dFk);
        // Broyden update
        auto kl    = mod(k, m);
        U.col(kl)  = dx - GdFk;
        Scalar den = dx.dot(GdFk);
        GTdx       = GTmul(dx);
        V.col(kl)  = GTdx / den;
    }
}

} // namespace pbat::sim::vbd

#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd] AcceleratedAndersonIntegrator")
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
    auto constexpr iterations = 15;
    Index constexpr m         = 5;
    using pbat::common::ToEigen;
    using pbat::sim::vbd::AcceleratedAndersonIntegrator;
    AcceleratedAndersonIntegrator avbd{sim::vbd::Data()
                                           .WithVolumeMesh(P, T)
                                           .WithSurfaceMesh(V, F)
                                           .WithAcceleratedAnderson(m)
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