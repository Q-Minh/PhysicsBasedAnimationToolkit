#include "BroydenIntegrator.h"

#include "pbat/common/Modulo.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <exception>

namespace pbat::sim::vbd {

BroydenIntegrator::BroydenIntegrator(Data dataIn)
    : Integrator(std::move(dataIn)),
      GvbdFk(data.x.size(), data.mWindowSize),
      Xk(data.x.size(), data.mWindowSize),
      gammak(data.mWindowSize),
      xkm1(data.x.size()),
      vbdfk(data.x.size()),
      vbdfkm1(data.x.size())
{
}

void BroydenIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    auto m = Xk.cols();

    // If x_{k+1} = x_k - VBD(f_k), then
    // VBD(f_k) = x_k - x_{k+1}
    xkm1 = data.x.reshaped();
    RunVbdIteration(sdt, sdt2);
    vbdfkm1 = xkm1 - data.x.reshaped();
    for (Index k = 1; k < iterations; ++k)
    {
        // \Delta x_{k-1} \leftarrow x_k - x_{k-1}
        auto dkl    = common::Modulo(k - 1, m);
        Xk.col(dkl) = data.x.reshaped() - xkm1;
        xkm1        = data.x.reshaped();
        // G_{k-m} VBD(f_k)
        RunVbdIteration(sdt, sdt2);
        vbdfk = xkm1 - data.x.reshaped();
        // G_{k-m} VBD(\Delta f_k) = VBD(f_k) - VBD(f_{k-1})
        GvbdFk.col(dkl) = vbdfk - vbdfkm1;
        vbdfkm1         = vbdfk;
        // Compute Broyden update
        auto mk = std::min(m, k);
        auto GvbdFkLS = GvbdFk.leftCols(mk);
        Eigen::LeastSquaresConjugateGradient<MatrixX> cg(GvbdFkLS);
        cg.setMaxIterations(m);
        cg.setTolerance(1e-10);
        // \gamma_k = [ VBD(F_k)^T VBD(F_k) ]^{-1} VBD(f_k)
        gammak.head(mk) = cg.solve(vbdfk);
        // x_{k+1} = x_k - VBD(f_k) - (X_k - G_{k-m} VBD(F_k)) \gamma_k
        data.x.reshaped() = xkm1 - vbdfk - Xk.leftCols(mk) * gammak.head(mk) +
                            GvbdFk.leftCols(mk) * gammak.head(mk);
    }
}

} // namespace pbat::sim::vbd

#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd] BroydenIntegrator")
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
    using pbat::sim::vbd::BroydenIntegrator;
    BroydenIntegrator avbd{sim::vbd::Data()
                               .WithVolumeMesh(P, T)
                               .WithSurfaceMesh(V, F)
                               .WithBroydenMethod(m)
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