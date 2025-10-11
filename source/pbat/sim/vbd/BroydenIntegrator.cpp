#include "BroydenIntegrator.h"

#include "pbat/common/Modulo.h"
#include "pbat/fem/Laplacian.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <exception>

namespace pbat::sim::vbd {

BroydenIntegrator::BroydenIntegrator(Data dataIn)
    : Integrator(std::move(dataIn)),
      vbdFk(data.x.size(), data.mWindowSize),
      Xk(data.x.size(), data.mWindowSize),
      gammak(data.mWindowSize),
      xkm1(data.x.size()),
      vbdfk(data.x.size()),
      vbdfkm1(data.x.size()),
      gradL2(data.x.size()),
      FkgradL2(data.x.size()),
      CSXFk(data.x.size(), data.mWindowSize),
      CSFk(data.x.size(), data.mWindowSize),
      Gkm(data.x.size(), data.mWindowSize)
{
}

void BroydenIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    auto m = Xk.cols();
    gammak.setZero();
    if (data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::DiagonalCauchySchwarz)
    {
        CSXFk.setZero();
        CSFk.setZero();
        Gkm.setOnes();
    }
    Scalar Fknorm2{0};
    Scalar beta = data.broydenBeta;

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
        // If x_{k+1} = x_k - VBD(f_k), then VBD(f_k) = x_k - x_{k+1}
        vbdfk = xkm1 - data.x.reshaped();
        // G_{k-m} VBD(\Delta f_k) = VBD(f_k) - VBD(f_{k-1})
        vbdFk.col(dkl) = vbdfk - vbdfkm1;
        vbdfkm1        = vbdfk;
        if (data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::DiagonalCauchySchwarz)
            Fknorm2 += vbdFk.col(dkl).squaredNorm();
        // Compute Broyden update
        auto mk         = std::min(m, k);
        auto Fk         = vbdFk.leftCols(mk);
        gradL2          = Fk.transpose() * vbdfk;
        FkgradL2        = Fk * gradL2;
        auto alpha      = gradL2.squaredNorm() / (FkgradL2).squaredNorm();
        gammak.head(mk) = alpha * gradL2;
        // Estimate diag(G_{k-m})
        if (data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::DiagonalCauchySchwarz)
        {
            if (k > m)
                Gkm.col(dkl).array() +=
                    (beta / Fknorm2) * CSXFk.col(dkl).array() * CSFk.col(dkl).array();
        }
        // x_{k+1} = x_k - G_{k-m} VBD(f_k) - (X_k - G_{k-m} VBD(F_k)) \gamma_k
        if (data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::DiagonalCauchySchwarz)
        {
            data.x.reshaped() = xkm1 - Gkm.col(dkl).asDiagonal() * vbdfk;
            data.x.reshaped() -= Xk.leftCols(mk) * gammak.head(mk);
            data.x.reshaped() += Gkm.col(dkl).asDiagonal() * (vbdFk.leftCols(mk) * gammak.head(mk));
        }
        else
        {
            data.x.reshaped() -=
                Xk.leftCols(mk) * gammak.head(mk) - vbdFk.leftCols(mk) * gammak.head(mk);
        }
        // Cauchy-Schwarz squared norms
        if (data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::DiagonalCauchySchwarz)
        {
            auto ddkl = common::Modulo(dkl - 1, m);
            CSXFk.col(dkl) =
                CSXFk.col(ddkl).array() + (Xk.col(dkl).array() - vbdFk.col(dkl).array()).square();
            CSFk.col(dkl) = CSFk.col(ddkl).array() + vbdFk.col(dkl).array().square();
        }
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
    V.reshaped().setLinSpaced(0, P.cols() - 1);
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