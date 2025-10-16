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
      FkRowNorm2(data.x.size(), data.mWindowSize),
      Gkm(data.x.size(), data.mWindowSize),
      Sigma(data.mWindowSize)
{
}

void BroydenIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    auto m = Xk.cols();
    gammak.setZero();
    switch (data.eBroydenJacobianEstimate)
    {
        case EBroydenJacobianEstimate::Identity: break;
        case EBroydenJacobianEstimate::ScaledIdentity: Sigma.setOnes(); break;
        case EBroydenJacobianEstimate::DiagonalCauchySchwarz: FkRowNorm2.setZero(); [[fallthrough]];
        case EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating: [[fallthrough]];
        case EBroydenJacobianEstimate::UsdDiagonal: Gkm.setOnes(); break;
        default: break;
    }
    bool const bHasUpdatingDiagonalJacobian =
        data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::DiagonalCauchySchwarz or
        data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::UsdDiagonal or
        data.eBroydenJacobianEstimate ==
            EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating;
    bool const bHasScaledIdentityJacobian =
        data.eBroydenJacobianEstimate == EBroydenJacobianEstimate::ScaledIdentity;
    Scalar Fknorm2{0};
    Scalar Bknorm2{0};
    Scalar betaF     = data.broydenBetaF;
    Scalar sqrtBetaB = std::sqrt(data.broydenBetaB);
    Scalar sigma{1};

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
        // Solve linear least-squares problem for \gamma_k
        auto mk              = std::min(m, k);
        auto Fk              = vbdFk.leftCols(mk);
        gradL2               = Fk.transpose() * vbdfk;
        FkgradL2             = Fk * gradL2;
        Scalar gradL2norm2   = gradL2.squaredNorm();
        Scalar FkgradL2norm2 = FkgradL2.squaredNorm();
        Scalar alpha         = FkgradL2norm2 > Scalar(0) ? gradL2norm2 / FkgradL2norm2 : Scalar(0);
        gammak.head(mk)      = alpha * gradL2;
        // Broyden step
        if (bHasUpdatingDiagonalJacobian)
        {
            // x_{k+1} = x_k - G_{k-m} VBD(f_k) - (X_k - G_{k-m} VBD(F_k)) \gamma_k
            data.x.reshaped() = xkm1 - Gkm.col(dkl).asDiagonal() * vbdfk;
            data.x.reshaped() -= Xk.leftCols(mk) * gammak.head(mk);
            data.x.reshaped() += Gkm.col(dkl).asDiagonal() * (vbdFk.leftCols(mk) * gammak.head(mk));
        }
        else if (bHasScaledIdentityJacobian)
        {
            sigma             = Sigma(dkl);
            data.x.reshaped() = xkm1 - sigma * vbdfk - Xk.leftCols(mk) * gammak.head(mk) +
                                sigma * (vbdFk.leftCols(mk) * gammak.head(mk));
        }
        else
        {
            data.x.reshaped() -=
                Xk.leftCols(mk) * gammak.head(mk) - vbdFk.leftCols(mk) * gammak.head(mk);
        }
        // Update Jacobian (inverse) estimate
        switch (data.eBroydenJacobianEstimate)
        {
            case EBroydenJacobianEstimate::ScaledIdentity: {
                auto sk      = Xk.col(dkl);
                auto yk      = vbdFk.col(dkl);
                Scalar ykTyk = yk.squaredNorm();
                Scalar skTyk = sk.dot(yk);
                Sigma(dkl)   = ykTyk > Scalar(0) ? skTyk / ykTyk : Scalar(1) /* fall back to VBD */;
            }
            break;
            case EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating: {
                auto ddkl        = common::Modulo(dkl - 1, m);
                auto Hkm1        = Gkm.col(ddkl);
                auto sk          = Xk.col(dkl);
                auto yk          = vbdFk.col(dkl);
                Scalar skTyk     = sk.dot(yk);
                auto Dkm1        = Hkm1.cwiseInverse();
                Scalar skTDkm1sk = sk.dot(Dkm1.asDiagonal() * sk);
                auto Ek          = sk.array().square();
                Scalar trEk2     = (Ek * Ek).sum();
                if (trEk2 > Scalar(0) and skTyk > skTDkm1sk)
                {
                    Gkm.col(dkl) =
                        (Dkm1.array() + ((skTyk - skTDkm1sk) / trEk2) * Ek).cwiseInverse();
                }
            }
            break;
            case EBroydenJacobianEstimate::UsdDiagonal: {
                // Compute Delta 1 and store it in Gkm.col(dkl)
                auto yk        = vbdFk.col(dkl);
                auto sk        = Xk.col(dkl);
                Scalar skTyk   = sk.dot(yk);
                Scalar ykTyk   = yk.squaredNorm();
                Scalar ykTsk2  = skTyk * skTyk;
                Scalar deltaki = skTyk / ykTyk;
                Scalar ykTDkyk = deltaki * ykTyk;
                Gkm.col(dkl).array() =
                    deltaki + (Scalar(1) / skTyk + ykTDkyk / ykTsk2) * sk.array().square() -
                    (Scalar(2) * deltaki / skTyk) * sk.array() * yk.array();
                // Compute Hkm and store it in Gkm.col(dkl)
                auto Yk        = yk.array().square();
                Scalar trYk2   = (Yk * Yk).sum();
                Scalar ykTD1yk = yk.dot(Gkm.col(dkl).asDiagonal() * yk);
                Gkm.col(dkl).array() += Scalar(1) + ((skTyk - ykTD1yk - ykTyk) / trYk2) * Yk;
            }
            break;
            case EBroydenJacobianEstimate::DiagonalCauchySchwarz: {
                // Accumulate lumped diagonal inverse Jacobian
                Fknorm2 += vbdFk.col(dkl).squaredNorm();
                Bknorm2 += (Xk.col(dkl).array() - Gkm.col(dkl).array() * vbdFk.col(dkl).array())
                               .square()
                               .sum();
                auto ddkl           = common::Modulo(dkl - 1, m);
                FkRowNorm2.col(dkl) = FkRowNorm2.col(ddkl) + vbdFk.col(dkl).cwiseSquare();
                sigma               = (betaF / sqrtBetaB) * (std::sqrt(Bknorm2) / Fknorm2);
                Gkm.col(dkl) += sigma * FkRowNorm2.col(dkl).cwiseSqrt();
            }
            break;
            default: break;
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