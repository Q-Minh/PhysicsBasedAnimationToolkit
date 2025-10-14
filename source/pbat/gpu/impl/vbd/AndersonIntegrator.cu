// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "AndersonIntegrator.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Modulo.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <cuda/api.hpp>
#include <limits>

namespace pbat::gpu::impl::vbd {

AndersonIntegrator::AndersonIntegrator(Data const& data)
    : Integrator(data),
      Fk(data.X.size()),
      Fkm1(data.X.size()),
      Gk(data.X.size()),
      Gkm1(data.X.size()),
      xkm1(data.X.size()),
      mDFK(data.X.size(), data.mWindowSize),
      mDGK(data.X.size(), data.mWindowSize),
      mQR(data.X.size(), data.mWindowSize),
      mTau(data.mWindowSize),
      mBlas(),
      mLinearSolver(),
      mLinearSolverWorkspace()
{
    auto const geqrfWorkspaceSize = mLinearSolver.GeqrfWorkspace(mQR);
    auto const ormqrWorkspaceSize = mLinearSolver.OrmqrWorkspace(mQR, Fk);
    auto workspaceSize            = std::max(geqrfWorkspaceSize, ormqrWorkspaceSize);
    mLinearSolverWorkspace.Resize(workspaceSize);
}

void AndersonIntegrator::Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.vbd.AndersonIntegrator.Solve");
    // Start Anderson acceleration
    xkm1.data = x;
    RunVbdIteration(bdf);
    Gkm1.data = x;
    mBlas.Copy(Gkm1, Fkm1);                // Fkm1 = Gkm1
    mBlas.Axpy(xkm1, Fkm1, GpuScalar(-1)); // Fkm1 = Gkm1 - xkm1
    for (auto k = 1; k < iterations; ++k)
    {
        xkm1.data = x;
        RunVbdIteration(bdf);
        UpdateAndersonWindow(k);
        TakeAndersonAcceleratedStep(k);
    }
}

void AndersonIntegrator::UpdateAndersonWindow(GpuIndex k)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.vbd.AndersonIntegrator.UpdateAndersonWindow");
    auto m   = mDFK.Cols();
    auto dkl = pbat::common::Modulo(k - 1, m);
    Gk.data  = x;
    mBlas.Copy(Gk, Fk);                  // Fk = Gk
    mBlas.Axpy(xkm1, Fk, GpuScalar(-1)); // Fk = Gk - xkm1
    auto DGKk = mDGK.Col(dkl).Flattened();
    auto DFKk = mDFK.Col(dkl).Flattened();
    mBlas.Copy(Gk, DGKk);                  // DGK_k = Gk
    mBlas.Copy(Fk, DFKk);                  // DFK_k = Fk
    mBlas.Axpy(Gkm1, DGKk, GpuScalar(-1)); // DGK_k = Gk - Gkm1
    mBlas.Axpy(Fkm1, DFKk, GpuScalar(-1)); // DFK_k = Fk - Fkm1
    mBlas.Copy(Gk, Gkm1);                  // Gkm1 = Gk
    mBlas.Copy(Fk, Fkm1);                  // Fkm1 = Fk
}

void AndersonIntegrator::TakeAndersonAcceleratedStep(GpuIndex k)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.vbd.AndersonIntegrator.TakeAndersonAcceleratedStep");
    auto m = mDFK.Cols();
    // Anderson update, i.e. solve \min_\alpha | DFK \alpha - Fk |
    auto mk  = std::min(k, m);
    mQR      = mDFK;
    auto QR  = mQR.LeftCols(mk);
    auto tau = mTau.Head(mk);
    mLinearSolver.Geqrf(QR, tau, mLinearSolverWorkspace);
    RegularizeRFactor(mk);
    mLinearSolver.Ormqr(QR.Transposed(), tau, Fk, mLinearSolverWorkspace);
    auto alpha = Fk.Head(mk);
    mBlas.UpperTriangularSolve(QR, alpha);
    auto DGK = mDGK.LeftCols(mk);
    mBlas.Gemv(DGK, alpha, Gk, GpuScalar(-1), GpuScalar(1)); // Gk = Gk - DGK * alpha
    // Copy Gk back to x
    x = Gk.data;
}

void AndersonIntegrator::RegularizeRFactor(GpuIndex mk)
{
    GpuScalar constexpr reg = GpuScalar(1e10) * std::numeric_limits<GpuScalar>::min();
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(mk),
        [QR = mQR.data.Raw(), ld = mQR.LeadingDimensions(), reg] PBAT_DEVICE(auto j) {
            auto rj = QR[j * ld + j];
            if (rj == GpuScalar(0))
                QR[j * ld + j] = reg;
        });
}

} // namespace pbat::gpu::impl::vbd

#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[pbat][gpu][impl][vbd] AndersonIntegrator")
{
    using namespace pbat;
    using pbat::common::ToEigen;
    // Arrange
    // Cube mesh
    MatrixX P(3, 8);
    IndexMatrixX V(1, 8);
    IndexMatrixX T(4, 5);
    IndexMatrixX F(3, 12);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
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
    auto constexpr dt         = GpuScalar{1e-2};
    auto constexpr substeps   = 1;
    auto constexpr iterations = 10;
    auto const worldMin       = P.rowwise().minCoeff().cast<GpuScalar>().eval();
    auto const worldMax       = P.rowwise().maxCoeff().cast<GpuScalar>().eval();

    // Act
    using pbat::gpu::impl::vbd::AndersonIntegrator;
    AndersonIntegrator vbd{sim::vbd::Data()
                               .WithVolumeMesh(P, T)
                               .WithSurfaceMesh(V, F)
                               .WithBodies(IndexVectorX::Ones(P.cols()))
                               .Construct()};
    vbd.SetSceneBoundingBox(worldMin, worldMax);
    vbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero = GpuScalar{1e-4};
    GpuMatrixX dx =
        ToEigen(vbd.x.Get()).reshaped(P.cols(), P.rows()).transpose() - P.cast<GpuScalar>();
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < GpuScalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}