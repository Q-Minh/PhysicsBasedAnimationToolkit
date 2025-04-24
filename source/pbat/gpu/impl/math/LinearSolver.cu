// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "LinearSolver.cuh"

namespace pbat::gpu::impl::math {

LinearSolver::LinearSolver(cuda::device_t device) : mCusolverHandle(), mDevice(device)
{
    mDevice.make_current();
    CUSOLVER_CHECK(cusolverDnCreate(&mCusolverHandle));
}

LinearSolver::~LinearSolver()
{
    mDevice.make_current();
    cusolverDnDestroy(mCusolverHandle);
}

void LinearSolver::TrySetStream(std::shared_ptr<cuda::stream_t> stream) const
{
    mDevice.make_current();
    if (stream)
    {
        if (stream->device() == mDevice)
        {
            CUSOLVER_CHECK(cusolverDnSetStream(mCusolverHandle, stream->handle()));
        }
        else
        {
            throw std::invalid_argument(
                "pbat::gpu::impl::math::LinearSolver::TrySetStream -> Tried to set cuSolver stream "
                "which does not belong to the same device as the cuSolver handle.");
        }
    }
    else
    {
        CUSOLVER_CHECK(cusolverDnSetStream(mCusolverHandle, mDevice.default_stream().handle()));
    }
}

} // namespace pbat::gpu::impl::math

#include "Blas.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Eigen.cuh"

#include <Eigen/QR>
#include <algorithm>
#include <doctest/doctest.h>

TEST_CASE("[gpu][impl][math] LinearSolver")
{
    using namespace pbat;
    using pbat::gpu::impl::common::Buffer;
    using pbat::gpu::impl::common::ToBuffer;
    using pbat::gpu::impl::common::ToEigen;
    using pbat::gpu::impl::math::Blas;
    using pbat::gpu::impl::math::LinearSolver;
    using namespace pbat::gpu::impl;

    // Arrange
    auto constexpr eps = 1e-6f;
    auto constexpr m   = 10;
    auto constexpr n   = 5;
    GpuMatrixX A       = GpuMatrixX::Random(m, n);
    GpuVectorX x       = GpuVectorX::Random(n);
    GpuVectorX b       = A * x;
    math::Matrix<GpuScalar> dQR(m, n);
    math::Matrix<GpuScalar> dB(m, 1);
    math::Vector<GpuScalar> dTau(n);
    ToBuffer(A, dQR.data);
    ToBuffer(b, dB.data);
    Blas blas;
    LinearSolver solver;

    // Act
    auto const geqrfWorkspaceSize = solver.GeqrfWorkspace(dQR);
    auto const ormqrWorkspaceSize = solver.OrmqrWorkspace(dQR, dB);
    auto workspaceSize            = std::max(geqrfWorkspaceSize, ormqrWorkspaceSize);
    Buffer<GpuScalar> workspace(workspaceSize);
    solver.Geqrf(dQR, dTau, workspace);
    solver.Ormqr(dQR.Transposed(), dTau, dB, workspace);
    auto dX = dB.TopRows(n);
    blas.UpperTriangularSolve(dQR, dX);

    // Assert
    GpuMatrixX QR        = ToEigen(dQR.data).reshaped(m, n);
    GpuVectorX tau       = ToEigen(dTau.data).reshaped();
    GpuVectorX xExpected = x;
    x                    = ToEigen(dB.data).reshaped().segment(0, n);
    bool const bAreEqual = x.isApprox(xExpected, eps);
    CHECK(bAreEqual);
}