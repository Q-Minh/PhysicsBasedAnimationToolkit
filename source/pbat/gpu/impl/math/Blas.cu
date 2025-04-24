// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Blas.cuh"

namespace pbat::gpu::impl::math {

Blas::Blas(cuda::device_t device) : mHandle(), mDevice(device)
{
    mDevice.make_current();
    CUBLAS_CHECK(cublasCreate(&mHandle));
}

Blas::~Blas()
{
    mDevice.make_current();
    /*CUBLAS_CHECK(*/ cublasDestroy(mHandle) /*)*/;
}

void Blas::TrySetStream(std::shared_ptr<cuda::stream_t> stream) const
{
    mDevice.make_current();
    if (stream and stream->device() == mDevice)
    {
        CUBLAS_CHECK(cublasSetStream(mHandle, stream->handle()));
    }
    else
    {
        CUBLAS_CHECK(cublasSetStream(mHandle, mDevice.default_stream().handle()));
    }
}

} // namespace pbat::gpu::impl::math

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Eigen.cuh"

#include <Eigen/Cholesky>
#include <doctest/doctest.h>

TEST_CASE("[gpu][impl][math] Blas")
{
    using namespace pbat;
    using pbat::gpu::impl::common::Buffer;
    using pbat::gpu::impl::common::ToBuffer;
    using pbat::gpu::impl::common::ToEigen;
    using pbat::gpu::impl::math::Blas;
    using namespace pbat::gpu::impl;

    // Arrange
    auto constexpr eps = 1e-6f;
    auto constexpr n   = 10;
    GpuMatrixX A       = GpuMatrixX::Random(n, n);
    GpuVectorX x       = GpuVectorX::Random(n);
    GpuVectorX b       = A * x;
    A                  = A.transpose() * A;
    auto LLT           = A.llt();
    GpuMatrixX L       = LLT.matrixL();
    GpuMatrixX U       = L.transpose();
    math::Matrix<GpuScalar> dL(n, n);
    math::Matrix<GpuScalar> dU(n, n);
    math::Matrix<GpuScalar> dB(n, 1);
    ToBuffer(L, dL.data);
    ToBuffer(U, dU.data);
    ToBuffer(b, dB.data);
    Blas blas;

    // Act
    blas.LowerTriangularSolve(dL, dB);
    blas.UpperTriangularSolve(dU, dB);

    // Assert
    GpuVectorX xEigen    = ToEigen(dB.data);
    GpuVectorX xExpected = LLT.solve(b);
    bool const bAreEqual = xEigen.isApprox(xExpected, eps);
    CHECK(bAreEqual);
}