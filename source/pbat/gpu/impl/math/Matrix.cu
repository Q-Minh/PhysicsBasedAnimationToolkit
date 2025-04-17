// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Matrix.cuh"

namespace pbat::gpu::impl::math {

Blas::Blas()
{
    CUBLAS_CHECK(cublasCreate(&mHandle));
}

Blas::~Blas()
{
    /*CUBLAS_CHECK(*/ cublasDestroy(mHandle) /*)*/;
}

} // namespace pbat::gpu::impl::math

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Eigen.cuh"

#include <doctest/doctest.h>

TEST_CASE("[gpu][impl][math] Blas")
{
    using namespace pbat;
    using pbat::gpu::impl::common::Buffer;
    using pbat::gpu::impl::common::ToBuffer;
    using pbat::gpu::impl::common::ToEigen;
    using pbat::gpu::impl::math::Blas;
    using namespace pbat::gpu::impl;

    auto constexpr m = 10;
    auto constexpr n = 5;
    GpuMatrixX A     = GpuMatrixX::Random(m, n);
    GpuVectorX x     = GpuVectorX::Random(n);
    GpuVectorX y     = A * x;
    math::Matrix<GpuScalar> dA(m, n);
    math::Vector<GpuScalar> dx(n);
    math::Vector<GpuScalar> dy(m);
    ToBuffer(A, dA.data);
    ToBuffer(x, dx.data);
    ToBuffer(y, dy.data);

    Blas blas;
    blas.Gemv(dA, dx, dy);

    auto yEigen          = ToEigen(dy.data);
    auto constexpr eps   = 1e-6f;
    bool const bAreEqual = y.isApprox(yEigen.reshaped(), eps);
    CHECK(bAreEqual);
}