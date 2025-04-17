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

TEST_CASE("[gpu][impl][math] Matrix")
{
    using namespace pbat;
    using pbat::gpu::impl::common::Buffer;
    using pbat::gpu::impl::common::ToBuffer;
    using pbat::gpu::impl::common::ToEigen;
    using pbat::gpu::impl::math::Blas;
    using namespace pbat::gpu::impl;
    // Arrange
    auto constexpr eps = 1e-6f;
    auto constexpr m   = 10;
    auto constexpr n   = 5;
    GpuMatrixX A       = GpuMatrixX::Random(m, n);
    GpuVectorX x       = GpuVectorX::Random(n);
    GpuVectorX y       = A * x;
    math::Matrix<GpuScalar> dA(m, n);
    math::Vector<GpuScalar> dx(n);
    math::Vector<GpuScalar> dy(m);
    ToBuffer(A, dA.data);
    ToBuffer(x, dx.data);
    ToBuffer(y, dy.data);
    Blas blas;

    SUBCASE("Full matrices and vectors")
    {
        // Act
        blas.Gemv(dA, dx, dy);
        // Assert
        auto yEigen          = ToEigen(dy.data);
        bool const bAreEqual = y.isApprox(yEigen.reshaped(), eps);
        CHECK(bAreEqual);
    }
    SUBCASE("Sub-matrices and sub-vectors")
    {
        // Arrange
        auto dAS              = dA.SubMatrix(1, 1, 3, 3);
        auto dxS              = dx.Segment(1, 3);
        auto dyS              = dy.Segment(1, 3);
        GpuVectorX ySExpected = A.block(1, 1, 3, 3) * x.segment(1, 3) + y.segment(1, 3);
        // Act
        blas.Gemv(dAS, dxS, dyS, 1.f, 1.f);
        // Assert
        auto yEigen          = ToEigen(dy.data);
        auto ySEigen         = yEigen.reshaped().segment(1, 3);
        bool const bAreEqual = ySEigen.isApprox(ySExpected, eps);
        CHECK(bAreEqual);
    }
}