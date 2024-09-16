// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Matrix.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"

#include <doctest/doctest.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace test {

template <class Func>
pbat::GpuMatrixX RunKernel(pbat::GpuMatrixX const& A)
{
    using namespace pbat::gpu;
    auto const toGpu = [](GpuMatrixX const& A) {
        common::Buffer<GpuScalar> buf(A.size());
        thrust::copy(A.data(), A.data() + A.size(), buf.Data());
        return buf;
    };
    auto const fromGpu = [](common::Buffer<GpuScalar> const& buf, auto rows, auto cols) {
        GpuMatrixX A(rows, cols);
        thrust::copy(buf.Data(), buf.Data() + buf.Size(), A.data());
        return A;
    };
    auto buf   = toGpu(A);
    auto begin = thrust::make_counting_iterator<GpuIndex>(0);
    auto end   = thrust::make_counting_iterator<GpuIndex>(1);
    thrust::for_each(thrust::device, begin, end, Func{buf.Raw()});
    return fromGpu(buf, A.rows(), A.cols());
}

struct FTranspose
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> Ad(data + 3 * 3 * i);
        Matrix<GpuScalar, 3, 3> AdT = Ad.Transpose();
        Ad                          = AdT;
    }
    GpuScalar* data;
};

struct FSubMatrix
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> Ad(data + 3 * 3 * i);
        Matrix<GpuScalar, 2, 2> Ads = Ad.Slice<2, 2>(1, 1);
        Ad.Slice<2, 2>(0, 0)        = Ads;
    }
    GpuScalar* data;
};

struct FTiledView
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> Ad(data + 3 * 3 * i);
        Ad.Slice<3, 2>(0, 1) = Repeat<1, 2>(Ad.Col(0));
    }
    GpuScalar* data;
};

struct FScaleAndSumTranspose
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> Ad(data + 3 * 3 * i);
        Matrix<GpuScalar, 3, 3> B = 2.f * Ad + Ad.Transpose();
        Ad                        = B;
    }
    GpuScalar* data;
};

struct FSquaredNorm
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 6, 3> Ad(data + 6 * 3 * i);
        GpuScalar norm2 = SquaredNorm(Ad);
        Ad.SetConstant(norm2);
    }
    GpuScalar* data;
};

struct FCrossProduct
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 2> Ad(data + 3 * 2 * i);
        Matrix<GpuScalar, 3, 1> cross = Cross(Ad.Col(0), Ad.Col(1));
        Ad                            = Repeat<1, 2>(cross);
    }
    GpuScalar* data;
};

struct FDeterminant
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> Ad(data + 3 * 3 * i);
        GpuScalar const det = Determinant(Ad);
        Ad.SetConstant(det);
    }
    GpuScalar* data;
};

struct FInverse
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> Ad(data + 3 * 3 * i);
        Matrix<GpuScalar, 3, 3> Ainv = Inverse(Ad);
        Ad                           = Ainv;
    }
    GpuScalar* data;
};

struct FComposedOperation
{
    __device__ void operator()(GpuIndex i)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 4> Ad(data + 3 * 4 * i);
        auto B              = Ad.Slice<3, 3>(0, 1) - Repeat<1, 3>(Ad.Col(0));
        GpuScalar const tr  = Trace(Inverse(2.f * B));
        GpuScalar const det = Determinant((B * Ad) * (B * Ad).Transpose());
        Ad.SetConstant(det + tr);
    }
    GpuScalar* data;
};

} // namespace test
} // namespace pbat

TEST_CASE("[gpu][math][linalg] Matrix operations")
{
    using namespace pbat;
    GpuScalar constexpr zero = 1e-10f;

    using namespace pbat::test;
    SUBCASE("Transpose")
    {
        GpuMatrixX A  = GpuMatrixX::Random(3, 3);
        GpuMatrixX A2 = RunKernel<FTranspose>(A);
        A             = A.transpose().eval();
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("SubMatrix")
    {
        GpuMatrixX A        = GpuMatrixX::Random(3, 3);
        GpuMatrixX A2       = RunKernel<FSubMatrix>(A);
        A.block<2, 2>(0, 0) = A.block<2, 2>(1, 1).eval();
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("Repeat")
    {
        GpuMatrixX A        = GpuMatrixX::Random(3, 3);
        GpuMatrixX A2       = RunKernel<FTiledView>(A);
        A.block<3, 2>(0, 1) = A.col(0).replicate<1, 2>();
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("Scale and Sum Transpose")
    {
        GpuMatrixX A  = GpuMatrixX::Random(3, 3);
        GpuMatrixX A2 = RunKernel<FScaleAndSumTranspose>(A);
        A             = (2.f * A + A.transpose()).eval();
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("Squared norm")
    {
        GpuMatrixX A  = GpuMatrixX::Random(6, 3);
        GpuMatrixX A2 = RunKernel<FSquaredNorm>(A);
        A.setConstant(A.squaredNorm());
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("Cross product")
    {
        GpuMatrixX A  = GpuMatrixX::Random(3, 2);
        GpuMatrixX A2 = RunKernel<FCrossProduct>(A);
        A.colwise()   = A.col(0).segment<3>(0).cross(A.col(1).segment<3>(0));
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("Determinant")
    {
        GpuMatrixX A  = GpuMatrixX::Random(3, 3);
        GpuMatrixX A2 = RunKernel<FDeterminant>(A);
        A.setConstant(A.block<3, 3>(0, 0).determinant());
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("Inverse")
    {
        GpuMatrixX A  = GpuMatrixX::Random(3, 3);
        GpuMatrixX A2 = RunKernel<FInverse>(A);
        A             = A.block<3, 3>(0, 0).inverse().eval();
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
    SUBCASE("Highly composed operation")
    {
        GpuMatrixX A        = GpuMatrixX::Random(3, 4);
        GpuMatrixX A2       = RunKernel<FComposedOperation>(A);
        GpuMatrixX B        = A.block<3, 3>(0, 1) - A.col(0).replicate(1, 3);
        GpuScalar const tr  = (2.f * B).inverse().trace();
        GpuScalar const det = ((B * A) * (B * A).transpose()).determinant();
        A.setConstant(det + tr);
        CHECK_LE((A2 - A).squaredNorm(), zero);
    }
}