// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Aabb.cuh"
#include "pbat/common/Eigen.h"
#include "pbat/gpu/common/Eigen.cuh"
#include "pbat/math/linalg/mini/Mini.h"

#include <doctest/doctest.h>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {
namespace test {

void RunAabbTests()
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    GpuMatrixX P(3, 8);
    GpuIndexMatrixX T(4, 5);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    using gpu::geometry::impl::Aabb;
    auto const nPts  = static_cast<GpuIndex>(P.cols());
    auto const nTets = static_cast<GpuIndex>(T.cols());
    gpu::common::Buffer<GpuScalar, 3> PG(nPts);
    gpu::common::Buffer<GpuIndex, 4> TG(nTets);
    gpu::common::ToBuffer(P, PG);
    gpu::common::ToBuffer(T, TG);
    Aabb<3> aabbs(nTets);
    // Act
    using namespace math::linalg;
    auto const fLowerUpperFromPts = [] PBAT_DEVICE(auto pts) {
        mini::SMatrix<GpuScalar, 3, 2> LU;
        pbat::common::ForRange<0, 3>([&]<auto d>() {
            LU(d, 0) = Min(pts.Row(d));
            LU(d, 1) = Max(pts.Row(d));
        });
        return LU;
    };
    aabbs.Construct([fLowerUpperFromPts, P = PG.Raw(), T = TG.Raw()] PBAT_DEVICE(GpuIndex s) {
        auto inds = mini::FromBuffers<4, 1>(T, s);
        auto pts  = mini::FromBuffers(P, inds.Transpose());
        return fLowerUpperFromPts(pts);
    });
    // Assert
    auto L                   = pbat::common::ToEigen(aabbs.b.Get()).reshaped(3, nTets).eval();
    auto U                   = pbat::common::ToEigen(aabbs.e.Get()).reshaped(3, nTets).eval();
    bool const bLowerAreZero = (L.array() == GpuScalar(0)).all();
    CHECK(bLowerAreZero);
    bool const bUpperAreOne = (U.array() == GpuScalar(1)).all();
    CHECK(bUpperAreOne);
}

} // namespace test
} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

TEST_CASE("[gpu][geometry][impl] Aabb")
{
    pbat::gpu::geometry::impl::test::RunAabbTests();
}