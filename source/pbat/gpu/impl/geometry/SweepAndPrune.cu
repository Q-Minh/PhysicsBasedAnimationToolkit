// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "SweepAndPrune.cuh"

namespace pbat {
namespace gpu {
namespace impl {
namespace geometry {

SweepAndPrune::SweepAndPrune(std::size_t nPrimitives) : inds(nPrimitives) {}

void SweepAndPrune::Reserve(std::size_t nPrimitives)
{
    inds.Resize(nPrimitives);
}

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/common/Hash.h"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/common/SynchronizedList.cuh"
#include "pbat/math/linalg/mini/Mini.h"

#include <cuda/std/utility>
#include <doctest/doctest.h>
#include <unordered_set>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {
namespace test {

void RunSweepAndPruneTests()
{
    using namespace pbat;
    // Arrange
    GpuMatrixX V(3, 7);
    GpuIndexMatrixX E1(2, 3);
    GpuIndexMatrixX F2(3, 1);
    // clang-format off
    V << 0.f,  1.f ,  2.f ,  3.f , 0.f,  2.f ,  0.f,
         0.f,  0.1f,  0.2f,  0.3f, 0.f,  0.1f,  0.f,
         0.f, 10.f , 20.f , 30.f , 0.f, 10.f ,  0.f;
    E1 << 1, 0, 2,
          2, 1, 3;
    F2 << 4,
          5,
          6;
    // clang-format on
    using gpu::impl::geometry::Aabb;
    auto const nEdges     = static_cast<GpuIndex>(E1.cols());
    auto const nTriangles = static_cast<GpuIndex>(F2.cols());
    gpu::impl::common::Buffer<GpuScalar, 3> VG(V.cols());
    gpu::impl::common::Buffer<GpuIndex, 2> EG(E1.cols());
    gpu::impl::common::Buffer<GpuIndex, 3> FG(F2.cols());
    gpu::impl::common::ToBuffer(V, VG);
    gpu::impl::common::ToBuffer(E1, EG);
    gpu::impl::common::ToBuffer(F2, FG);
    Aabb<3> aabbs(nEdges + nTriangles);
    using namespace math::linalg;
    auto const fLowerUpperFromPts = [] PBAT_DEVICE(auto pts) {
        mini::SMatrix<GpuScalar, 3, 2> LU;
        pbat::common::ForRange<0, 3>([&]<auto d>() {
            LU(d, 0) = Min(pts.Row(d));
            LU(d, 1) = Max(pts.Row(d));
        });
        return LU;
    };
    auto const fEdgeAabb =
        [v = VG.Raw(), e = EG.Raw(), fLowerUpperFromPts] PBAT_DEVICE(GpuIndex s) {
            auto inds = mini::FromBuffers<2, 1>(e, s);
            auto pts  = mini::FromBuffers(v, inds.Transpose());
            return fLowerUpperFromPts(pts);
        };
    auto const fTriangleAabb =
        [v = VG.Raw(), f = FG.Raw(), fLowerUpperFromPts] PBAT_DEVICE(GpuIndex s) {
            auto inds = mini::FromBuffers<3, 1>(f, s);
            auto pts  = mini::FromBuffers(v, inds.Transpose());
            return fLowerUpperFromPts(pts);
        };
    aabbs.Construct([nEdges, fEdgeAabb, fTriangleAabb] PBAT_DEVICE(GpuIndex s) {
        return (s < nEdges) ? fEdgeAabb(s) : fTriangleAabb(s - nEdges);
    });
    using OverlapType = cuda::std::pair<GpuIndex, GpuIndex>;
    struct Hash
    {
        std::size_t operator()(OverlapType const& overlap) const
        {
            return pbat::common::HashCombine(overlap.first, overlap.second);
        }
    };
    using OverlapSetType = std::unordered_set<OverlapType, Hash>;
    OverlapSetType overlapsExpected{{{0, 0}, {1, 0}}};
    // Act
    gpu::impl::common::SynchronizedList<OverlapType> overlaps(3 * overlapsExpected.size());
    gpu::impl::geometry::SweepAndPrune sap{};
    sap.SortAndSweep(
        aabbs,
        [nEdges, o = overlaps.Raw()] PBAT_DEVICE(GpuIndex si, GpuIndex sj) mutable {
            if (si < nEdges and sj >= nEdges)
                o.Append(OverlapType{si, sj - nEdges});
            if (si >= nEdges and sj < nEdges)
                o.Append(OverlapType{sj, si - nEdges});
        });
    std::vector<OverlapType> overlapsCpu = overlaps.Get();
    // Assert
    for (OverlapType overlap : overlapsCpu)
    {
        auto it                             = overlapsExpected.find(overlap);
        bool const bExpectedOverlapDetected = it != overlapsExpected.end();
        CHECK(bExpectedOverlapDetected);
        overlapsExpected.erase(it);
    }
    CHECK(overlapsExpected.empty());
}

} // namespace test
} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

TEST_CASE("[gpu][impl][geometry] Sweep and prune")
{
    pbat::gpu::geometry::impl::test::RunSweepAndPruneTests();
}
