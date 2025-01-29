// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VertexTriangleMixedCcdDcd.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pbat::gpu::impl::contact {

VertexTriangleMixedCcdDcd::VertexTriangleMixedCcdDcd(
    common::Buffer<GpuIndex> const& Vin,
    common::Buffer<GpuIndex, 3> const& Fin)
    : V(Vin),
      F(Fin),
      inds(Vin.Size()),
      morton(Vin.Size()),
      Paabbs(static_cast<GpuIndex>(Vin.Size())),
      Faabbs(static_cast<GpuIndex>(Fin.Size())),
      Fbvh(static_cast<GpuIndex>(Fin.Size())),
      active(Vin.Size()),
      av(Vin.Size()),
      nActive(0),
      nn(Vin.Size() * kMaxNeighbours),
      sd(Vin.Size())
{
}

void VertexTriangleMixedCcdDcd::InitializeActiveSet(
    common::Buffer<GpuScalar, 3> const& xt,
    common::Buffer<GpuScalar, 3> const& xtp1,
    geometry::Morton::Bound const& wmin,
    geometry::Morton::Bound const& wmax)
{
    auto const nPts = Paabbs.Size();
    // 1. Compute aabbs of the swept points (i.e. line segments)
    Paabbs.Construct(
        [xt = xt.Raw(), x = xtp1.Raw(), V = V.Raw(), inds = inds.Raw()] PBAT_DEVICE(GpuIndex v) {
            GpuIndex const i = V[inds[v]];
            using namespace pbat::math::linalg::mini;
            SMatrix<GpuScalar, 3, 2> xe;
            xe.Col(0) = FromBuffers<3, 1>(xt, i);
            xe.Col(1) = FromBuffers<3, 1>(x, i);
            SMatrix<GpuScalar, 3, 2> LU;
            pbat::common::ForRange<0, kDims>([&]<auto d>() {
                LU(d, 0) = Min(xe.Row(d));
                LU(d, 1) = Max(xe.Row(d));
            });
            return LU;
        });
    // 2. Sort line segments by morton code
    morton.Encode(Paabbs, wmin, wmax);
    auto zip = thrust::make_zip_iterator(
        inds.Data(),
        Paabbs.b[0].begin(),
        Paabbs.b[1].begin(),
        Paabbs.b[2].begin(),
        Paabbs.e[0].begin(),
        Paabbs.e[1].begin(),
        Paabbs.e[2].begin());
    thrust::sort_by_key(morton.codes.Data(), morton.codes.Data() + nPts, zip);
    // 3. Compute aabbs of the swept triangle volumes
    Faabbs.Construct([xt = xt.Raw(), x = xtp1.Raw(), F = F.Raw()] PBAT_DEVICE(GpuIndex f) {
        using namespace pbat::math::linalg::mini;
        SMatrix<GpuScalar, 3, 6> xf;
        xf.Col(0) = FromBuffers<3, 1>(xt, F[0][f]);
        xf.Col(1) = FromBuffers<3, 1>(xt, F[1][f]);
        xf.Col(2) = FromBuffers<3, 1>(xt, F[2][f]);
        xf.Col(3) = FromBuffers<3, 1>(x, F[0][f]);
        xf.Col(4) = FromBuffers<3, 1>(x, F[1][f]);
        xf.Col(5) = FromBuffers<3, 1>(x, F[2][f]);
        SMatrix<GpuScalar, 3, 2> LU;
        pbat::common::ForRange<0, kDims>([&]<auto d>() {
            LU(d, 0) = Min(xf.Row(d));
            LU(d, 1) = Max(xf.Row(d));
        });
        return LU;
    });
    // 4. Construct the bvh of the swept triangle volumes
    Fbvh.Build(Faabbs, wmin, wmax);
    // 5. Detect overlaps between swept points and swept triangle volumes
    using namespace pbat::math::linalg;
    Fbvh.RangeSearch(
        Faabbs,
        nPts,
        [b = Paabbs.b.Raw(), e = Paabbs.e.Raw()] PBAT_DEVICE(GpuIndex q) {
            mini::SMatrix<GpuScalar, 3, 2> LU;
            LU.Col(0) = mini::FromBuffers<3, 1>(b, q);
            LU.Col(1) = mini::FromBuffers<3, 1>(e, q);
            return LU;
        } /*fGetQueryObject*/,
        [b = Faabbs.b.Raw(),
         e = Faabbs.e.Raw()] PBAT_DEVICE(GpuIndex leaf, [[maybe_unused]] GpuIndex i) {
            mini::SMatrix<GpuScalar, 3, 2> LU;
            LU.Col(0) = mini::FromBuffers<3, 1>(b, leaf);
            LU.Col(1) = mini::FromBuffers<3, 1>(e, leaf);
            return LU;
        } /*fGetLeafObject*/,
        [] PBAT_DEVICE(
            mini::SMatrix<GpuScalar, 3, 2> const& LU1,
            mini::SVector<GpuScalar, 3> const& L2,
            mini::SVector<GpuScalar, 3> const& U2) {
            bool const bBoxesOverlap = pbat::geometry::OverlapQueries::AxisAlignedBoundingBoxes(
                LU1.Col(0),
                LU1.Col(1),
                L2,
                U2);
            return static_cast<GpuScalar>(not bBoxesOverlap);
        } /*fMinDistanceToBox*/,
        [] PBAT_DEVICE(
            mini::SMatrix<GpuScalar, 3, 2> const& LU1,
            mini::SMatrix<GpuScalar, 3, 2> const& LU2) {
            return GpuScalar(0);
        } /*fDistanceToLeaf*/,
        [] PBAT_DEVICE(GpuIndex q) {
            return GpuScalar(0);
        } /*fQueryUpperBound*/,
        [active = active.Raw(), inds = inds.Raw()] PBAT_DEVICE(
            GpuIndex q,
            [[maybe_unused]] GpuIndex f,
            [[maybe_unused]] GpuScalar d) {
            active[inds[q]] = true;
        } /*fOnFound*/);
    // 6. Compact active vertices in sorted order
    auto it = thrust::copy_if(
        inds.Data(),
        inds.Data() + nPts,
        av.Data(),
        [active = active.Raw()] PBAT_DEVICE(GpuIndex v) { return active[v]; });
    nActive = static_cast<GpuIndex>(thrust::distance(av.Data(), it));
}

void VertexTriangleMixedCcdDcd::UpdateActiveSet(common::Buffer<GpuScalar, 3> const& x)
{
    // 1. Recompute triangle AABBs
    Faabbs.Construct([x = x.Raw(), F = F.Raw(), finds = Fbvh.inds.Raw()] PBAT_DEVICE(GpuIndex i) {
        GpuIndex const f = finds[i];
        using namespace pbat::math::linalg::mini;
        auto fv                     = FromBuffers<3, 1>(F, f);
        SMatrix<GpuScalar, 3, 3> xf = FromBuffers(x, fv.Transpose());
        SMatrix<GpuScalar, 3, 2> LU;
        pbat::common::ForRange<0, kDims>([&]<auto d>() {
            LU(0, d) = Min(xf.Row(d));
            LU(1, d) = Max(xf.Row(d));
        });
        return LU;
    });
    // 2. Update BVH internal node boxes
    Fbvh.ConstructBoxes(Faabbs);
    // 3. Compute nearest neighbours and signed distances
    using namespace pbat::math::linalg::mini;
    nn.SetConstant(GpuIndex(-1));
    Fbvh.NearestNeighbours(
        Faabbs,
        nActive,
        [x = x.Raw(), V = V.Raw(), av = av.Raw()] PBAT_DEVICE(GpuIndex q) {
            GpuIndex const v = av[q];
            return FromBuffers<3, 1>(x, V[v]);
        } /*fGetQueryObject*/,
        [x = x.Raw(), F = F.Raw()] PBAT_DEVICE([[maybe_unused]] GpuIndex leaf, GpuIndex f) {
            auto fv                           = FromBuffers<3, 1>(F, f);
            SMatrix<GpuScalar, 3, 3> const xf = FromBuffers(x, fv.Transpose());
            return xf;
        } /*fGetLeafObject*/,
        [] PBAT_DEVICE(
            SVector<GpuScalar, 3> const& xi,
            SVector<GpuScalar, 3> const& L,
            SVector<GpuScalar, 3> const& U) {
            return pbat::geometry::DistanceQueries::PointAxisAlignedBoundingBox(xi, L, U);
        } /*fMinDistanceToBox*/,
        [] PBAT_DEVICE(SVector<GpuScalar, 3> const& xi, SMatrix<GpuScalar, 3, 3> const& xf) {
            return pbat::geometry::DistanceQueries::PointTriangle(
                xi,
                xf.Col(0),
                xf.Col(1),
                xf.Col(2));
        } /*fDistanceToLeaf*/,
        [sd = sd.Raw(), av = av.Raw()] PBAT_DEVICE(GpuIndex q) {
            GpuScalar const sd2 = sd[av[q]];
            return abs(sd2);
        } /*fDistanceUpperBound*/,
        [x  = x.Raw(),
         V  = V.Raw(),
         F  = F.Raw(),
         av = av.Raw(),
         sd = sd.Raw(),
         nn = nn.Raw()] PBAT_DEVICE(GpuIndex q, GpuIndex f, GpuScalar d2, GpuIndex k) {
            GpuIndex const v = av[q];
            // Compute signed squared distance
            SVector<GpuScalar, 3> const xv    = FromBuffers<3, 1>(x, V[v]);
            auto fv                           = FromBuffers<3, 1>(F, f);
            SMatrix<GpuScalar, 3, 3> const xf = FromBuffers(x, fv.Transpose());
            GpuScalar const sgn =
                pbat::geometry::DistanceQueries::PointPlane(xv, xf.Col(0), xf.Col(1), xf.Col(2)) >=
                GpuScalar(0);
            sd[v] = sgn * d2;
            // Add active contact (i,f)
            nn[v * kMaxNeighbours + k] = f;
        } /*fOnNearestNeighbourFound*/);
}

void VertexTriangleMixedCcdDcd::FinalizeActiveSet(
    [[maybe_unused]] common::Buffer<GpuScalar, 3> const& x)
{
    // TODO: Implement
}

} // namespace pbat::gpu::impl::contact