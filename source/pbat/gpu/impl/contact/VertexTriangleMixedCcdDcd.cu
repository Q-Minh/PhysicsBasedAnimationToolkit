// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VertexTriangleMixedCcdDcd.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"

#include <limits>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pbat::gpu::impl::contact {

VertexTriangleMixedCcdDcd::VertexTriangleMixedCcdDcd(
    Eigen::Ref<GpuIndexVectorX const> const& Bin,
    Eigen::Ref<GpuIndexVectorX const> const& Vin,
    Eigen::Ref<GpuIndexMatrixX const> const& Fin)
    : B(Bin.size()),
      V(Vin.size()),
      F(Fin.size()),
      inds(Vin.size()),
      morton(Vin.size()),
      Paabbs(static_cast<GpuIndex>(Vin.size())),
      Faabbs(static_cast<GpuIndex>(Fin.size())),
      Fbvh(static_cast<GpuIndex>(Fin.size())),
      active(Vin.size()),
      av(Vin.size()),
      nActive(0),
      nn(Vin.size() * kMaxNeighbours),
      distances(Vin.size())
{
    thrust::sequence(inds.Data(), inds.Data() + inds.Size());
    active.SetConstant(false);
    av.SetConstant(-1);
    distances.SetConstant(std::numeric_limits<GpuScalar>::max());
}

void VertexTriangleMixedCcdDcd::InitializeActiveSet(
    common::Buffer<GpuScalar, 3> const& xt,
    common::Buffer<GpuScalar, 3> const& xtp1,
    geometry::Morton::Bound const& wmin,
    geometry::Morton::Bound const& wmax)
{
    auto const nVerts = Paabbs.Size();
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
    thrust::sort_by_key(morton.codes.Data(), morton.codes.Data() + nVerts, zip);
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
    Fbvh.DetectOverlaps(
        Faabbs,
        Paabbs,
        [xt     = xt.Raw(),
         x      = xtp1.Raw(),
         V      = V.Raw(),
         F      = F.Raw(),
         B      = B.Raw(),
         inds   = inds.Raw(),
         active = active.Raw()] PBAT_DEVICE(GpuIndex q, GpuIndex f) {
            // If i is already active, skip costly checks
            GpuIndex const i = V[inds[q]];
            if (active[i])
                return;

            using namespace pbat::math::linalg::mini;
            // Reject potential self collisions
            auto fv                  = FromBuffers<3, 1>(F, f);
            bool const bFromSameBody = (B[i] == B[fv[0]]);
            if (bFromSameBody)
                return;
            // Check if swept point intersects swept triangle
            SMatrix<GpuScalar, 3, 3> const xtf = FromBuffers(xt, fv.Transpose());
            SMatrix<GpuScalar, 3, 3> const xf  = FromBuffers(x, fv.Transpose());
            SVector<GpuScalar, 3> const xtv    = FromBuffers<3, 1>(xt, i);
            SVector<GpuScalar, 3> const xv     = FromBuffers<3, 1>(x, i);
            bool const bIntersects = pbat::geometry::OverlapQueries::LineSegmentSweptTriangle3D(
                xtv,
                xv,
                xtf.Col(0),
                xtf.Col(1),
                xtf.Col(2),
                xf.Col(0),
                xf.Col(1),
                xf.Col(2));
            // Make particle i active since if it might penetrate
            active[i] = bIntersects;
        });
    // 6. Compact active vertices in sorted order
    auto it = thrust::copy_if(
        inds.Data(),
        inds.Data() + nVerts,
        av.Data(),
        [active = active.Raw()] PBAT_DEVICE(GpuIndex v) { return active[v]; });
    nActive = static_cast<GpuIndex>(thrust::distance(av.Data(), it));
}

void VertexTriangleMixedCcdDcd::UpdateActiveSet(common::Buffer<GpuScalar, 3> const& x)
{
    UpdateBvh(x);
    // Compute distance from V to F via nn search
    using namespace pbat::math::linalg::mini;
    nn.SetConstant(GpuIndex(-1));
    this->ForEachNearestNeighbour(
        x,
        [av = av.Raw(),
         d  = distances.Raw(),
         nn = nn.Raw()] PBAT_DEVICE(GpuIndex q, GpuIndex f, GpuScalar dmin, GpuIndex k) {
            GpuIndex const v = av[q];
            // Store approximate squared distance to surface
            d[v] = dmin;
            // Add active contact (i,f)
            nn[v * kMaxNeighbours + k] = f;
        });
}

void VertexTriangleMixedCcdDcd::FinalizeActiveSet(
    [[maybe_unused]] common::Buffer<GpuScalar, 3> const& x)
{
    UpdateBvh(x);
    this->ForEachNearestNeighbour(
        x,
        [x      = x.Raw(),
         V      = V.Raw(),
         F      = F.Raw(),
         active = active.Raw(),
         av     = av.Raw(),
         d      = distances.Raw(),
         nn     = nn.Raw()] PBAT_DEVICE(GpuIndex q, GpuIndex f, GpuScalar dmin, GpuIndex k) {
            GpuIndex const v = av[q];
            // Check if vertex has exited surface
            using namespace pbat::math::linalg::mini;
            auto xv = FromBuffers<3, 1>(x, V[v]);
            auto fv = FromBuffers<3, 1>(F, f);
            auto xf = FromBuffers(x, fv.Transpose());
            GpuScalar sgn =
                pbat::geometry::DistanceQueries::PointPlane(xv, xf.Col(0), xf.Col(1), xf.Col(2));
            sgn /= abs(sgn);
            // Remove inactive vertices
            bool const bIsPenetrating = sgn < GpuScalar(0);
            auto dmax                 = std::numeric_limits<GpuScalar>::max();
            d[v]                      = bIsPenetrating * dmin + (not bIsPenetrating) * dmax;
            active[v]                 = bIsPenetrating;
        });
}

void pbat::gpu::impl::contact::VertexTriangleMixedCcdDcd::UpdateBvh(
    common::Buffer<GpuScalar, 3> const& x)
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
}

} // namespace pbat::gpu::impl::contact