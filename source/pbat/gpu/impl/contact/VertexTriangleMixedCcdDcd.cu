// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VertexTriangleMixedCcdDcd.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/profiling/Profiling.h"
#include "pbat/math/linalg/mini/Mini.h"

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
    : av(Vin.size()),
      nActive(0),
      nn(Vin.size() * kMaxNeighbours),
      B(Bin.size()),
      V(Vin.size()),
      F(Fin.cols()),
      inds(Vin.size()),
      morton(Vin.size()),
      Paabbs(static_cast<GpuIndex>(Vin.size())),
      Faabbs(static_cast<GpuIndex>(Fin.cols())),
      Fbvh(static_cast<GpuIndex>(Fin.cols())),
      active(Vin.size()),
      dupper(Vin.size()),
      eps(std::numeric_limits<GpuScalar>::epsilon())
{
    thrust::sequence(inds.Data(), inds.Data() + inds.Size());
    active.SetConstant(false);
    av.SetConstant(-1);
    dupper.SetConstant(std::numeric_limits<GpuScalar>::max());

    common::ToBuffer(Bin, B);
    common::ToBuffer(Vin, V);
    common::ToBuffer(Fin, F);
}

void VertexTriangleMixedCcdDcd::InitializeActiveSet(
    common::Buffer<GpuScalar, 3> const& xt,
    common::Buffer<GpuScalar, 3> const& xtp1,
    geometry::Morton::Bound const& wmin,
    geometry::Morton::Bound const& wmax)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE(
        "pbat.gpu.impl.contact.VertexTriangleMixedCcdDcd.InitializeActiveSet");

    // 1. Compute aabbs of the swept points (i.e. line segments)
    Paabbs.Construct(
        [xt = xt.Raw(), x = xtp1.Raw(), V = V.Raw(), inds = inds.Raw()] PBAT_DEVICE(GpuIndex q) {
            GpuIndex const i = V[inds[q]];
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
    thrust::sort_by_key(morton.codes.Data(), morton.codes.Data() + morton.codes.Size(), zip);
    // 3. Compute aabbs of the swept triangle volumes
    Faabbs.Construct([xt = xt.Raw(), x = xtp1.Raw(), F = F.Raw()] PBAT_DEVICE(GpuIndex f) {
        using namespace pbat::math::linalg::mini;
        auto fv = FromBuffers<3, 1>(F, f);
        SMatrix<GpuScalar, 3, 6> xf;
        xf.Slice<3, 3>(0, 0) = FromBuffers(xt, fv.Transpose());
        xf.Slice<3, 3>(0, 3) = FromBuffers(x, fv.Transpose());
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
         active = active.Raw(),
         dupper = dupper.Raw()] PBAT_DEVICE(GpuIndex q, GpuIndex f) {
            // If i is already active, skip costly checks
            GpuIndex const v = inds[q];
            GpuIndex const i = V[v];
            using namespace pbat::math::linalg::mini;
            // Reject potential self collisions
            auto fv                  = FromBuffers<3, 1>(F, f);
            bool const bFromSameBody = (B[i] == B[fv[0]]);
            if (bFromSameBody)
                return;
            // Compute bounding box of both potentially colliding primitives, and use
            // its diagonal as the NN search radius
            SMatrix<GpuScalar, 3, 8> xvf;
            xvf.Slice<3, 3>(0, 0) = FromBuffers(xt, fv.Transpose());
            xvf.Slice<3, 3>(0, 3) = FromBuffers(x, fv.Transpose());
            xvf.Col(6)            = FromBuffers<3, 1>(xt, i);
            xvf.Col(7)            = FromBuffers<3, 1>(x, i);
            SVector<GpuScalar, 3> L;
            SVector<GpuScalar, 3> U;
            pbat::common::ForRange<0, kDims>([&]<auto d>() {
                L(d) = Min(xvf.Row(d));
                U(d) = Max(xvf.Row(d));
            });
            dupper[v] = SquaredNorm(U - L);
            // Make particle i active since it might penetrate
            active[v] = true;
        });
    // 6. Compact active vertices in sorted order
    auto it = thrust::copy_if(
        inds.Data(),
        inds.Data() + inds.Size(),
        av.Data(),
        [active = active.Raw()] PBAT_DEVICE(GpuIndex v) { return active[v]; });
    nActive = static_cast<GpuIndex>(thrust::distance(av.Data(), it));
}

void VertexTriangleMixedCcdDcd::UpdateActiveSet(
    common::Buffer<GpuScalar, 3> const& x,
    bool bComputeBoxes)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE(
        "pbat.gpu.impl.contact.VertexTriangleMixedCcdDcd.UpdateActiveSet");

    if (bComputeBoxes)
        UpdateBvh(x);
    // Compute distance from V to F via nn search
    nn.SetConstant(GpuIndex(-1));
    this->ForEachNearestNeighbour(
        x,
        [av = av.Raw(), nn = nn.Raw()] PBAT_DEVICE(
            GpuIndex q,
            GpuIndex f,
            [[maybe_unused]] GpuScalar dmin,
            GpuIndex k) {
            // Store approximate squared distance to surface
            // Add active contact (i,f)
            GpuIndex const v           = av[q];
            nn[v * kMaxNeighbours + k] = f;
        });
}

void VertexTriangleMixedCcdDcd::FinalizeActiveSet(
    common::Buffer<GpuScalar, 3> const& x,
    bool bComputeBoxes)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE(
        "pbat.gpu.impl.contact.VertexTriangleMixedCcdDcd.FinalizeActiveSet");

    if (bComputeBoxes)
        UpdateBvh(x);
    this->ForEachNearestNeighbour(
        x,
        [x      = x.Raw(),
         V      = V.Raw(),
         F      = F.Raw(),
         active = active.Raw(),
         av     = av.Raw(),
         nn     = nn.Raw()] PBAT_DEVICE(GpuIndex q, GpuIndex f, GpuScalar dmin, GpuIndex k) {
            GpuIndex const v = av[q];
            // Check if vertex has exited surface
            using namespace pbat::math::linalg::mini;
            auto xv = FromBuffers<3, 1>(x, V[v]);
            auto fv = FromBuffers<3, 1>(F, f);
            auto xf = FromBuffers(x, fv.Transpose());
            GpuScalar const sgn =
                pbat::geometry::DistanceQueries::PointPlane(xv, xf.Col(0), xf.Col(1), xf.Col(2));
            // Remove inactive vertices
            bool const bIsPenetrating = sgn < GpuScalar(0);
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
            LU(d, 0) = Min(xf.Row(d));
            LU(d, 1) = Max(xf.Row(d));
        });
        return LU;
    });
    // 2. Update BVH internal node boxes
    Fbvh.ConstructBoxes(Faabbs);
}

} // namespace pbat::gpu::impl::contact

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>
#include <ranges>

TEST_CASE("[gpu][impl][contact] VertexTriangleMixedCcdDcd.cu")
{
    using namespace pbat;
    using gpu::impl::common::Buffer;
    using gpu::impl::common::ToBuffer;
    using gpu::impl::common::ToEigen;
    using gpu::impl::contact::VertexTriangleMixedCcdDcd;
    using gpu::impl::geometry::Morton;

    SUBCASE("Bottom tetrahedron's top vertex penetrating top tetrahedron's bottom face")
    {
        // Arrange
        // (2 tets with bottom tet penetrating the top tet via bottom tet's top vertex through the
        // top tet's bottom face)
        auto constexpr nVerts        = 4;
        auto constexpr nCells        = 1;
        auto constexpr kDims         = 3;
        auto constexpr kFacesPerCell = 4;
        GpuMatrixX XT(kDims, 2 * nVerts);
        GpuIndexMatrixX T(4, 2 * nCells);
        GpuIndexMatrixX F(3, 2 * nCells * kFacesPerCell);
        GpuIndexVectorX V(2 * nVerts);
        // clang-format off
        XT << 0.f, 1.f, 0.f, 0.1f, 0.f  , 1.f  , 0.f  ,  0.1f,
            0.f, 0.f, 1.f, 0.1f, 0.f  , 0.f  , 1.f  ,  0.1f,
            0.f, 0.f, 0.f, 1.f , 1.01f, 1.01f, 1.01f, 2.01f;
        T << 0, 4,
            1, 5,
            2, 6,
            3, 7;
        F << 0, 1, 2, 0, 4, 5, 6, 4,
            1, 2, 0, 2, 5, 6, 4, 6,
            3, 3, 3, 1, 7, 7, 7, 5;
        V << 0, 1, 2, 3, 4, 5, 6, 7;
        // clang-format on
        GpuMatrixX X = XT;
        Eigen::Vector<GpuScalar, kDims> dX{0.f, 0.f, 0.01f};
        X.leftCols(nVerts).colwise() += dX;
        X.rightCols(nVerts).colwise() -= dX;
        GpuIndexVectorX B(2 * nVerts);
        B.head(nVerts).setZero();
        B.tail(nVerts).setOnes();
        Morton::Bound const wmin{0.f, 0.f, 0.f};
        Morton::Bound const wmax{1.f, 1.f, 2.01f};

        Buffer<GpuScalar, 3> xt(XT.cols());
        Buffer<GpuScalar, 3> x(X.cols());
        ToBuffer(XT, xt);
        ToBuffer(X, x);

        // Act
        VertexTriangleMixedCcdDcd ccd(B, V, F);
        SUBCASE("InitializeActiveSet")
        {
            // Act
            ccd.InitializeActiveSet(xt, x, wmin, wmax);
            // Assert
            auto active = ccd.active.Get();
            auto av     = ToEigen(ccd.av);

            auto const nExpectedActiveVertices = 4; ///< Bottom tet's top vertex (1 vert) passing
                                                    ///< through top tet's bottom triangle (3 verts)
            auto const nExpectedInactiveVertices = V.size() - nExpectedActiveVertices;
            CHECK_EQ(ccd.nActive, nExpectedActiveVertices);
            CHECK_EQ(std::ranges::count(active, true), nExpectedActiveVertices);
            CHECK((av.array() == 3).any());
            CHECK_EQ((av.array() == -1).count(), nExpectedInactiveVertices);
            SUBCASE("UpdateActiveSet")
            {
                // Act
                ccd.UpdateActiveSet(x);
                auto nn = ToEigen(ccd.nn)
                              .reshaped(ccd.kMaxNeighbours, ccd.nn.Size() / ccd.kMaxNeighbours)
                              .eval();
                // Assert
                for (auto v = 0ULL; v < active.size(); ++v)
                {
                    auto const nNearestNeighbours = (nn.col(v).array() != -1).count();
                    if (active[v])
                    {
                        CHECK_GT(nNearestNeighbours, 0);
                    }
                    else
                    {
                        CHECK_EQ(nNearestNeighbours, 0);
                    }
                }
                SUBCASE("FinalizeActiveSet")
                {
                    // Act
                    ccd.FinalizeActiveSet(x);
                    // Assert
                }
            }
        }
    }
    SUBCASE("Vertically stacked cubes") {}
}