// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Bvh.cuh"
#include "pbat/HostDevice.h"

#include <cuda/atomic>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {
namespace kernels {

namespace mini = pbat::math::linalg::mini;

struct FGenerateHierarchy
{
    using MortonCodeType = typename Bvh::MortonCodeType;

    struct Range
    {
        GpuIndex i, j, l;
        int d;
    };

    PBAT_DEVICE int Delta(GpuIndex i, GpuIndex j) const
    {
        if (j < 0 or j >= n)
            return -1;
        if (morton[i] == morton[j])
            return sizeof(MortonCodeType) * 8 /* #bits/byte */ + __clz(i ^ j);
        return __clz(morton[i] ^ morton[j]);
    }

    PBAT_DEVICE Range DetermineRange(GpuIndex i) const
    {
        // Compute range direction
        bool const dsign = (Delta(i, i + 1) - Delta(i, i - 1)) > 0;
        int const d      = 2 * dsign - 1;
        // Lower bound on length of internal node i's common prefix
        int const dmin = Delta(i, i - d);
        // Compute conservative upper bound on the range's size
        GpuIndex lmax{2};
        while (Delta(i, i + lmax * d) > dmin)
            lmax <<= 1;
        // Binary search in the "inflated" range for the actual end (or start) of internal node i's
        // range, considering that i is its start (or end).
        GpuIndex l{0};
        do
        {
            lmax >>= 1;
            if (Delta(i, i + (l + lmax) * d) > dmin)
                l += lmax;
        } while (lmax > 1);
        GpuIndex j = i + l * d;
        return Range{i, j, l, d};
    }

    PBAT_DEVICE GpuIndex FindSplit(Range R) const
    {
        // Calculate the number of highest bits that are the same
        // for all objects.
        int const dnode = Delta(R.i, R.j);

        // Use binary search to find where the next bit differs.
        // Specifically, we are looking for the highest object that
        // shares more than dnode bits with the first one.
        GpuIndex s{0};
        do
        {
            R.l = (R.l + 1) >> 1;
            if (Delta(R.i, R.i + (s + R.l) * R.d) > dnode)
                s += R.l;
        } while (R.l > 1);
        GpuIndex gamma = R.i + s * R.d + min(R.d, 0);
        return gamma;
    }

    PBAT_DEVICE void operator()(auto in)
    {
        // Find out which range of objects the node corresponds to.
        Range R = DetermineRange(in);
        // Determine where to split the range.
        GpuIndex gamma = FindSplit(R);
        // Select left+right child
        GpuIndex i  = min(R.i, R.j);
        GpuIndex j  = max(R.i, R.j);
        GpuIndex lc = (i == gamma) ? leafBegin + gamma : gamma;
        GpuIndex rc = (j == gamma + 1) ? leafBegin + gamma + 1 : gamma + 1;
        // Record parent-child relationships
        child[0][in] = lc;
        child[1][in] = rc;
        parent[lc]   = in;
        parent[rc]   = in;
        // Record subtree relationships
        rightmost[0][in] = leafBegin + gamma;
        rightmost[1][in] = leafBegin + j;
    }

    MortonCodeType const* morton;
    std::array<GpuIndex*, 2> child;
    GpuIndex* parent;
    std::array<GpuIndex*, 2> rightmost;
    GpuIndex leafBegin;
    GpuIndex n;
};

} // namespace kernels

Bvh::Bvh(GpuIndex nBoxes)
    : inds(nBoxes),
      morton(nBoxes),
      child(nBoxes - 1),
      parent(2 * nBoxes - 1),
      rightmost(nBoxes - 1),
      iaabbs(nBoxes - 1),
      visits(nBoxes - 1)
{
    parent.SetConstant(GpuIndex(-1));
}

void Bvh::Build(Aabb<kDims>& aabbs, Morton::Bound const& WL, Morton::Bound const& WU)
{
    using namespace pbat::math::linalg;
    GpuIndex const n                = aabbs.Size();
    GpuIndex const leafBegin        = n - 1;
    common::Buffer<GpuScalar, 3>& b = aabbs.b;
    common::Buffer<GpuScalar, 3>& e = aabbs.e;

    // 1. Reset intermediate data
    visits.SetConstant(GpuIndex(0));

    // 2. Compute Morton codes for each leaf node
    Morton::Encode(aabbs, WL, WU, morton);

    // 3. Sort leaves based on Morton codes
    thrust::sequence(thrust::device, inds.Data(), inds.Data() + n);
    auto zip = thrust::make_zip_iterator(
        b[0].begin(),
        b[1].begin(),
        b[2].begin(),
        e[0].begin(),
        e[1].begin(),
        e[2].begin(),
        inds.Data());
    // Using a stable sort preserves the initial ordering of simplex indices 0...n-1, resulting in
    // simplices sorted by Morton codes first, and then by simplex index.
    thrust::stable_sort_by_key(thrust::device, morton.Data(), morton.Data() + n, zip);

    // 4. Construct hierarchy
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n - 1),
        kernels::FGenerateHierarchy{
            morton.Raw(),
            child.Raw(),
            parent.Raw(),
            rightmost.Raw(),
            leafBegin,
            n});

    // 5. Construct internal node bounding boxes
    auto& ib = iaabbs.b;
    auto& ie = iaabbs.e;
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(n - 1),
        thrust::make_counting_iterator(2 * n - 1),
        [leafBegin,
         parent = parent.Raw(),
         child  = child.Raw(),
         b      = b.Raw(),
         e      = e.Raw(),
         ib     = ib.Raw(),
         ie     = ie.Raw(),
         visits = visits.Raw()] PBAT_DEVICE(auto leaf) {
            auto p = parent[leaf];
            auto k = 0;
            for (; (k < 64) and (p >= 0); ++k)
            {
                cuda::atomic_ref<GpuIndex, cuda::thread_scope_device> ap{visits[p]};
                // The first thread that gets access to the internal node p will terminate,
                // while the second thread visiting p will be allowed to continue execution.
                // This ensures that there is no race condition where a thread can access an
                // internal node too early, i.e. before both children of the internal node
                // have finished computing their bounding boxes.
                if (ap++ == 0)
                    break;

                GpuIndex lc             = child[0][p];
                GpuIndex rc             = child[1][p];
                bool const bIsLeftLeaf  = lc >= leafBegin;
                bool const bIsRightLeaf = rc >= leafBegin;
                lc -= bIsLeftLeaf * leafBegin;
                rc -= bIsRightLeaf * leafBegin;
                auto bl = bIsLeftLeaf ? b : ib;
                auto el = bIsLeftLeaf ? e : ie;
                auto br = bIsRightLeaf ? b : ib;
                auto er = bIsRightLeaf ? e : ie;
                for (auto d = 0; d < 3; ++d)
                {
                    ib[d][p] = min(bl[d][lc], br[d][rc]);
                    ie[d][p] = max(el[d][lc], er[d][rc]);
                }
                // Move up the binary tree
                p = parent[p];
            }
            assert(k < 64);
        });
}

} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Eigen.h"
#include "pbat/gpu/common/SynchronizedList.cuh"

#include <algorithm>
#include <cuda/std/utility>
#include <doctest/doctest.h>
#include <unordered_set>

#pragma nv_diag_suppress 177

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {
namespace test {
namespace Bvh {

struct FOnOverlapDetected
{
    using Overlap = cuda::std::pair<GpuIndex, GpuIndex>;
    std::array<GpuIndex*, 4> T;
    gpu::common::DeviceSynchronizedList<Overlap> o;
    PBAT_DEVICE void operator()(GpuIndex si, GpuIndex sj)
    {
        using namespace pbat::math::linalg::mini;
        auto ti = FromBuffers<4, 1>(T, si);
        auto tj = FromBuffers<4, 1>(T, sj);
        bool bConnected{false};
        pbat::common::ForRange<0, 4>([&]<auto d>() { bConnected |= Any(tj == ti[d]); });
        if (not bConnected)
        {
            o.Append(Overlap{si, sj});
        }
    };
};

} // namespace Bvh
} // namespace test
} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#include "pbat/gpu/common/Eigen.cuh"

TEST_CASE("[gpu][geometry][impl] Bvh")
{
    using namespace pbat;
    // Cube mesh
    GpuMatrixX V(3, 8);
    GpuIndexMatrixX C(4, 5);
    // clang-format off
    V << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    C << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    using gpu::geometry::impl::Aabb;
    using gpu::geometry::impl::Bvh;
    auto Vmin                = V.topRows<3>().rowwise().minCoeff().eval();
    auto Vmax                = V.topRows<3>().rowwise().maxCoeff().eval();
    using Overlap            = cuda::std::pair<GpuIndex, GpuIndex>;
    using Overlaps           = gpu::common::SynchronizedList<Overlap>;
    using FOnOverlapDetected = gpu::geometry::impl::test::Bvh::FOnOverlapDetected;
    using namespace math::linalg;
    auto const fCheckCubeBvhTopology = [](Bvh const& bvh) {
        auto child = gpu::common::ToEigen(bvh.child).transpose().eval();
        CHECK_EQ(child.rows(), 4);
        CHECK_EQ(child.cols(), 2);
        CHECK_EQ(child(0, 0), 3);
        CHECK_EQ(child(0, 1), 8);
        CHECK_EQ(child(1, 0), 4);
        CHECK_EQ(child(1, 1), 5);
        CHECK_EQ(child(2, 0), 6);
        CHECK_EQ(child(2, 1), 7);
        CHECK_EQ(child(3, 0), 1);
        CHECK_EQ(child(3, 1), 2);
        auto parent = gpu::common::ToEigen(bvh.parent);
        CHECK_EQ(parent.rows(), 9);
        CHECK_EQ(parent.cols(), 1);
        CHECK_EQ(parent(0), GpuIndex{-1});
        CHECK_EQ(parent(1), 3);
        CHECK_EQ(parent(2), 3);
        CHECK_EQ(parent(3), 0);
        CHECK_EQ(parent(4), 1);
        CHECK_EQ(parent(5), 1);
        CHECK_EQ(parent(6), 2);
        CHECK_EQ(parent(7), 2);
        CHECK_EQ(parent(8), 0);
        auto rightmost       = gpu::common::ToEigen(bvh.rightmost).transpose().eval();
        auto const leafBegin = 4;
        CHECK_EQ(rightmost.rows(), 4);
        CHECK_EQ(rightmost.cols(), 2);
        CHECK_EQ(rightmost(0, 0), leafBegin + 3);
        CHECK_EQ(rightmost(0, 1), leafBegin + 4);
        CHECK_EQ(rightmost(1, 0), leafBegin + 0);
        CHECK_EQ(rightmost(1, 1), leafBegin + 1);
        CHECK_EQ(rightmost(2, 0), leafBegin + 2);
        CHECK_EQ(rightmost(2, 1), leafBegin + 3);
        CHECK_EQ(rightmost(3, 0), leafBegin + 1);
        CHECK_EQ(rightmost(3, 1), leafBegin + 3);
        auto visits = gpu::common::ToEigen(bvh.visits);
        CHECK_EQ(visits.rows(), 4);
        CHECK_EQ(visits.cols(), 1);
        bool const bTwoVisitsPerInternalNode = (visits.array() == 2).all();
        CHECK(bTwoVisitsPerInternalNode);
    };
    auto const fCheckInternalBoundingBoxComputation = [](Bvh const& bvh, Aabb<3> const& aabbs) {
        GpuIndexMatrixX child     = gpu::common::ToEigen(bvh.child);
        GpuMatrixX ib             = gpu::common::ToEigen(bvh.iaabbs.b);
        GpuMatrixX ie             = gpu::common::ToEigen(bvh.iaabbs.e);
        GpuMatrixX b              = gpu::common::ToEigen(aabbs.b);
        GpuMatrixX e              = gpu::common::ToEigen(aabbs.e);
        auto const nInternalNodes = bvh.iaabbs.Size();
        // For all internal nodes, check that their bounding box is the "union" of their children's
        for (auto i = 0; i < nInternalNodes; ++i)
        {
            auto lc           = child(0, i);
            auto rc           = child(1, i);
            auto lmin         = lc < nInternalNodes ? ib.col(lc).head<3>().eval() :
                                                      b.col(lc - nInternalNodes).head<3>().eval();
            auto lmax         = lc < nInternalNodes ? ie.col(lc).head<3>().eval() :
                                                      e.col(lc - nInternalNodes).head<3>().eval();
            auto rmin         = rc < nInternalNodes ? ib.col(rc).head<3>().eval() :
                                                      b.col(rc - nInternalNodes).head<3>().eval();
            auto rmax         = rc < nInternalNodes ? ie.col(rc).head<3>().eval() :
                                                      e.col(rc - nInternalNodes).head<3>().eval();
            auto iminExpected = lmin.cwiseMin(rmin).head<3>().eval();
            auto imaxExpected = lmax.cwiseMax(rmax).head<3>().eval();
            auto imin         = ib.col(i).head<3>().eval();
            auto imax         = ie.col(i).head<3>().eval();
            CHECK(imin.isApprox(iminExpected));
            CHECK(imax.isApprox(imaxExpected));
        }
    };
    SUBCASE("Connected non self-overlapping mesh")
    {
        // Arrange
        gpu::common::Buffer<GpuScalar, 3> VG(V.cols());
        gpu::common::ToBuffer(V, VG);
        gpu::common::Buffer<GpuIndex, 4> CG(C.cols());
        gpu::common::ToBuffer(C, CG);
        Aabb<3> aabbs{VG, CG};
        Overlaps overlaps(1);
        // Act
        Bvh bvh(aabbs.Size());
        bvh.Build(aabbs, mini::FromEigen(Vmin), mini::FromEigen(Vmax));
        bvh.DetectOverlaps(aabbs, FOnOverlapDetected{CG.Raw(), overlaps.Raw()});
        // Assert
        CHECK_EQ(overlaps.Size(), 0);
        fCheckCubeBvhTopology(bvh);
        fCheckInternalBoundingBoxComputation(bvh, aabbs);
    }
    SUBCASE("Disconnected mesh")
    {
        V = V(Eigen::placeholders::all, C.reshaped()).eval();
        C.resize(4, C.cols());
        C.reshaped().setLinSpaced(0, static_cast<GpuIndex>(V.cols() - 1));
        // Arrange
        gpu::common::Buffer<GpuScalar, 3> VG(V.cols());
        gpu::common::ToBuffer(V, VG);
        gpu::common::Buffer<GpuIndex, 4> CG(C.cols());
        gpu::common::ToBuffer(C, CG);
        Aabb<3> aabbs{VG, CG};
        // Because we only support overlaps between i,j s.t. i<j to prevent duplicates, we use the
        // summation identity \sum_i=1^n i = n*(n+1)/2, and remove the n occurrences where i=j.
        auto const nSimplices        = aabbs.Size();
        auto const nExpectedOverlaps = (nSimplices * (nSimplices + 1) / 2) - nSimplices;
        Overlaps overlaps(2 * nExpectedOverlaps);
        // Act
        Bvh bvh(aabbs.Size());
        bvh.Build(aabbs, mini::FromEigen(Vmin), mini::FromEigen(Vmax));
        bvh.DetectOverlaps(aabbs, FOnOverlapDetected{CG.Raw(), overlaps.Raw()});
        // Assert
        CHECK_EQ(overlaps.Size(), nExpectedOverlaps);
        fCheckCubeBvhTopology(bvh);
        fCheckInternalBoundingBoxComputation(bvh, aabbs);
    }
    SUBCASE("Non-overlapping line segment collection")
    {
        // Arrange
        // Bunch of disconnected line segments that do not overlap.
        V.setZero(3, 10);
        for (auto d = 0; d < 3; ++d)
        {
            V.row(d).setLinSpaced(GpuScalar(0), GpuScalar(9));
            V.row(d).reshaped(2, 5).row(1).array() -= GpuScalar(0.1);
        }
        Vmin = V.rowwise().minCoeff().eval();
        Vmax = V.rowwise().maxCoeff().eval();
        C.resize(4, 5);
        C.topRows(2).reshaped().setLinSpaced(0, static_cast<GpuIndex>(V.cols() - 1));
        C.bottomRows(2).reshaped().setLinSpaced(0, static_cast<GpuIndex>(V.cols() - 1));
        // Swap some columns of C to make sure that the order of simplices does not matter.
        C.col(0).swap(C.col(1));
        C.col(2).swap(C.col(3));
        gpu::common::Buffer<GpuScalar, 3> VG(V.cols());
        gpu::common::ToBuffer(V, VG);
        gpu::common::Buffer<GpuIndex, 4> CG(C.cols());
        gpu::common::ToBuffer(C, CG);
        Aabb<3> aabbs{VG, CG};
        GpuIndex const nExpectedOverlaps{0};
        Overlaps overlaps(2 * nExpectedOverlaps);

        // Act
        Bvh bvh(aabbs.Size());
        bvh.Build(aabbs, mini::FromEigen(Vmin), mini::FromEigen(Vmax));
        bvh.DetectOverlaps(aabbs, FOnOverlapDetected{CG.Raw(), overlaps.Raw()});

        // Assert
        CHECK_EQ(overlaps.Size(), nExpectedOverlaps);
        fCheckInternalBoundingBoxComputation(bvh, aabbs);
    }
}
