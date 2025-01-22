// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Bvh.cuh"
#include "BvhKernels.cuh"
#include "pbat/HostDevice.h"
#include "pbat/common/Stack.h"
#include "pbat/gpu/common/Eigen.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"

#include <array>
#include <cuda/atomic>
#include <exception>
#include <string>
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
            return sizeof(MortonCodeType) * 8 + __clz(i ^ j);
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

struct FDetectSelfOverlaps
{
    using OverlapType = typename pbat::gpu::geometry::impl::Bvh::OverlapType;

    PBAT_DEVICE bool AreSimplicesTopologicallyAdjacent(GpuIndex si, GpuIndex sj) const
    {
        auto count{0};
        for (auto i = 0; i < inds.size(); ++i)
            for (auto j = 0; j < inds.size(); ++j)
                count += (inds[i][si] == inds[j][sj]);
        return count > 0;
    }

    PBAT_DEVICE bool AreBoxesOverlapping(GpuIndex i, GpuIndex j) const
    {
        // clang-format off
        return (e[0][i] >= b[0][j]) and (b[0][i] <= e[0][j]) and
               (e[1][i] >= b[1][j]) and (b[1][i] <= e[1][j]) and
               (e[2][i] >= b[2][j]) and (b[2][i] <= e[2][j]);
        // clang-format on
    }

    PBAT_DEVICE void operator()(auto leaf)
    {
        // Traverse nodes depth-first starting from the root=0 node
        using pbat::common::Stack;
        Stack<GpuIndex, 64> stack{};
        stack.Push(0);
        do
        {
            assert(not stack.IsFull());
            GpuIndex const node = stack.Pop();
            // Check each child node for overlap.
            GpuIndex const lc = child[0][node];
            GpuIndex const rc = child[1][node];
            bool const bLeftBoxOverlaps =
                AreBoxesOverlapping(leaf, lc) and (rightmost[0][node] > leaf);
            bool const bRightBoxOverlaps =
                AreBoxesOverlapping(leaf, rc) and (rightmost[1][node] > leaf);

            // Leaf overlaps another leaf node -> report collision if topologically separate
            // simplices
            bool const bIsLeftLeaf = lc >= leafBegin;
            if (bLeftBoxOverlaps and bIsLeftLeaf)
            {
                GpuIndex const si = simplex[leaf - leafBegin];
                GpuIndex const sj = simplex[lc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(si, sj) and not overlaps.Append({si, sj}))
                    break;
            }
            bool const bIsRightLeaf = rc >= leafBegin;
            if (bRightBoxOverlaps and bIsRightLeaf)
            {
                GpuIndex const si = simplex[leaf - leafBegin];
                GpuIndex const sj = simplex[rc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(si, sj) and not overlaps.Append({si, sj}))
                    break;
            }

            // Leaf overlaps an internal node -> traverse
            bool const bTraverseLeft  = bLeftBoxOverlaps and not bIsLeftLeaf;
            bool const bTraverseRight = bRightBoxOverlaps and not bIsRightLeaf;
            if (bTraverseLeft)
                stack.Push(lc);
            if (bTraverseRight)
                stack.Push(rc);
        } while (not stack.IsEmpty());
    }

    GpuIndex* simplex;
    std::array<GpuIndex const*, 4> inds;
    std::array<GpuIndex*, 2> child;
    std::array<GpuIndex*, 2> rightmost;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuIndex leafBegin;
    common::DeviceSynchronizedList<OverlapType> overlaps;
};

} // namespace kernels

Bvh::Bvh(std::size_t nPrimitives, std::size_t nOverlaps)
    : simplex(nPrimitives),
      morton(nPrimitives),
      child(nPrimitives - 1),
      parent(2 * nPrimitives - 1),
      rightmost(nPrimitives - 1),
      b(2 * nPrimitives - 1),
      e(2 * nPrimitives - 1),
      visits(nPrimitives - 1),
      overlaps(nOverlaps)
{
    parent.SetConstant(GpuIndex(-1));
}

void Bvh::Build(
    Points const& P,
    Simplices const& S,
    Eigen::Vector<GpuScalar, 3> const& WL,
    Eigen::Vector<GpuScalar, 3> const& WU,
    GpuScalar expansion)
{
    auto const n = S.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < n)
    {
        std::string const what = "Allocated memory for " +
                                 std::to_string(NumberOfAllocatedBoxes()) +
                                 " boxes, but received " + std::to_string(n) + " simplices.";
        throw std::invalid_argument(what);
    }

    // 0. Reset intermediate data
    visits.SetConstant(GpuIndex(0));

    // 1. Construct leaf node (i.e. simplex) bounding boxes
    auto const leafBegin        = n - 1;
    auto const nSimplexVertices = static_cast<int>(S.eSimplexType);
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        BvhImplKernels::FLeafBoundingBoxes{
            P.x.Raw(),
            S.inds.Raw(),
            nSimplexVertices,
            b.Raw(),
            e.Raw(),
            leafBegin,
            expansion});

    // 2. Compute Morton codes for each leaf node (i.e. simplex)
    std::array<GpuScalar, 3> sb{WL(0), WL(1), WL(2)};
    std::array<GpuScalar, 3> sbe{WU(0) - WL(0), WU(1) - WL(1), WU(2) - WL(2)};
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        [sb, sbe, b = b.Raw(), e = e.Raw(), morton = morton.Raw(), leafBegin] PBAT_DEVICE(auto s) {
            auto const bs = leafBegin + s;
            // Compute Morton code of the centroid of the bounding box of simplex s
            std::array<GpuScalar, 3> c{};
            for (auto d = 0; d < 3; ++d)
            {
                auto cd = GpuScalar{0.5} * (b[d][bs] + e[d][bs]);
                c[d]    = (cd - sb[d]) / sbe[d];
            }
            using pbat::geometry::Morton3D;
            morton[s] = Morton3D(c);
        });

    // 3. Sort simplices based on Morton codes
    thrust::sequence(thrust::device, simplex.Data(), simplex.Data() + n);
    auto zip = thrust::make_zip_iterator(
        b[0].begin() + leafBegin,
        b[1].begin() + leafBegin,
        b[2].begin() + leafBegin,
        e[0].begin() + leafBegin,
        e[1].begin() + leafBegin,
        e[2].begin() + leafBegin,
        simplex.Data());
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
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(n - 1),
        thrust::make_counting_iterator(2 * n - 1),
        [parent = parent.Raw(),
         child  = child.Raw(),
         b      = b.Raw(),
         e      = e.Raw(),
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

                GpuIndex lc = child[0][p];
                GpuIndex rc = child[1][p];
                for (auto d = 0; d < 3; ++d)
                {
                    b[d][p] = min(b[d][lc], b[d][rc]);
                    e[d][p] = max(e[d][lc], e[d][rc]);
                }
                // Move up the binary tree
                p = parent[p];
            }
            assert(k < 64);
        });
}

void Bvh::DetectSelfOverlaps(Simplices const& S)
{
    auto const n = S.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < n)
    {
        std::string const what = "Allocated memory for " +
                                 std::to_string(NumberOfAllocatedBoxes()) +
                                 " boxes, but received " + std::to_string(n) + " simplices.";
        throw std::invalid_argument(what);
    }
    overlaps.Clear();
    auto const leafBegin = n - 1;
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(n - 1),
        thrust::make_counting_iterator(2 * n - 1),
        kernels::FDetectSelfOverlaps{
            simplex.Raw(),
            S.inds.Raw(),
            child.Raw(),
            rightmost.Raw(),
            b.Raw(),
            e.Raw(),
            leafBegin,
            overlaps.Raw()});
}

std::size_t Bvh::NumberOfAllocatedBoxes() const
{
    return simplex.Size();
}

Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Bvh::Min() const
{
    return common::ToEigen(b);
}

Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Bvh::Max() const
{
    return common::ToEigen(e);
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> Bvh::SimplexOrdering() const
{
    return common::ToEigen(simplex);
}

Eigen::Vector<typename Bvh::MortonCodeType, Eigen::Dynamic> Bvh::MortonCodes() const
{
    return common::ToEigen(morton);
}

Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Bvh::Child() const
{
    return common::ToEigen(child).transpose();
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> Bvh::Parent() const
{
    return common::ToEigen(parent);
}

Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Bvh::Rightmost() const
{
    return common::ToEigen(rightmost).transpose();
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> Bvh::Visits() const
{
    return common::ToEigen(visits);
}

} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <algorithm>
#include <doctest/doctest.h>
#include <unordered_set>

#pragma nv_diag_suppress 177

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
    using gpu::geometry::impl::Bvh;
    using gpu::geometry::impl::Points;
    using gpu::geometry::impl::Simplices;
    auto const assert_cube = [](Bvh const& bvh) {
        auto child = bvh.Child();
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
        auto parent = bvh.Parent();
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
        auto rightmost       = bvh.Rightmost();
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
        auto visits = bvh.Visits();
        CHECK_EQ(visits.rows(), 4);
        CHECK_EQ(visits.cols(), 1);
        bool const bTwoVisitsPerInternalNode = (visits.array() == 2).all();
        CHECK(bTwoVisitsPerInternalNode);
    };
    GpuScalar constexpr expansion = std::numeric_limits<GpuScalar>::epsilon();
    auto const Vmin = (V.topRows<3>().rowwise().minCoeff().array() - expansion).eval();
    auto const Vmax = (V.topRows<3>().rowwise().maxCoeff().array() + expansion).eval();
    SUBCASE("Connected non self-overlapping mesh")
    {
        // Arrange
        Points P(V);
        Simplices S(C);
        // Act
        Bvh bvh(S.NumberOfSimplices(), S.NumberOfSimplices());
        bvh.Build(P, S, Vmin, Vmax);
        bvh.DetectSelfOverlaps(S);
        // Assert
        assert_cube(bvh);
        auto overlaps = bvh.overlaps.Get();
        CHECK_EQ(overlaps.size(), 0ULL);
    }
    SUBCASE("Disconnected mesh")
    {
        V = V(Eigen::placeholders::all, C.reshaped()).eval();
        C.resize(4, C.cols());
        C.reshaped().setLinSpaced(0, static_cast<GpuIndex>(V.cols() - 1));
        // Arrange
        Points P(V);
        Simplices S(C);
        // Because we only support overlaps between i,j s.t. i<j to prevent duplicates, we use the
        // summation identity \sum_i=1^n i = n*(n+1)/2, and remove the n occurrences where i=j.
        auto const nExpectedOverlaps =
            (S.NumberOfSimplices() * (S.NumberOfSimplices() + 1) / 2) - S.NumberOfSimplices();
        // Act
        Bvh bvh(S.NumberOfSimplices(), nExpectedOverlaps);
        bvh.Build(P, S, Vmin, Vmax);
        bvh.DetectSelfOverlaps(S);
        // Assert
        assert_cube(bvh);
        auto overlaps = bvh.overlaps.Get();
        CHECK_EQ(overlaps.size(), static_cast<std::size_t>(nExpectedOverlaps));
    }
}
