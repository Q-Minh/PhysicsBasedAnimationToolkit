#ifndef PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH
#define PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH

#include "BvhImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Stack.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"

#include <array>
#include <cuda/atomic>
#include <cuda/std/cmath>

namespace pbat {
namespace gpu {
namespace geometry {
namespace BvhImplKernels {

struct FLeafBoundingBoxes
{
    __device__ void operator()(int s)
    {
        using namespace cuda::std;
        for (auto d = 0; d < 3; ++d)
        {
            auto bs  = leafBegin + s;
            b[d][bs] = x[d][inds[0][s]];
            e[d][bs] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][bs] = fminf(b[d][bs], x[d][inds[m][s]]);
                e[d][bs] = fmaxf(e[d][bs], x[d][inds[m][s]]);
            }
            b[d][bs] -= r;
            e[d][bs] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex const*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuIndex leafBegin;
    GpuScalar r;
};

struct FComputeMortonCode
{
    using MortonCodeType = typename BvhImpl::MortonCodeType;

    __device__ void operator()(int s)
    {
        auto const bs = leafBegin + s;
        // Compute Morton code of the centroid of the bounding box of simplex s
        std::array<GpuScalar, 3> c{0.f, 0.f, 0.f};
        for (auto d = 0; d < 3; ++d)
            c[d] += GpuScalar{0.5} * (b[d][bs] + e[d][bs]);
        morton[s] = common::Morton3D(c);
    }

    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    MortonCodeType* morton;
    GpuIndex leafBegin;
};

struct FGenerateHierarchy
{
    using MortonCodeType = typename BvhImpl::MortonCodeType;

    struct Range
    {
        GpuIndex i, j, l;
        int d;
    };

    __device__ int Delta(GpuIndex i, GpuIndex j) const
    {
        if (j < 0 or j >= n)
            return -1;
        if (i == j)
            return __clz(i ^ j);
        return __clz(morton[i] ^ morton[j]);
    }

    __device__ Range DetermineRange(GpuIndex i) const
    {
        // Compute range direction
        int const d = (Delta(i, i + 1) - Delta(i, i - 1)) >= 0;
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

    __device__ GpuIndex FindSplit(Range R) const
    {
        // Identical Morton codes => split the range in the middle.
        if (morton[R.i] == morton[R.j])
            return (R.i + R.j) >> 1;

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
        GpuIndex const gamma = R.i + s * R.d + min(R.d, 0);
        return gamma;
    }

    __device__ void operator()(auto in)
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
        rightmost[0][in] = gamma;
        rightmost[1][in] = j;
    }

    MortonCodeType const* morton;
    std::array<GpuIndex*, 2> child;
    GpuIndex* parent;
    std::array<GpuIndex*, 2> rightmost;
    GpuIndex leafBegin;
    GpuIndex n;
};

struct FInternalNodeBoundingBoxes
{
    __device__ void operator()(auto leaf)
    {
        using namespace cuda::std;
        auto p = parent[leaf];
        while (p >= 0)
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
                b[d][p] = fminf(b[d][lc], b[d][rc]);
                e[d][p] = fmaxf(e[d][lc], e[d][rc]);
            }
            // Move up the binary tree
            p = parent[p];
        }
    }

    GpuIndex const* parent;
    std::array<GpuIndex*, 2> child;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuIndex* visits;
};

struct FDetectSelfOverlaps
{
    using OverlapType = typename BvhImpl::OverlapType;

    __device__ bool AreSimplicesTopologicallyAdjacent(GpuIndex si, GpuIndex sj) const
    {
        for (auto i = 0; i < inds.size(); ++i)
            for (auto j = 0; j < inds.size(); ++j)
                if (inds[i][si] == inds[j][sj])
                    return true;
        return false;
    }

    __device__ bool AreBoxesOverlapping(GpuIndex i, GpuIndex j) const
    {
        // clang-format off
        return (e[0][i] >= b[0][j]) and (b[0][i] <= e[0][j]) and
               (e[1][i] >= b[1][j]) and (b[1][i] <= e[1][j]) and
               (e[2][i] >= b[2][j]) and (b[2][i] <= e[2][j]);
        // clang-format on
    }

    __device__ void operator()(auto leaf)
    {
        // Traverse nodes depth-first starting from the root.
        common::Stack<GpuIndex, 64> stack{};
        stack.Push(0);
        do
        {
            GpuIndex const node = stack.Pop();
            // Check each child node for overlap.
            GpuIndex const lc = child[0][node];
            GpuIndex const rc = child[1][node];
            bool const bLeftOverlaps =
                AreBoxesOverlapping(leaf, lc) and (rightmost[0][node] > leaf);
            bool const bRightOverlaps =
                AreBoxesOverlapping(leaf, rc) and (rightmost[1][node] > leaf);

            // Leaf overlaps another leaf node => report collision if topologically separate
            // simplices
            bool const bIsLeftLeaf = lc >= leafBegin;
            if (bLeftOverlaps and bIsLeftLeaf)
            {
                GpuIndex const si = simplex[leaf - leafBegin];
                GpuIndex const sj = simplex[lc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(si, sj) and not overlaps.Append({si, sj}))
                    break;
            }
            bool const bIsRightLeaf = rc >= leafBegin;
            if (bRightOverlaps and bIsRightLeaf)
            {
                GpuIndex const si = simplex[leaf - leafBegin];
                GpuIndex const sj = simplex[rc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(si, sj) and not overlaps.Append({si, sj}))
                    break;
            }

            // Leaf overlaps an internal node => traverse.
            bool const bTraverseLeft  = bLeftOverlaps and not bIsLeftLeaf;
            bool const bTraverseRight = bRightOverlaps and not bIsRightLeaf;
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

} // namespace BvhImplKernels
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH