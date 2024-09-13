#ifndef PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH
#define PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH

#include "BvhQueryImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Morton.cuh"
#include "pbat/gpu/common/Stack.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"

#include <array>
#include <cuda/std/cmath>

namespace pbat {
namespace gpu {
namespace geometry {
namespace BvhQueryImplKernels {

struct FComputeAabb
{
    __device__ void operator()(int s)
    {
        for (auto d = 0; d < 3; ++d)
        {
            b[d][s] = x[d][inds[0][s]];
            e[d][s] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][s] = cuda::std::min(b[d][s], x[d][inds[m][s]]);
                e[d][s] = cuda::std::max(e[d][s], x[d][inds[m][s]]);
            }
            b[d][s] -= r;
            e[d][s] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex const*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuScalar r;
};

struct FComputeMortonCode
{
    using MortonCodeType = common::MortonCodeType;

    __device__ void operator()(int s)
    {
        // Compute Morton code of the centroid of the bounding box of simplex s
        std::array<GpuScalar, 3> c{};
        for (auto d = 0; d < 3; ++d)
        {
            auto cd = GpuScalar{0.5} * (b[d][s] + e[d][s]);
            c[d]    = (cd - sb[d]) / sbe[d];
        }
        morton[s] = common::Morton3D(c);
    }

    std::array<GpuScalar, 3> sb;
    std::array<GpuScalar, 3> sbe;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    MortonCodeType* morton;
};

struct FDetectOverlaps
{
    __device__ bool AreSimplicesTopologicallyAdjacent(GpuIndex si, GpuIndex sj) const
    {
        auto count{0};
        for (auto i = 0; i < queryInds.size(); ++i)
            for (auto j = 0; j < inds.size(); ++j)
                count += (queryInds[i][si] == inds[j][sj]);
        return count > 0;
    }

    __device__ bool AreBoxesOverlapping(GpuIndex i, GpuIndex j) const
    {
        // clang-format off
        return (queryE[0][i] >= b[0][j]) and (queryB[0][i] <= e[0][j]) and
               (queryE[1][i] >= b[1][j]) and (queryB[1][i] <= e[1][j]) and
               (queryE[2][i] >= b[2][j]) and (queryB[2][i] <= e[2][j]);
        // clang-format on
    }

    __device__ void operator()(auto query)
    {
        // Traverse nodes depth-first starting from the root.
        common::Stack<GpuIndex, 64> stack{};
        stack.Push(0);
        do
        {
            assert(not stack.IsFull());
            GpuIndex const node = stack.Pop();
            // Check each child node for overlap.
            GpuIndex const lc            = child[0][node];
            GpuIndex const rc            = child[1][node];
            bool const bLeftBoxOverlaps  = AreBoxesOverlapping(query, lc);
            bool const bRightBoxOverlaps = AreBoxesOverlapping(query, rc);

            // Query overlaps another leaf node -> report collision if topologically separate
            // simplices
            bool const bIsLeftLeaf = lc >= leafBegin;
            if (bLeftBoxOverlaps and bIsLeftLeaf)
            {
                GpuIndex const si = querySimplex[query];
                GpuIndex const sj = simplex[lc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(
                        si,
                        sj) /* and AreSimplicesOverlapping(si, sj) */
                    and not overlaps.Append({si, sj}))
                    break;
            }
            bool const bIsRightLeaf = rc >= leafBegin;
            if (bRightBoxOverlaps and bIsRightLeaf)
            {
                GpuIndex const si = querySimplex[query];
                GpuIndex const sj = simplex[rc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(
                        si,
                        sj) /* and AreSimplicesOverlapping(si, sj) */
                    and not overlaps.Append({si, sj}))
                    break;
            }

            // Query overlaps an internal node -> traverse.
            bool const bTraverseLeft  = bLeftBoxOverlaps and not bIsLeftLeaf;
            bool const bTraverseRight = bRightBoxOverlaps and not bIsRightLeaf;
            if (bTraverseLeft)
                stack.Push(lc);
            if (bTraverseRight)
                stack.Push(rc);
        } while (not stack.IsEmpty());
    }

    std::array<GpuScalar const*, 3> x;

    GpuIndex* querySimplex;
    std::array<GpuIndex const*, 4> queryInds;
    std::array<GpuScalar*, 3> queryB;
    std::array<GpuScalar*, 3> queryE;

    GpuIndex const* simplex;
    std::array<GpuIndex const*, 4> inds;
    std::array<GpuScalar const*, 3> b;
    std::array<GpuScalar const*, 3> e;
    std::array<GpuIndex const*, 2> child;
    GpuIndex leafBegin;

    common::DeviceSynchronizedList<typename BvhQueryImpl::OverlapType> overlaps;
};

} // namespace BvhQueryImplKernels
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH
