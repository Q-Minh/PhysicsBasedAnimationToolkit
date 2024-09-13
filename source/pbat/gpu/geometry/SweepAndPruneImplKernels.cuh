#ifndef PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH
#define PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH

#include "SweepAndPruneImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/SynchronizedList.cuh"

#include <array>
#include <cuda/atomic>

namespace pbat {
namespace gpu {
namespace geometry {
namespace SweepAndPruneImplKernels {

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
                b[d][s] = min(b[d][s], x[d][inds[m][s]]);
                e[d][s] = max(e[d][s], x[d][inds[m][s]]);
            }
            b[d][s] -= r;
            e[d][s] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuScalar r;
};

struct FComputeMean
{
    __device__ void operator()(int s)
    {
        cuda::atomic_ref<GpuScalar, cuda::thread_scope_device> amu[3] = {
            cuda::atomic_ref<GpuScalar, cuda::thread_scope_device>(mu[0]),
            cuda::atomic_ref<GpuScalar, cuda::thread_scope_device>(mu[1]),
            cuda::atomic_ref<GpuScalar, cuda::thread_scope_device>(mu[2])};
        for (auto d = 0; d < 3; ++d)
        {
            amu[d] += (b[d][s] + e[d][s]) / (2.f * static_cast<GpuScalar>(nBoxes));
        }
    }

    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuScalar* mu;
    GpuIndex nBoxes;
};

struct FComputeVariance
{
    __device__ void operator()(int s)
    {
        cuda::atomic_ref<GpuScalar, cuda::thread_scope_device> asigma[3] = {
            cuda::atomic_ref<GpuScalar, cuda::thread_scope_device>(sigma[0]),
            cuda::atomic_ref<GpuScalar, cuda::thread_scope_device>(sigma[1]),
            cuda::atomic_ref<GpuScalar, cuda::thread_scope_device>(sigma[2])};
        for (auto d = 0; d < 3; ++d)
        {
            GpuScalar const cd = (b[d][s] + e[d][s]) / 2.f;
            GpuScalar const dx = cd - mu[d];
            asigma[d] += dx * dx / static_cast<GpuScalar>(nBoxes);
        }
    }

    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuScalar* mu;
    GpuScalar* sigma;
    GpuIndex nBoxes;
};

struct FSweep
{
    using OverlapType = typename SweepAndPruneImpl::OverlapType;

    /**
     * @brief If (si,sj) are from the same simplex set, or if (si,sj) share a common vertex, they
     * should not be considered for overlap testing.
     * @param sinds Simplex vertex indices in both sets
     * @param nSimplices Number of simplices in each simplex set
     * @param si Index of first simplex in pair to test
     * @param sj Index of second simplex in pair to test
     * @return
     */
    __device__ bool
    AreSimplicesOverlapCandidates(GpuIndex si, GpuIndex sj, bool bIsSiFromSourceSet) const
    {
        auto count{0};
        bool const bIsSjFromSourceSet = binds[sj] < nSimplices[0];
        count += (bIsSiFromSourceSet == bIsSjFromSourceSet);
        for (auto i = 0; i < sinds.size(); ++i)
            for (auto j = 0; j < sinds.size(); ++j)
                count += (sinds[i][si] == sinds[j][sj]);
        return count == 0;
    }

    __device__ bool AreSimplexCandidatesOverlapping(GpuIndex si, GpuIndex sj) const
    {
        return (e[axis[0]][si] >= b[axis[0]][sj]) and (b[axis[0]][si] <= e[axis[0]][sj]) and
               (e[axis[1]][si] >= b[axis[1]][sj]) and (b[axis[1]][si] <= e[axis[1]][sj]);
    }

    __device__ void operator()(GpuIndex si)
    {
        bool const bIsSiFromSourceSet = binds[si] < nSimplices[0];
        for (auto sj = si + 1; (sj < nBoxes) and (e[saxis][si] >= b[saxis][sj]); ++sj)
        {
            if (not AreSimplicesOverlapCandidates(si, sj, bIsSiFromSourceSet) or
                not AreSimplexCandidatesOverlapping(si, sj))
                continue;

            auto const overlap = bIsSiFromSourceSet ?
                                     OverlapType{binds[si], binds[sj] - nSimplices[0]} :
                                     OverlapType{binds[sj], binds[si] - nSimplices[0]};
            if (not overlaps.Append(overlap))
                break;
        }
    }

    GpuIndex* binds;
    std::array<GpuIndex*, 4> sinds;
    std::array<GpuIndex, 2> nSimplices;
    std::array<GpuScalar*, 3> b, e;
    GpuIndex saxis;
    std::array<GpuIndex, 2> axis;
    GpuIndex nBoxes;
    common::DeviceSynchronizedList<OverlapType> overlaps;
};

} // namespace SweepAndPruneImplKernels
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVH_QUERY_IMPL_KERNELS_CUH