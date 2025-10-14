#ifndef PBAT_GPU_IMPL_GEOMETRY_SWEEPANDPRUNE_H
#define PBAT_GPU_IMPL_GEOMETRY_SWEEPANDPRUNE_H

#include "Aabb.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/profiling/Profiling.h"

#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pbat {
namespace gpu {
namespace impl {
namespace geometry {

/**
 * @brief Single-axis parallel sweep and prune algorithm for detecting overlapping bounding boxes.
 *
 * Implements @cite david1992dynamic
 *
 */
class SweepAndPrune
{
  public:
    static auto constexpr kDims = 3;

    /**
     * @brief Construct a new Sweep And Prune object
     *
     * @param nPrimitives
     */
    SweepAndPrune(std::size_t nPrimitives = 0ULL);
    /**
     * @brief
     *
     * @param nPrimitives
     */
    void Reserve(std::size_t nPrimitives);
    /**
     * @brief Compute overlapping boxes in aabbs
     * @param aabbs
     */
    template <class FOnOverlapDetected>
    void SortAndSweep(Aabb<kDims>& aabbs, FOnOverlapDetected&& fOnOverlapDetected);

  private:
    common::Buffer<GpuIndex> inds; ///< Box indices
};

template <class FOnOverlapDetected>
inline void SweepAndPrune::SortAndSweep(Aabb<kDims>& aabbs, FOnOverlapDetected&& fOnOverlapDetected)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.geometry.SweepAndPrune.SortAndSweep");

    // 1. Preprocess internal data
    auto const nBoxes = static_cast<GpuIndex>(aabbs.Size());
    if (inds.Size() < nBoxes)
        inds.Resize(nBoxes);
    thrust::sequence(thrust::device, inds.Data(), inds.Data() + nBoxes);

    // 2. Compute mean and variance of bounding box centers
    // NOTE:
    // We could use the streams API here to parallelize the computation of mu and sigma along each
    // dimension.
    auto& b = aabbs.b;
    auto& e = aabbs.e;
    std::array<GpuScalar, kDims> mu{}, sigma{};
    {
        PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.geometry.SweepAndPrune.MeanVariance");
        for (auto d = 0; d < kDims; ++d)
        {
            mu[d] = thrust::transform_reduce(
                thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(nBoxes),
                cuda::proclaim_return_type<GpuScalar>(
                    [b = b.Raw()[d], e = e.Raw()[d], div = 2 * nBoxes] PBAT_DEVICE(
                        GpuIndex i) -> GpuScalar { return (b[i] + e[i]) / div; }),
                GpuScalar(0),
                thrust::plus<GpuScalar>());
        }
        for (auto d = 0; d < kDims; ++d)
        {
            sigma[d] = thrust::transform_reduce(
                thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(nBoxes),
                cuda::proclaim_return_type<GpuScalar>(
                    [b = b.Raw()[d], e = e.Raw()[d], mu = mu[d], nBoxes] PBAT_DEVICE(
                        GpuIndex i) -> GpuScalar {
                        GpuScalar cd       = (b[i] + e[i]) / GpuScalar(2);
                        GpuScalar const dx = cd - mu;
                        return dx * dx / nBoxes;
                    }),
                GpuScalar(0),
                thrust::plus<GpuScalar>());
        }
    }

    // 3. Sort bounding boxes along largest variance axis
    GpuIndex const saxis =
        (sigma[0] > sigma[1]) ? (sigma[0] > sigma[2] ? 0 : 2) : (sigma[1] > sigma[2] ? 1 : 2);
    std::array<GpuIndex, kDims - 1> axis{};
    pbat::common::ForRange<1, kDims>([&]<auto d>() { axis[d - 1] = (saxis + d) % kDims; });
    {
        PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.geometry.SweepAndPrune.Sort");
        auto zip = thrust::make_zip_iterator(
            b[axis[0]].begin(),
            b[axis[1]].begin(),
            e[saxis].begin(),
            e[axis[0]].begin(),
            e[axis[1]].begin(),
            inds.Data());
        thrust::sort_by_key(thrust::device, b[saxis].begin(), b[saxis].end(), zip);
    }

    // 4. Sweep to find overlaps
    {
        PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.geometry.SweepAndPrune.Sweep");
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nBoxes),
            [nBoxes,
             saxis,
             axis,
             b                  = b.Raw(),
             e                  = e.Raw(),
             inds               = inds.Raw(),
             fOnOverlapDetected = std::forward<FOnOverlapDetected>(
                 fOnOverlapDetected)] PBAT_DEVICE(GpuIndex i) mutable {
                for (auto j = i + 1; (j < nBoxes) and (e[saxis][i] >= b[saxis][j]); ++j)
                {
                    // NOTE:
                    // We only need to compare along non-major axis'. Thus, we're not using the
                    // box-box overlap test from OverlapQueries, which naturally compares all axis'.
                    bool const bBoxesOverlap =
                        (e[axis[0]][i] >= b[axis[0]][j]) and (b[axis[0]][i] <= e[axis[0]][j]) and
                        (e[axis[1]][i] >= b[axis[1]][j]) and (b[axis[1]][i] <= e[axis[1]][j]);
                    if (bBoxesOverlap)
                    {
                        fOnOverlapDetected(inds[i], inds[j]);
                    }
                }
            });
    }
}

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_GEOMETRY_SWEEPANDPRUNE_H
