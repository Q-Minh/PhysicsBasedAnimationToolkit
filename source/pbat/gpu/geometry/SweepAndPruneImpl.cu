// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "SweepAndPruneImpl.cuh"

#include <cuda/atomic>
#include <cuda/std/cmath>
#include <exception>
#include <string>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pbat {
namespace gpu {
namespace geometry {

SweepAndPruneImpl::SweepAndPruneImpl(std::size_t nPrimitives, std::size_t nOverlaps)
    : binds(nPrimitives),
      sinds(nPrimitives),
      b(nPrimitives),
      e(nPrimitives),
      mu(3),
      sigma(3),
      no(),
      o(nOverlaps)
{
}

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
                b[d][s] = cuda::std::fminf(b[d][s], x[d][inds[m][s]]);
                e[d][s] = cuda::std::fmaxf(e[d][s], x[d][inds[m][s]]);
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
    /**
     * @brief If (si,sj) are from the same simplex set, or if (si,sj) share a common vertex, they
     * should not be considered for overlap testing.
     * @param sinds Simplex vertex indices in both sets
     * @param nSimplices Number of simplices in each simplex set
     * @param si Index of first simplex in pair to test
     * @param sj Index of second simplex in pair to test
     * @return
     */
    __device__ bool AreSimplicesOverlapCandidates(GpuIndex si, GpuIndex sj) const
    {
        if ((binds[si] < nSimplices[0]) == (binds[sj] < nSimplices[0]))
            return false;
        for (auto i = 0; i < sinds.size(); ++i)
            for (auto j = 0; j < sinds.size(); ++j)
                if (sinds[i][si] == sinds[j][sj])
                    return false;
        return true;
    }

    __device__ bool AreSimplexCandidatesOverlapping(GpuIndex si, GpuIndex sj) const
    {
        return (e[axis[0]][si] >= b[axis[0]][sj]) and (b[axis[0]][si] <= e[axis[0]][sj]) and
               (e[axis[1]][si] >= b[axis[1]][sj]) and (b[axis[1]][si] <= e[axis[1]][sj]);
    }

    __device__ void operator()(GpuIndex si)
    {
        cuda::atomic_ref<GpuIndex, cuda::thread_scope_device> ano{*no};
        bool const bSwap = binds[si] >= nSimplices[0];
        for (auto sj = si + 1; (sj < nBoxes) and (e[saxis][si] >= b[saxis][sj]); ++sj)
        {
            if (not AreSimplicesOverlapCandidates(si, sj))
                continue;
            if (not AreSimplexCandidatesOverlapping(si, sj))
                continue;

            GpuIndex k = ano++;
            if (k >= nOverlapCapacity)
            {
                ano.store(nOverlapCapacity);
                break;
            }

            if (not bSwap)
            {
                o[k] = {binds[si], binds[sj] - nSimplices[0]};
            }
            else
            {
                o[k] = {binds[sj], binds[si] - nSimplices[0]};
            }
        }
    }

    GpuIndex* binds;
    std::array<GpuIndex*, 4> sinds;
    std::array<GpuIndex, 2> nSimplices;
    std::array<GpuScalar*, 3> b, e;
    GpuIndex saxis;
    std::array<GpuIndex, 2> axis;
    GpuIndex* no;
    SweepAndPruneImpl::OverlapType* o;
    GpuIndex nBoxes;
    GpuIndex nOverlapCapacity;
};

void SweepAndPruneImpl::SortAndSweep(
    PointsImpl const& P,
    SimplicesImpl const& S1,
    SimplicesImpl const& S2,
    GpuScalar expansion)
{
    auto const nBoxes = S1.NumberOfSimplices() + S2.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < nBoxes)
    {
        std::string const what = "Allocated memory for " +
                                 std::to_string(NumberOfAllocatedBoxes()) +
                                 " boxes, but received " + std::to_string(nBoxes) + " simplices.";
        throw std::invalid_argument(what);
    }

    // 0. Preprocess internal data
    thrust::fill(mu.Data(), mu.Data() + mu.Size(), 0.f);
    thrust::fill(sigma.Data(), sigma.Data() + sigma.Size(), 0.f);
    no = 0;
    thrust::sequence(thrust::device, binds.Data(), binds.Data() + nBoxes);

    // 1. Compute bounding boxes of S1 and S2
    for (auto m = 0; m < 4; ++m)
    {
        thrust::copy(thrust::device, S1.inds[m].begin(), S1.inds[m].end(), sinds[m].begin());
        thrust::copy(
            thrust::device,
            S2.inds[m].begin(),
            S2.inds[m].end(),
            sinds[m].begin() + S1.NumberOfSimplices());
    }
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(S1.NumberOfSimplices()),
        FComputeAabb{
            P.x.Raw(),
            sinds.Raw(),
            static_cast<int>(S1.eSimplexType),
            b.Raw(),
            e.Raw(),
            expansion});
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(S1.NumberOfSimplices()),
        thrust::make_counting_iterator(nBoxes),
        FComputeAabb{
            P.x.Raw(),
            sinds.Raw(),
            static_cast<int>(S2.eSimplexType),
            b.Raw(),
            e.Raw(),
            expansion});

    // 2. Compute mean and variance of bounding box centers
    FComputeMean fComputeMean{b.Raw(), e.Raw(), mu.Raw(), nBoxes};
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nBoxes),
        fComputeMean);
    FComputeVariance fComputeVariance{b.Raw(), e.Raw(), mu.Raw(), sigma.Raw(), nBoxes};
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nBoxes),
        fComputeVariance);

    // 3. Sort bounding boxes along largest variance axis
    auto sigmaPtr        = sigma.Data();
    GpuIndex const saxis = (sigmaPtr[0] > sigmaPtr[1]) ? (sigmaPtr[0] > sigmaPtr[2] ? 0 : 2) :
                                                         (sigmaPtr[1] > sigmaPtr[2] ? 1 : 2);
    std::array<GpuIndex, 2> const axis = {(saxis + 1) % 3, (saxis + 2) % 3};
    auto zip                           = thrust::make_zip_iterator(
        b[axis[0]].begin(),
        b[axis[1]].begin(),
        e[saxis].begin(),
        e[axis[0]].begin(),
        e[axis[1]].begin(),
        sinds[0].begin(),
        sinds[1].begin(),
        sinds[2].begin(),
        sinds[3].begin(),
        binds.Data());
    thrust::sort_by_key(thrust::device, b[saxis].begin(), b[saxis].begin() + nBoxes, zip);

    // 4. Sweep to find overlaps
    FSweep fSweep{
        binds.Raw(),
        sinds.Raw(),
        {S1.NumberOfSimplices(), S2.NumberOfSimplices()},
        b.Raw(),
        e.Raw(),
        saxis,
        axis,
        no.Raw(),
        o.Raw(),
        nBoxes,
        static_cast<GpuIndex>(o.Size())};
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nBoxes),
        fSweep);
}

std::size_t SweepAndPruneImpl::NumberOfAllocatedBoxes() const
{
    return binds.Size();
}

std::size_t SweepAndPruneImpl::NumberOfAllocatedOverlaps() const
{
    return o.Size();
}

thrust::host_vector<SweepAndPruneImpl::OverlapType> SweepAndPruneImpl::Overlaps() const
{
    GpuIndex const nOverlaps = no;
    thrust::host_vector<OverlapType> overlaps{o.Data(), o.Data() + nOverlaps};
    return overlaps;
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include "pbat/common/Hash.h"

#include <doctest/doctest.h>
#include <unordered_set>

TEST_CASE("[gpu][geometry] Sweep and prune")
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
    using OverlapType = gpu::geometry::SweepAndPruneImpl::OverlapType;
    struct Hash
    {
        std::size_t operator()(OverlapType const& overlap) const
        {
            return common::HashCombine(overlap.first, overlap.second);
        }
    };
    using OverlapSetType = std::unordered_set<OverlapType, Hash>;
    OverlapSetType overlapsExpected{{{0, 0}, {1, 0}}};
    gpu::geometry::PointsImpl P(V);
    gpu::geometry::SimplicesImpl S1(E1);
    gpu::geometry::SimplicesImpl S2(F2);
    // Act
    gpu::geometry::SweepAndPruneImpl sap(4, 2);
    sap.SortAndSweep(P, S1, S2);
    // Assert
    thrust::host_vector<OverlapType> overlaps = sap.Overlaps();
    for (OverlapType overlap : overlaps)
    {
        auto it                             = overlapsExpected.find(overlap);
        bool const bExpectedOverlapDetected = it != overlapsExpected.end();
        CHECK(bExpectedOverlapDetected);
        overlapsExpected.erase(it);
    }
    CHECK(overlapsExpected.empty());
}
