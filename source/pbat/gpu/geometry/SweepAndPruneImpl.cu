// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "SweepAndPruneImpl.cuh"
#include "SweepAndPruneImplKernels.cuh"

#include <exception>
#include <string>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
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
      overlaps(nOverlaps)
{
}

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
    overlaps.Clear();
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
        SweepAndPruneImplKernels::FComputeAabb{
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
        SweepAndPruneImplKernels::FComputeAabb{
            P.x.Raw(),
            sinds.Raw(),
            static_cast<int>(S2.eSimplexType),
            b.Raw(),
            e.Raw(),
            expansion});

    // 2. Compute mean and variance of bounding box centers
    SweepAndPruneImplKernels::FComputeMean fComputeMean{b.Raw(), e.Raw(), mu.Raw(), nBoxes};
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nBoxes),
        fComputeMean);
    SweepAndPruneImplKernels::FComputeVariance
        fComputeVariance{b.Raw(), e.Raw(), mu.Raw(), sigma.Raw(), nBoxes};
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
    SweepAndPruneImplKernels::FSweep fSweep{
        binds.Raw(),
        sinds.Raw(),
        {S1.NumberOfSimplices(), S2.NumberOfSimplices()},
        b.Raw(),
        e.Raw(),
        saxis,
        axis,
        nBoxes,
        overlaps.Raw()};
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
    return overlaps.Capacity();
}

std::vector<typename SweepAndPruneImpl::OverlapType> SweepAndPruneImpl::Overlaps() const
{
    return overlaps.Get();
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
    std::vector<OverlapType> overlaps = sap.Overlaps();
    // Assert
    for (OverlapType overlap : overlaps)
    {
        auto it                             = overlapsExpected.find(overlap);
        bool const bExpectedOverlapDetected = it != overlapsExpected.end();
        CHECK(bExpectedOverlapDetected);
        overlapsExpected.erase(it);
    }
    CHECK(overlapsExpected.empty());
}
