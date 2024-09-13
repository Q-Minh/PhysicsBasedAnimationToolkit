// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "BvhQueryImpl.cuh"
#include "BvhQueryImplKernels.cuh"

#include <exception>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pbat {
namespace gpu {
namespace geometry {

BvhQueryImpl::BvhQueryImpl(
    std::size_t nPrimitives,
    std::size_t nOverlaps,
    std::size_t nNearestNeighbours)
    : simplex(nPrimitives),
      morton(nPrimitives),
      b(nPrimitives),
      e(nPrimitives),
      overlaps(nOverlaps),
      neighbours(nNearestNeighbours)
{
}

void BvhQueryImpl::Build(
    PointsImpl const& P,
    SimplicesImpl const& S,
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max,
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
    // Compute bounding boxes
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        BvhQueryImplKernels::FComputeAabb{
            P.x.Raw(),
            S.inds.Raw(),
            static_cast<int>(S.eSimplexType),
            b.Raw(),
            e.Raw(),
            expansion});
    // Compute simplex morton codes
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        BvhQueryImplKernels::FComputeMortonCode{
            {min(0), min(1), min(2)},
            {max(0) - min(0), max(1) - min(1), max(2) - min(2)},
            b.Raw(),
            e.Raw(),
            morton.Raw()});
    // Sort simplices+boxes by morton codes to try and improve data locality in future queries
    thrust::sequence(thrust::device, simplex.Data(), simplex.Data() + simplex.Size());
    auto zip = thrust::make_zip_iterator(
        b[0].begin(),
        b[1].begin(),
        b[2].begin(),
        e[0].begin(),
        e[1].begin(),
        e[2].begin(),
        simplex.Data());
    thrust::stable_sort_by_key(thrust::device, morton.Data(), morton.Data() + n, zip);
}

void BvhQueryImpl::DetectOverlaps(
    PointsImpl const& P,
    SimplicesImpl const& S1,
    SimplicesImpl const& S2,
    BvhImpl const& bvh)
{
    auto const nQueries = S1.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < nQueries)
    {
        std::string const what =
            "Allocated memory for " + std::to_string(NumberOfAllocatedBoxes()) +
            " boxes, but received " + std::to_string(nQueries) + " query simplices.";
        throw std::invalid_argument(what);
    }
    overlaps.Clear();
    auto const leafBegin = bvh.simplex.Size() - 1;
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nQueries),
        BvhQueryImplKernels::FDetectOverlaps{
            P.x.Raw(),
            simplex.Raw(),
            S1.inds.Raw(),
            b.Raw(),
            e.Raw(),
            bvh.simplex.Raw(),
            S2.inds.Raw(),
            bvh.b.Raw(),
            bvh.e.Raw(),
            bvh.child.Raw(),
            static_cast<GpuIndex>(leafBegin),
            overlaps.Raw()});
}

void BvhQueryImpl::DetectNearestNeighbours(
    PointsImpl const& P,
    SimplicesImpl const& S1,
    SimplicesImpl const& S2,
    BvhImpl const& bvh,
    GpuScalar dhat)
{
    if (S1.eSimplexType != SimplicesImpl::ESimplexType::Vertex)
    {
        std::string const what = "Only vertex simplices are supported as the query simplices";
        throw std::invalid_argument(what);
    }
    auto const nQueries = S1.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < nQueries)
    {
        std::string const what =
            "Allocated memory for " + std::to_string(NumberOfAllocatedBoxes()) +
            " boxes, but received " + std::to_string(nQueries) + " query simplices.";
        throw std::invalid_argument(what);
    }
}

std::size_t BvhQueryImpl::NumberOfAllocatedBoxes() const
{
    return simplex.Size();
}

} // namespace geometry
} // namespace gpu
} // namespace pbat