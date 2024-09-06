// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "BvhImpl.cuh"
#include "BvhImplKernels.cuh"

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

BvhImpl::BvhImpl(std::size_t nPrimitives, std::size_t nOverlaps)
    : simplex(nPrimitives),
      morton(nPrimitives),
      child(nPrimitives - 1),
      parent(2 * nPrimitives - 1),
      rightmost(nPrimitives - 1),
      b(2 * nPrimitives - 1),
      e(2 * nPrimitives - 1),
      visits(nPrimitives - 1),
      no(0),
      o(nOverlaps)
{
    thrust::fill(thrust::device, parent.Data(), parent.Data() + parent.Size(), GpuIndex{-1});
}

void BvhImpl::Build(PointsImpl const& P, SimplicesImpl const& S, GpuScalar expansion)
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
    thrust::fill(thrust::device, visits.Raw(), visits.Raw() + visits.Size(), GpuIndex{0});

    // 1. Construct leaf node (i.e. simplex) bounding boxes
    auto const leafBegin = n - 1;
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        BvhImplKernels::FLeafBoundingBoxes{
            P.x.Raw(),
            S.inds.Raw(),
            static_cast<int>(S.eSimplexType),
            b.Raw(),
            e.Raw(),
            leafBegin,
            expansion});

    // 2. Compute Morton codes for each leaf node (i.e. simplex)
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        BvhImplKernels::FComputeMortonCode{b.Raw(), e.Raw(), morton.Raw(), leafBegin});

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
        BvhImplKernels::FGenerateHierarchy{
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
        BvhImplKernels::FInternalNodeBoundingBoxes{
            parent.Raw(),
            child.Raw(),
            b.Raw(),
            e.Raw(),
            visits.Raw()});
}

void BvhImpl::DetectSelfOverlaps(SimplicesImpl const& S)
{
    auto const n = S.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < n)
    {
        std::string const what = "Allocated memory for " +
                                 std::to_string(NumberOfAllocatedBoxes()) +
                                 " boxes, but received " + std::to_string(n) + " simplices.";
        throw std::invalid_argument(what);
    }
    auto const leafBegin = n - 1;
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(n - 1),
        thrust::make_counting_iterator(2 * n - 1),
        BvhImplKernels::FDetectSelfOverlaps{
            simplex.Raw(),
            S.inds.Raw(),
            child.Raw(),
            rightmost.Raw(),
            b.Raw(),
            e.Raw(),
            leafBegin,
            no.Raw(),
            o.Raw(),
            static_cast<GpuIndex>(o.Size())});
}

std::size_t BvhImpl::NumberOfAllocatedBoxes() const
{
    return simplex.Size();
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>
#include <unordered_set>

TEST_CASE("[gpu][geometry] Sweep and prune")
{
    using namespace pbat;
    // Arrange
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
    gpu::geometry::PointsImpl P(V);
    gpu::geometry::SimplicesImpl S(C);
    // Act
    gpu::geometry::BvhImpl bvh(S.NumberOfSimplices(), S.NumberOfSimplices());
    bvh.Build(P, S);
    // Assert
}
