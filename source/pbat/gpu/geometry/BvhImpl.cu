// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "BvhImpl.cuh"
#include "BvhImplKernels.cuh"
#include "pbat/common/Eigen.h"

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
      overlaps(nOverlaps)
{
    thrust::fill(thrust::device, parent.Data(), parent.Data() + parent.Size(), GpuIndex{-1});
}

void BvhImpl::Build(
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

    // 0. Reset intermediate data
    thrust::fill(thrust::device, visits.Raw(), visits.Raw() + visits.Size(), GpuIndex{0});

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
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        BvhImplKernels::FComputeMortonCode{
            {min(0), min(1), min(2)},
            {max(0) - min(0), max(1) - min(1), max(2) - min(2)},
            b.Raw(),
            e.Raw(),
            morton.Raw(),
            leafBegin});

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
    overlaps.Clear();
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
            overlaps.Raw()});
}

std::size_t BvhImpl::NumberOfAllocatedBoxes() const
{
    return simplex.Size();
}

Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> BvhImpl::Min() const
{
    using pbat::common::ToEigen;
    return ToEigen(b.Get()).reshaped(b.Size(), b.Dimensions());
}

Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> BvhImpl::Max() const
{
    using pbat::common::ToEigen;
    return ToEigen(e.Get()).reshaped(e.Size(), e.Dimensions());
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> BvhImpl::SimplexOrdering() const
{
    using pbat::common::ToEigen;
    return ToEigen(simplex.Get());
}

Eigen::Vector<typename BvhImpl::MortonCodeType, Eigen::Dynamic> BvhImpl::MortonCodes() const
{
    using pbat::common::ToEigen;
    return ToEigen(morton.Get());
}

Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> BvhImpl::Child() const
{
    using pbat::common::ToEigen;
    return ToEigen(child.Get()).reshaped(child.Size(), child.Dimensions());
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> BvhImpl::Parent() const
{
    using pbat::common::ToEigen;
    return ToEigen(parent.Get());
}

Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> BvhImpl::Rightmost() const
{
    using pbat::common::ToEigen;
    return ToEigen(rightmost.Get()).reshaped(rightmost.Size(), rightmost.Dimensions());
}

Eigen::Vector<GpuIndex, Eigen::Dynamic> BvhImpl::Visits() const
{
    using pbat::common::ToEigen;
    return ToEigen(visits.Get());
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <algorithm>
#include <doctest/doctest.h>
#include <unordered_set>

TEST_CASE("[gpu][geometry] BvhImpl")
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
    using gpu::geometry::BvhImpl;
    using gpu::geometry::PointsImpl;
    using gpu::geometry::SimplicesImpl;
    auto const assert_cube = [](BvhImpl const& bvh) {
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
        PointsImpl P(V);
        SimplicesImpl S(C);
        // Act
        BvhImpl bvh(S.NumberOfSimplices(), S.NumberOfSimplices());
        bvh.Build(P, S, Vmin, Vmax);
        bvh.DetectSelfOverlaps(S);
        // Assert
        assert_cube(bvh);
        auto overlaps = bvh.overlaps.Get();
        CHECK_EQ(overlaps.size(), 0ULL);
    }
    SUBCASE("Disconnected mesh")
    {
        V = V(Eigen::all, C.reshaped()).eval();
        C.resize(4, C.cols());
        C.reshaped().setLinSpaced(0, static_cast<GpuIndex>(V.cols() - 1));
        // Arrange
        PointsImpl P(V);
        SimplicesImpl S(C);
        // Because we only support overlaps between i,j s.t. i<j to prevent duplicates, we use the
        // summation identity \sum_i=1^n i = n*(n+1)/2, and remove the n occurrences where i=j.
        auto const nExpectedOverlaps =
            (S.NumberOfSimplices() * (S.NumberOfSimplices() + 1) / 2) - S.NumberOfSimplices();
        // Act
        BvhImpl bvh(S.NumberOfSimplices(), nExpectedOverlaps);
        bvh.Build(P, S, Vmin, Vmax);
        bvh.DetectSelfOverlaps(S);
        // Assert
        assert_cube(bvh);
        auto overlaps = bvh.overlaps.Get();
        CHECK_EQ(overlaps.size(), static_cast<std::size_t>(nExpectedOverlaps));
    }
}
