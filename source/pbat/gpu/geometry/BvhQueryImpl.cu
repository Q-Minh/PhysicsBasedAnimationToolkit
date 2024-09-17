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
    if (NumberOfSimplices() < n)
    {
        std::string const what = "Allocated memory for " + std::to_string(NumberOfSimplices()) +
                                 " simplices, but received " + std::to_string(n) + " simplices.";
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
    if (NumberOfSimplices() < nQueries)
    {
        std::string const what = "Allocated memory for " + std::to_string(NumberOfSimplices()) +
                                 " query simplices, but received " + std::to_string(nQueries) +
                                 " query simplices.";
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
            S1.eSimplexType,
            simplex.Raw(),
            S1.inds.Raw(),
            b.Raw(),
            e.Raw(),
            S2.eSimplexType,
            bvh.simplex.Raw(),
            S2.inds.Raw(),
            bvh.b.Raw(),
            bvh.e.Raw(),
            bvh.child.Raw(),
            static_cast<GpuIndex>(leafBegin),
            overlaps.Raw()});
}

void BvhQueryImpl::DetectContactPairsFromOverlaps(
    PointsImpl const& P,
    SimplicesImpl const& S1,
    SimplicesImpl const& S2,
    BodiesImpl const& B1,
    BodiesImpl const& B2,
    BvhImpl const& bvh,
    GpuScalar dhat,
    GpuScalar dzero)
{
    if (S1.eSimplexType != SimplicesImpl::ESimplexType::Vertex)
    {
        std::string const what = "Only vertex simplices are supported as the query simplices";
        throw std::invalid_argument(what);
    }
    if (S2.eSimplexType != SimplicesImpl::ESimplexType::Triangle)
    {
        std::string const what = "Only triangle simplices are supported as the target simplices";
        throw std::invalid_argument(what);
    }
    auto const nQueries = S1.NumberOfSimplices();
    if (NumberOfSimplices() < nQueries)
    {
        std::string const what = "Allocated memory for " + std::to_string(NumberOfSimplices()) +
                                 " query simplices, but received " + std::to_string(nQueries) +
                                 " query simplices.";
        throw std::invalid_argument(what);
    }
    neighbours.Clear();
    auto const leafBegin = static_cast<GpuIndex>(bvh.simplex.Size() - 1);
    thrust::for_each(
        thrust::device,
        overlaps.Begin(),
        overlaps.End(),
        BvhQueryImplKernels::FContactPairs{
            P.x.Raw(),
            B1.body.Raw(),
            S1.inds.Raw(),
            dhat,
            dzero,
            B2.body.Raw(),
            S2.inds.Raw(),
            bvh.simplex.Raw(),
            bvh.b.Raw(),
            bvh.e.Raw(),
            bvh.child.Raw(),
            leafBegin,
            neighbours.Raw()});
}

std::size_t BvhQueryImpl::NumberOfSimplices() const
{
    return simplex.Size();
}

std::size_t BvhQueryImpl::NumberOfAllocatedOverlaps() const
{
    return overlaps.Capacity();
}

std::size_t BvhQueryImpl::NumberOfAllocatedNeighbours() const
{
    return neighbours.Capacity();
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>

namespace pbat {
namespace gpu {
namespace geometry {
namespace test {

template <class MatrixType>
auto hstack(auto A1, auto A2)
{
    MatrixType A(A1.rows(), A1.cols() + A2.cols());
    A << A1, A2;
    return A;
};

} // namespace test
} // namespace geometry
} // namespace gpu
} // namespace pbat

TEST_CASE("[gpu][geometry] BvhQueryImpl")
{
    using namespace pbat;
    using namespace pbat::gpu::geometry::test;
    // Arrange

    // Cube mesh
    GpuMatrixX P(3, 8);
    GpuIndexMatrixX T(4, 5);
    GpuIndexMatrixX F(3, 12);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
         1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
         4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // clang-format on

    // Duplicate mesh into 2 bodies and translate 2nd mesh to get a single overlapping vertex (15)
    // of body 2 into body 1. We make it so that overlapping vertex 15 should have nearest face 0
    // from body 1. Similarly, vertex 0 of body 1 will overlap body 2.
    Eigen::Vector<GpuScalar, 3> d{
        -GpuScalar{0.5},
        -GpuScalar{1} + GpuScalar{1e-5},
        -GpuScalar{1} + GpuScalar{1e-4}};
    T                 = hstack<GpuIndexMatrixX>(T, T.array() + P.cols());
    F                 = hstack<GpuIndexMatrixX>(F, F.array() + P.cols());
    P                 = hstack<GpuMatrixX>(P, P.colwise() + d);
    GpuIndexMatrixX V = GpuIndexVectorX::LinSpaced(P.cols(), 0, static_cast<GpuIndex>(P.cols() - 1))
                            .reshaped(1, P.cols());
    GpuIndexMatrixX BV = hstack<GpuIndexMatrixX>(
        GpuIndexVectorX::Zero(V.cols() / 2).reshaped(1, V.cols() / 2),
        GpuIndexVectorX::Ones(V.cols() / 2).reshaped(1, V.cols() / 2));
    GpuIndexMatrixX BF = hstack<GpuIndexMatrixX>(
        GpuIndexVectorX::Zero(F.cols() / 2).reshaped(1, F.cols() / 2),
        GpuIndexVectorX::Ones(F.cols() / 2).reshaped(1, F.cols() / 2));
    using gpu::geometry::BodiesImpl;
    using gpu::geometry::BvhImpl;
    using gpu::geometry::BvhQueryImpl;
    using gpu::geometry::PointsImpl;
    using gpu::geometry::SimplicesImpl;
    auto constexpr nExpectedOverlaps                     = 2;
    auto constexpr nFalseOverlaps                        = 7;
    auto constexpr nExpectedNearestNeighbours            = 2;
    auto constexpr nFalseNearestNeighbours               = 7;
    GpuIndex constexpr expectedOverlappingVertex[2]      = {15, 0};
    GpuIndex constexpr expectedOverlappingTetrahedron[2] = {0, 8};
    GpuIndex constexpr expectedNearestVertex[2]          = {15, 0};
    GpuIndex constexpr expectedNearestTriangle[2]        = {0, 17};

    Eigen::Vector<GpuScalar, 3> const min{GpuScalar{0}, GpuScalar{0}, GpuScalar{0}};
    Eigen::Vector<GpuScalar, 3> const max{GpuScalar{1}, GpuScalar{1}, GpuScalar{1}};
    GpuScalar constexpr expansion{0};
    GpuScalar constexpr dhat{1};
    GpuScalar constexpr dzero{0};

    PointsImpl PG(P);
    SimplicesImpl VG(V);
    SimplicesImpl FG(F);
    SimplicesImpl TG(T);
    BodiesImpl BVG(BV.reshaped());
    BodiesImpl BFG(BF.reshaped());

    BvhImpl Tbvh(T.cols(), nExpectedOverlaps + nFalseOverlaps);
    BvhImpl Fbvh(F.cols(), 0);
    BvhQueryImpl Vquery(
        V.cols(),
        nExpectedOverlaps,
        nExpectedNearestNeighbours + nFalseNearestNeighbours);

    Tbvh.Build(PG, TG, min, max, expansion);
    Fbvh.Build(PG, FG, min, max, expansion);
    Vquery.Build(PG, VG, min, max, expansion);

    // Act
    Vquery.DetectOverlaps(PG, VG, TG, Tbvh);
    Vquery.DetectContactPairsFromOverlaps(PG, VG, FG, BVG, BFG, Fbvh, dhat, dzero);

    // Assert
    auto const overlaps = Vquery.overlaps.Get();
    CHECK_EQ(overlaps.size(), nExpectedOverlaps);
    bool const bOverlappingVerticesFound = (overlaps[0].first == expectedOverlappingVertex[0] and
                                            overlaps[1].first == expectedOverlappingVertex[1]) or
                                           (overlaps[0].first == expectedOverlappingVertex[1] and
                                            overlaps[1].first == expectedOverlappingVertex[0]);
    CHECK(bOverlappingVerticesFound);
    bool const bOverlappingTetrahedraFound =
        (overlaps[0].second == expectedOverlappingTetrahedron[0] and
         overlaps[1].second == expectedOverlappingTetrahedron[1]) or
        (overlaps[0].second == expectedOverlappingTetrahedron[1] and
         overlaps[1].second == expectedOverlappingTetrahedron[0]);
    CHECK(bOverlappingTetrahedraFound);
    auto const nearestNeighbours = Vquery.neighbours.Get();
    CHECK_EQ(nearestNeighbours.size(), nExpectedNearestNeighbours);
    bool const bNearestVerticesFound = (nearestNeighbours[0].first == expectedNearestVertex[0] and
                                        nearestNeighbours[1].first == expectedNearestVertex[1]) or
                                       (nearestNeighbours[0].first == expectedNearestVertex[1] and
                                        nearestNeighbours[1].first == expectedNearestVertex[0]);
    CHECK(bNearestVerticesFound);
    for (auto const& [v, f] : nearestNeighbours)
    {
        // Nearest neighbour contact pairs should not come from the same body, i.e. self-contact is
        // not supported as of now.
        CHECK_NE(BV(0, v), BF(0, f));
    }
    bool const bNearestTrianglesFound =
        (nearestNeighbours[0].second == expectedNearestTriangle[0] and
         nearestNeighbours[1].second == expectedNearestTriangle[1]) or
        (nearestNeighbours[0].second == expectedNearestTriangle[1] and
         nearestNeighbours[1].second == expectedNearestTriangle[0]);
    CHECK(bNearestTrianglesFound);
}